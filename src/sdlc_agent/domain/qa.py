"""QA-role artifacts: the test plan, its execution result and the final quality gate.

This is the weakest link in comparable open-source systems (MetaGPT ships its
QA role commented out), so the models here are deliberately strict: a test run
that did not actually execute is distinguishable from one that executed and
passed, and the quality gate cannot be approved without evidence.

The QA agent derives its plan from the PRD's acceptance criteria and the code
change, runs pytest inside the sandbox, and reports back. A failure returns the
run to the developer node, bounded by a retry limit (HARNESS_SPEC.md
section 2.1).
"""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sdlc_agent.domain.base import Artifact, ArtifactKind, Verdict, reject_blank


class TestPlanItem(BaseModel):
    """One planned test, traced back to the criterion it proves.

    Attributes:
        criterion: The PRD acceptance criterion this test covers. Kept as text
            rather than an id because a criterion may be reworded during review
            and the trace must stay readable.
        description: What the test does and what it asserts.
        test_path: Where the test will live in the sandbox repository.
    """

    # These classes are named for the domain (a plan of tests, a run of tests),
    # not for pytest. The prefix makes pytest try to collect them as test
    # classes, so opt out explicitly on each one below.
    __test__ = False

    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion: str = Field(min_length=1)
    description: str = Field(min_length=1)
    test_path: str = Field(min_length=1)

    @field_validator("criterion", "description", "test_path")
    @classmethod
    def _strip(cls, value: str) -> str:
        return reject_blank(value)


class TestPlan(Artifact):
    """What the QA agent intends to verify, produced before any test runs.

    Writing the plan before execution is what makes the criteria a real gate
    rather than a post-hoc rationalization (HARNESS_SPEC.md section 5.1).

    Attributes:
        prd_id: The PRD whose acceptance criteria drive this plan.
        code_change_id: The change under test.
        items: The planned tests.
        uncovered_criteria: Acceptance criteria this plan does not test, with
            the reason. Surfacing the gap beats silently skipping it, because
            an uncovered criterion is a reviewer decision, not an agent one.
    """

    kind: ArtifactKind = ArtifactKind.TEST_PLAN

    __test__ = False

    prd_id: uuid.UUID
    code_change_id: uuid.UUID
    items: tuple[TestPlanItem, ...] = Field(min_length=1)
    uncovered_criteria: tuple[str, ...] = ()

    @field_validator("uncovered_criteria")
    @classmethod
    def _strip_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(reject_blank(item) for item in value)


class TestStatus(StrEnum):
    """Outcome of executing the test plan."""

    __test__ = False

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    NOT_RUN = "not_run"


class TestCaseResult(BaseModel):
    """Result of one test.

    Attributes:
        test_path: Test file, matching the plan item.
        test_name: Test function or class within it.
        status: Outcome. ``NOT_RUN`` is a distinct value because "we did not
            get to it" must never be silently folded into "passed".
        duration_seconds: Wall time, for spotting slow tests that will make the
            agent loop stall.
        failure_output: Captured output when not passing. Truncated by the
            caller before it enters an LLM context (HARNESS_SPEC.md section 4.1).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    __test__ = False

    test_path: str = Field(min_length=1)
    test_name: str = Field(min_length=1)
    status: TestStatus
    duration_seconds: float = Field(default=0.0, ge=0.0)
    failure_output: str = ""

    @field_validator("test_path", "test_name")
    @classmethod
    def _strip(cls, value: str) -> str:
        return reject_blank(value)


class TestRun(Artifact):
    """The executed result of a test plan.

    Attributes:
        test_plan_id: The plan that was executed.
        status: Aggregate outcome. ``NOT_RUN`` means the suite never executed
            (collection error, missing sandbox, budget exhausted), which is a
            different situation from tests that ran and failed.
        results: Per-test outcomes.
        command: The exact command run, so a human can reproduce it.
        exit_code: Raw process exit code, kept alongside the interpreted status
            so a disagreement between the two is visible.
    """

    kind: ArtifactKind = ArtifactKind.TEST_RUN

    __test__ = False

    test_plan_id: uuid.UUID
    status: TestStatus
    results: tuple[TestCaseResult, ...] = ()
    command: str = Field(min_length=1)
    exit_code: int

    @field_validator("command")
    @classmethod
    def _strip_command(cls, value: str) -> str:
        return reject_blank(value)

    @model_validator(mode="after")
    def _status_matches_results(self) -> TestRun:
        """Refuse a run whose claimed status contradicts its own results.

        A green suite with zero tests executed is the single most dangerous
        false signal this pipeline can produce: it walks straight through the
        quality gate and out to a human reviewer as "verified". Catching the
        contradiction at construction keeps it from ever becoming an artifact.
        """
        if self.status is TestStatus.PASSED:
            if not self.results:
                raise ValueError("status PASSED requires at least one executed test result")
            not_passed = [r for r in self.results if r.status is not TestStatus.PASSED]
            if not_passed:
                names = ", ".join(r.test_name for r in not_passed[:5])
                raise ValueError(
                    f"status PASSED contradicts {len(not_passed)} failing result(s): {names}"
                )
        elif self.status is TestStatus.FAILED and not any(
            r.status is not TestStatus.PASSED for r in self.results
        ):
            raise ValueError("status FAILED requires at least one non-passing result")
        return self

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.status is TestStatus.PASSED)

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if result.status is not TestStatus.PASSED)


class QualityGate(Artifact):
    """The final verdict before the code merge gate.

    Attributes:
        test_run_id: The run this verdict is based on.
        verdict: ``APPROVED`` only when tests actually ran and passed.
            ``NEEDS_HUMAN`` when the evidence is ambiguous, such as a run that
            never executed or criteria left uncovered.
        rationale: Why this verdict, in terms a reviewer can check against the
            test run. Required so an approval is never a bare boolean.
        pr_url: Populated once the branch has been pushed and the PR opened.
    """

    kind: ArtifactKind = ArtifactKind.QUALITY_GATE

    test_run_id: uuid.UUID
    verdict: Verdict
    rationale: str = Field(min_length=1)
    pr_url: str = ""

    @field_validator("rationale")
    @classmethod
    def _strip_rationale(cls, value: str) -> str:
        return reject_blank(value)

    @classmethod
    def from_evidence(cls, run: TestRun, uncovered_criteria: tuple[str, ...] = ()) -> QualityGate:
        """Derive the verdict from the run rather than letting a caller assert it.

        Keeping this rule here means the flow code cannot approve a run that
        never executed, and the reasoning stays in one place when the policy
        changes. The three cases:

        - tests ran and all passed, nothing left uncovered -> ``APPROVED``
        - tests ran and something failed -> ``REJECTED``
        - the suite never executed, or criteria went untested -> ``NEEDS_HUMAN``

        The last case is the one that matters. An unexecuted run is not a
        failure, it is missing evidence, and missing evidence goes to a human
        instead of being quietly rounded down to either verdict.
        """
        if run.status is TestStatus.NOT_RUN:
            return cls(
                test_run_id=run.artifact_id,
                verdict=Verdict.NEEDS_HUMAN,
                rationale=(
                    f"Test suite never executed (command: {run.command!r}, "
                    f"exit code {run.exit_code}); no evidence to judge."
                ),
            )

        if run.status is TestStatus.PASSED:
            if uncovered_criteria:
                listed = "; ".join(uncovered_criteria)
                return cls(
                    test_run_id=run.artifact_id,
                    verdict=Verdict.NEEDS_HUMAN,
                    rationale=(
                        f"All {run.passed_count} executed test(s) passed, but "
                        f"{len(uncovered_criteria)} acceptance criterion/criteria "
                        f"went untested: {listed}."
                    ),
                )
            return cls(
                test_run_id=run.artifact_id,
                verdict=Verdict.APPROVED,
                rationale=(
                    f"All {run.passed_count} executed test(s) passed and every "
                    f"acceptance criterion is covered."
                ),
            )

        failures = ", ".join(
            result.test_name for result in run.results if result.status is not TestStatus.PASSED
        )
        return cls(
            test_run_id=run.artifact_id,
            verdict=Verdict.REJECTED,
            rationale=(
                f"{run.failed_count} of {len(run.results)} test(s) did not pass: {failures}."
            ),
        )

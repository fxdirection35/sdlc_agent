"""Test execution: the deterministic boundary between QA's plan and its verdict.

The QA role produces a *plan*. Turning that plan into a verdict needs two things
that are not LLM work: running the suite, and deriving a gate from the result.
Keeping them out of the role nodes matters for HARNESS_SPEC.md section 2.1 -
the four-role handoff ceiling counts agent-to-agent handoffs, and a deterministic
step is not one. It also means a test failure is evidence rather than an
opinion.

P1 supplies :class:`StubExecutor`. P2 replaces it with an OpenHands-sandbox
executor; the graph does not change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from sdlc_agent.domain import ArtifactKind
from sdlc_agent.domain.qa import QualityGate, TestCaseResult, TestPlan, TestRun, TestStatus
from sdlc_agent.pipeline.roles import RoleContext
from sdlc_agent.pipeline.state import AuditEvent, PipelineState


class TestExecutor(Protocol):
    """Runs a test plan and reports what happened.

    Implementations must be honest about not running: a suite that could not
    execute returns ``TestStatus.NOT_RUN`` rather than an empty passing result.
    The whole point of this boundary is that the verdict is evidence, so an
    executor that reports green on a suite it never ran defeats it.
    """

    def execute(self, plan: TestPlan, context: RoleContext) -> TestRun: ...


@dataclass(frozen=True, slots=True)
class StubExecutor:
    """Executor whose outcome is decided by the caller, for testing the graph.

    Attributes:
        status: Outcome to report for every plan.
        failure_names: Test names to mark as failing when ``status`` is FAILED.
            Ignored otherwise.
    """

    status: TestStatus = TestStatus.PASSED
    failure_names: Sequence[str] = ()
    command: str = ".venv/Scripts/python -m pytest -q"
    exit_code: int | None = None

    def execute(self, plan: TestPlan, context: RoleContext) -> TestRun:
        code = self.exit_code
        if code is None:
            code = 0 if self.status is TestStatus.PASSED else 1

        if self.status is TestStatus.NOT_RUN:
            # Nothing ran, so there are no results. QualityGate.from_evidence
            # routes this to a human instead of rounding it to pass or fail.
            return TestRun(
                test_plan_id=plan.artifact_id,
                status=TestStatus.NOT_RUN,
                results=(),
                command=self.command,
                exit_code=code,
            )

        results = tuple(
            TestCaseResult(
                test_path=item.test_path,
                test_name=_test_name(item.test_path, index),
                status=self._status_for(_test_name(item.test_path, index)),
                duration_seconds=0.1,
            )
            for index, item in enumerate(plan.items)
        )
        # Derive the aggregate from the per-test results rather than echoing
        # self.status, so the two cannot disagree and trip the model validator.
        aggregate = (
            TestStatus.PASSED
            if results and all(result.status is TestStatus.PASSED for result in results)
            else TestStatus.FAILED
        )
        return TestRun(
            test_plan_id=plan.artifact_id,
            status=aggregate,
            results=results,
            command=self.command,
            exit_code=0 if aggregate is TestStatus.PASSED else 1,
        )

    def _status_for(self, test_name: str) -> TestStatus:
        """Per-test outcome, from ``failure_names`` when given, else ``status``.

        Honouring ``failure_names`` alone would let a stub configured with
        ``status=FAILED`` but no names report a clean run, contradicting its own
        configuration and making the retry loop look untested when it is not.
        """
        if self.failure_names:
            return TestStatus.FAILED if test_name in self.failure_names else TestStatus.PASSED
        if self.status is TestStatus.ERROR:
            return TestStatus.ERROR
        if self.status is TestStatus.FAILED:
            return TestStatus.FAILED
        return TestStatus.PASSED


def _test_name(test_path: str, index: int) -> str:
    """A stable synthetic test name for a planned item.

    The plan records where a test will live, not its function name, so the stub
    derives one. Named after the file so a failure in the report is traceable to
    the plan item it came from.
    """
    stem = test_path.rsplit("/", 1)[-1]
    if stem.endswith(".py"):
        stem = stem[:-3]
    return f"test_{stem}_{index}" if index else f"test_{stem}"


def _audit(stage: str, event: str, detail: str = "") -> tuple[AuditEvent, ...]:
    return (AuditEvent(stage=stage, event=event, detail=detail),)


def make_verify_node(executor: TestExecutor, stage: str) -> Any:
    """Build the node that runs a plan and derives the quality gate.

    Two artifacts come out of one node because they are one deterministic
    operation: the run and the verdict derived from it. Splitting them would add
    a node boundary with no decision at it.
    """

    def node(state: PipelineState) -> dict[str, Any]:
        artifacts = state.get("artifacts", {})
        plan = artifacts.get(ArtifactKind.TEST_PLAN)
        if not isinstance(plan, TestPlan):
            # Cannot verify what was never planned. This is a graph wiring bug,
            # not a test failure, so escalate rather than loop.
            return {
                "audit": _audit(stage, "no_test_plan", "QA produced no plan to execute"),
            }

        run = executor.execute(
            plan,
            RoleContext(
                requirement=state["requirement"],
                # The workspace the checkout node cloned, so the executor runs
                # the suite against real code instead of reporting NOT_RUN.
                repo_root=state.get("repo_root", ""),
            ),
        )
        gate = QualityGate.from_evidence(run, uncovered_criteria=plan.uncovered_criteria)

        return {
            "artifacts": {
                ArtifactKind.TEST_RUN: run,
                ArtifactKind.QUALITY_GATE: gate,
            },
            # Incremented here, in the flow that owns the retry bound, so no
            # role implementation can forget it.
            "qa_attempts": state.get("qa_attempts", 0) + 1,
            "audit": _audit(
                stage,
                "verified",
                f"{run.status.value}: {run.passed_count} passed, {run.failed_count} not passed; "
                f"gate -> {gate.verdict.value}",
            ),
        }

    return node

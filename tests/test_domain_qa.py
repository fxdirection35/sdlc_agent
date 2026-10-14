"""Tests for the QA-role artifacts and the evidence-to-verdict rule.

The quality gate is the last automated step before a human reviews the PR, so
these tests focus hard on the one failure that would be worst in production: a
run that never executed being reported as green.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from sdlc_agent.domain.base import Verdict
from sdlc_agent.domain.qa import (
    QualityGate,
    TestCaseResult,
    TestPlan,
    TestPlanItem,
    TestRun,
    TestStatus,
)


def make_plan_item(**overrides: object) -> TestPlanItem:
    payload: dict[str, object] = {
        "criterion": "Export produces a CSV with a header row",
        "description": "Writes two rows, asserts header and row count",
        "test_path": "tests/test_export.py",
    }
    payload.update(overrides)
    return TestPlanItem.model_validate(payload)


def make_plan(**overrides: object) -> TestPlan:
    payload: dict[str, object] = {
        "prd_id": uuid.uuid4(),
        "code_change_id": uuid.uuid4(),
        "items": (make_plan_item(),),
    }
    payload.update(overrides)
    return TestPlan.model_validate(payload)


def make_result(
    status: TestStatus = TestStatus.PASSED,
    name: str = "test_csv_shape",
    **overrides: object,
) -> TestCaseResult:
    payload: dict[str, object] = {
        "test_path": "tests/test_export.py",
        "test_name": name,
        "status": status,
        "duration_seconds": 0.42,
    }
    payload.update(overrides)
    return TestCaseResult.model_validate(payload)


def make_run(
    status: TestStatus = TestStatus.PASSED,
    results: tuple[TestCaseResult, ...] | None = None,
    **overrides: object,
) -> TestRun:
    payload: dict[str, object] = {
        "test_plan_id": uuid.uuid4(),
        "status": status,
        "results": results if results is not None else (make_result(),),
        "command": ".venv/Scripts/python -m pytest -q",
        "exit_code": 0 if status is TestStatus.PASSED else 1,
    }
    payload.update(overrides)
    return TestRun.model_validate(payload)


class TestTestPlanItem:
    @pytest.mark.parametrize("field", ["criterion", "description", "test_path"])
    def test_rejects_blank_fields(self, field: str) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_plan_item(**{field: "  "})


class TestTestPlan:
    def test_valid_plan(self) -> None:
        plan = make_plan()

        assert plan.kind.value == "test_plan"
        assert len(plan.items) == 1
        assert plan.uncovered_criteria == ()

    def test_requires_at_least_one_item(self) -> None:
        """An empty plan would let a change through with nothing verified."""
        with pytest.raises(ValidationError):
            make_plan(items=())

    def test_records_uncovered_criteria(self) -> None:
        """A gap in coverage is recorded, not silently dropped."""
        plan = make_plan(uncovered_criteria=("Works with 1M rows: needs perf harness",))

        assert plan.uncovered_criteria[0].startswith("Works with 1M rows")

    def test_rejects_blank_uncovered_entry(self) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_plan(uncovered_criteria=("real gap", ""))

    def test_json_round_trip_preserves_items(self) -> None:
        original = make_plan(items=(make_plan_item(), make_plan_item(test_path="tests/b.py")))

        restored = TestPlan.from_json(original.to_json())

        assert len(restored.items) == 2
        assert restored.items[1].test_path == "tests/b.py"


class TestTestCaseResult:
    def test_defaults_are_safe(self) -> None:
        result = TestCaseResult.model_validate(
            {"test_path": "tests/a.py", "test_name": "test_x", "status": "passed"}
        )

        assert result.duration_seconds == 0.0
        assert result.failure_output == ""

    def test_rejects_negative_duration(self) -> None:
        with pytest.raises(ValidationError):
            make_result(duration_seconds=-1.0)

    def test_not_run_is_a_distinct_status(self) -> None:
        """Never conflated with passed: 'we did not get to it' is not success."""
        result = make_result(status=TestStatus.NOT_RUN)

        assert result.status is TestStatus.NOT_RUN

    def test_not_run_and_passed_are_distinct_members(self) -> None:
        """The enum must keep them apart, or the gate logic in qa.py collapses.

        Asserted over the whole enum rather than by comparing two literals:
        a distinct value per member is what prevents aliases from folding
        NOT_RUN into PASSED, which would let an unexecuted suite read as green.
        """
        values = [member.value for member in TestStatus]

        assert len(values) == len(set(values)), "enum members must have distinct values"
        assert TestStatus("not_run") is TestStatus.NOT_RUN
        assert TestStatus("passed") is TestStatus.PASSED
        assert len(list(TestStatus)) == 4

    def test_rejects_unknown_status(self) -> None:
        """Validated through model_validate, which is where untrusted JSON lands."""
        with pytest.raises(ValidationError):
            TestCaseResult.model_validate(
                {
                    "test_path": "tests/a.py",
                    "test_name": "test_x",
                    "status": "flaky",
                }
            )


class TestTestRunConsistency:
    """The model validator that keeps a run's status honest about its results."""

    def test_passing_run_with_all_passed_results(self) -> None:
        run = make_run(results=(make_result(), make_result(name="test_second")))

        assert run.status is TestStatus.PASSED
        assert run.passed_count == 2
        assert run.failed_count == 0

    def test_rejects_passed_status_with_no_results(self) -> None:
        """The dangerous false green: a suite that never produced a result."""
        with pytest.raises(ValidationError, match="at least one executed test result"):
            make_run(results=())

    def test_rejects_passed_status_with_a_failure(self) -> None:
        with pytest.raises(ValidationError, match="contradicts") as excinfo:
            make_run(results=(make_result(), make_result(status=TestStatus.FAILED, name="bad")))

        assert "bad" in str(excinfo.value)

    def test_rejects_passed_status_when_a_test_did_not_run(self) -> None:
        """NOT_RUN inside a green suite is still not green."""
        with pytest.raises(ValidationError, match="contradicts"):
            make_run(
                results=(make_result(), make_result(status=TestStatus.NOT_RUN, name="skipped"))
            )

    def test_error_message_truncates_long_failure_lists(self) -> None:
        """Names are capped at five so the error stays readable in a log."""
        results = tuple(
            make_result(status=TestStatus.FAILED, name=f"test_fail_{i}") for i in range(8)
        )

        with pytest.raises(ValidationError, match=r"8 failing result\(s\)"):
            make_run(results=results)

    def test_rejects_failed_status_with_all_passing_results(self) -> None:
        """The mirror case: claiming failure while every test passed."""
        with pytest.raises(ValidationError, match="at least one non-passing"):
            make_run(status=TestStatus.FAILED, results=(make_result(),), exit_code=1)

    def test_failed_run_with_a_failure_is_accepted(self) -> None:
        run = make_run(
            status=TestStatus.FAILED,
            results=(make_result(), make_result(status=TestStatus.FAILED, name="test_broken")),
        )

        assert run.passed_count == 1
        assert run.failed_count == 1

    def test_not_run_status_allows_empty_results(self) -> None:
        """Collection errors and missing sandboxes produce no results at all."""
        run = make_run(status=TestStatus.NOT_RUN, results=(), exit_code=2)

        assert run.passed_count == 0
        assert run.failed_count == 0

    def test_rejects_blank_command(self) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_run(command="  ")

    def test_json_round_trip_preserves_status(self) -> None:
        original = make_run(
            status=TestStatus.FAILED,
            results=(make_result(status=TestStatus.FAILED),),
            exit_code=1,
        )

        restored = TestRun.from_json(original.to_json())

        assert restored.status is TestStatus.FAILED
        assert restored.exit_code == 1


class TestQualityGateFromEvidence:
    """The verdict is derived from evidence, never asserted by a caller."""

    def test_all_passed_and_fully_covered_approves(self) -> None:
        run = make_run(results=(make_result(), make_result(name="test_b")))

        gate = QualityGate.from_evidence(run)

        assert gate.verdict is Verdict.APPROVED
        assert "2 executed test(s) passed" in gate.rationale
        assert "every acceptance criterion is covered" in gate.rationale

    def test_not_run_needs_human(self) -> None:
        """Missing evidence is escalated, not rounded down to pass or fail."""
        run = make_run(status=TestStatus.NOT_RUN, results=(), exit_code=2)

        gate = QualityGate.from_evidence(run)

        assert gate.verdict is Verdict.NEEDS_HUMAN
        assert "never executed" in gate.rationale
        assert run.command in gate.rationale

    def test_passed_with_uncovered_criteria_needs_human(self) -> None:
        """Green tests do not excuse a criterion nobody tested."""
        run = make_run()

        gate = QualityGate.from_evidence(run, uncovered_criteria=("Handles 1M rows",))

        assert gate.verdict is Verdict.NEEDS_HUMAN
        assert "Handles 1M rows" in gate.rationale
        assert "went untested" in gate.rationale

    def test_failure_rejects_and_names_the_tests(self) -> None:
        run = make_run(
            status=TestStatus.FAILED,
            results=(make_result(), make_result(status=TestStatus.FAILED, name="test_broken")),
            exit_code=1,
        )

        gate = QualityGate.from_evidence(run)

        assert gate.verdict is Verdict.REJECTED
        assert "test_broken" in gate.rationale

    def test_uncovered_criteria_ignored_when_tests_failed(self) -> None:
        """Rejection already routes back to the developer; no need to escalate too."""
        run = make_run(
            status=TestStatus.FAILED,
            results=(make_result(status=TestStatus.FAILED),),
            exit_code=1,
        )

        gate = QualityGate.from_evidence(run, uncovered_criteria=("Handles 1M rows",))

        assert gate.verdict is Verdict.REJECTED

    def test_gate_points_at_the_run_it_judged(self) -> None:
        run = make_run()

        gate = QualityGate.from_evidence(run)

        assert gate.test_run_id == run.artifact_id
        assert gate.kind.value == "quality_gate"

    def test_pr_url_defaults_empty(self) -> None:
        gate = QualityGate.from_evidence(make_run())

        assert gate.pr_url == ""

    def test_rejects_blank_rationale(self) -> None:
        """An approval with no reasoning is not reviewable."""
        with pytest.raises(ValidationError, match="non-whitespace"):
            QualityGate.model_validate(
                {
                    "kind": "quality_gate",
                    "test_run_id": uuid.uuid4(),
                    "verdict": "approved",
                    "rationale": "  ",
                }
            )

    def test_json_round_trip(self) -> None:
        original = QualityGate.from_evidence(make_run())

        restored = QualityGate.from_json(original.to_json())

        assert restored.verdict is Verdict.APPROVED
        assert restored.rationale == original.rationale

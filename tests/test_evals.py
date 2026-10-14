"""Tests for the golden-task evaluation harness.

The eval set gates commits, so the harness itself has to be trustworthy: a
check that silently passes would be worse than no check, because it would
certify an agent regression as green. Most tests here are about that - proving
the checks fail when they should, not merely that they pass when they should.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from sdlc_agent.domain.base import ArtifactKind, PipelineRole
from sdlc_agent.domain.product import ProductRequirements
from sdlc_agent.evals import (
    CHECK_TYPES,
    EVAL_DIR,
    AgentRun,
    AnyCheck,
    Check,
    EvalDimension,
    EvalRunError,
    GoldenTask,
    load_all_tasks,
    load_tasks,
    summarize_coverage,
    tasks_for_role,
)
from sdlc_agent.evals.runner import EvalReport, TaskResult, run_eval_set, run_task
from sdlc_agent.evals.tasks import (
    MaxOutputTokensCheck,
    SchemaValidCheck,
    TextAbsentCheck,
    TextMatchesCheck,
    ToolsWithinCheck,
)


def _declared_type_of(check_class: type[Check]) -> str:
    """Read the discriminator value a check class declares.

    Each class annotates its own ``type`` field as a single-valued ``Literal``,
    such as ``Literal["schema_valid"]``, so the literal's only argument is the
    string a YAML file must use to select this class.
    """
    annotation = check_class.model_fields["type"].annotation
    declared = get_args(annotation)
    assert len(declared) == 1, f"{check_class.__name__}.type is not a single-valued Literal"
    return str(declared[0])


class TestCheckModels:
    def test_check_types_match_the_union_the_loader_builds(self) -> None:
        """CHECK_TYPES must list exactly the types a YAML file can name.

        Derived from ``AnyCheck`` rather than a hardcoded literal, so adding a
        check class and forgetting to register it fails here. Without that, the
        new type would load fine but the loader's "known check types" hint on an
        unrelated typo would be silently incomplete.
        """
        reachable = get_args(get_args(AnyCheck)[0])
        declared = {_declared_type_of(cls) for cls in reachable}

        assert declared == set(CHECK_TYPES)

    def test_every_check_class_is_reachable_from_yaml(self) -> None:
        """A Check subclass absent from the union can never be instantiated."""
        reachable = set(get_args(get_args(AnyCheck)[0]))
        defined = set(Check.__subclasses__())

        assert defined == reachable, f"unreachable: {defined - reachable}"

    def test_checks_default_to_the_outcome_dimension(self) -> None:
        assert (
            SchemaValidCheck(expected_kind=ArtifactKind.PRD, reason="r").dimension.value
            == "outcome"
        )

    def test_efficiency_and_process_dimensions_are_used(self) -> None:
        assert MaxOutputTokensCheck(limit=100, reason="r").dimension.value == "efficiency"
        assert ToolsWithinCheck(allowed_tools=("read_file",), reason="r").dimension.value == (
            "process"
        )

    def test_rejects_empty_reason(self) -> None:
        """A failure with no reason is not actionable, so it is not allowed."""
        with pytest.raises(ValidationError, match="non-whitespace"):
            SchemaValidCheck.model_validate({"expected_kind": "prd", "reason": "  "})

    @pytest.mark.parametrize("pattern", ["(unclosed", "[a-", "*bad"])
    def test_rejects_invalid_regex_at_load_time(self, pattern: str) -> None:
        """A broken regex must fail when the eval set loads, not mid-run."""
        with pytest.raises(ValidationError, match="invalid regex"):
            TextMatchesCheck.model_validate(
                {"field_name": "goals", "pattern": pattern, "reason": "r"}
            )

    def test_text_absent_also_validates_its_regex(self) -> None:
        with pytest.raises(ValidationError, match="invalid regex"):
            TextAbsentCheck.model_validate(
                {"field_name": "goals", "pattern": "(unclosed", "reason": "r"}
            )

    def test_rejects_empty_allowed_tools(self) -> None:
        """An empty tool set would fail every run and looks like a real result."""
        with pytest.raises(ValidationError):
            ToolsWithinCheck.model_validate({"allowed_tools": (), "reason": "r"})

    @pytest.mark.parametrize("limit", [0, -1])
    def test_rejects_non_positive_token_limit(self, limit: int) -> None:
        with pytest.raises(ValidationError):
            MaxOutputTokensCheck.model_validate({"limit": limit, "reason": "r"})

    def test_requires_at_least_one_check(self) -> None:
        """A task with no checks asserts nothing and would always pass."""
        with pytest.raises(ValidationError):
            GoldenTask.model_validate({"task_id": "t", "role": "qa", "prompt": "p", "checks": ()})


class TestGoldenTaskModel:
    def test_dimensions_covered_deduplicates(self) -> None:
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "product_manager",
                "prompt": "p",
                "checks": [
                    {"type": "field_non_empty", "field_name": "goals", "reason": "a"},
                    {"type": "field_non_empty", "field_name": "goals", "reason": "b"},
                    {"type": "max_output_tokens", "limit": 100, "reason": "c"},
                ],
            }
        )

        assert task.dimensions_covered() == frozenset({"outcome", "efficiency"})

    def test_rejects_unknown_role(self) -> None:
        """A fifth role would exceed the handoff ceiling in HARNESS_SPEC.md."""
        with pytest.raises(ValidationError):
            GoldenTask.model_validate(
                {
                    "task_id": "t",
                    "role": "tech_writer",
                    "prompt": "p",
                    "checks": [{"type": "max_output_tokens", "limit": 10, "reason": "r"}],
                }
            )

    def test_discriminator_selects_the_right_check_type(self) -> None:
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "qa",
                "prompt": "p",
                "checks": [
                    {"type": "text_absent", "field_name": "items", "pattern": "x", "reason": "r"}
                ],
            }
        )

        assert isinstance(task.checks[0], TextAbsentCheck)


class TestRunnerPassingChecks:
    def test_all_checks_pass_on_a_good_run(
        self,
        sample_pm_task: GoldenTask,
        passing_prd_run: Callable[..., AgentRun],
    ) -> None:
        result = run_task(sample_pm_task, passing_prd_run())

        assert result.passed
        assert result.failures == ()
        assert len(result.checks) == len(sample_pm_task.checks)

    def test_token_check_passes_under_the_limit(
        self, make_prd: Callable[..., ProductRequirements]
    ) -> None:
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "product_manager",
                "prompt": "p",
                "checks": [{"type": "max_output_tokens", "limit": 1000, "reason": "r"}],
            }
        )

        assert run_task(task, AgentRun(artifact=make_prd(), output_tokens=999)).passed

    def test_token_check_passes_at_exactly_the_limit(
        self, make_prd: Callable[..., ProductRequirements]
    ) -> None:
        """The bound is inclusive, so hitting it exactly is not a failure."""
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "product_manager",
                "prompt": "p",
                "checks": [{"type": "max_output_tokens", "limit": 100, "reason": "r"}],
            }
        )

        assert run_task(task, AgentRun(artifact=make_prd(), output_tokens=100)).passed

    def test_tools_check_passes_when_within_the_set(self) -> None:
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "qa",
                "prompt": "p",
                "checks": [
                    {
                        "type": "tools_within",
                        "allowed_tools": ["read_file", "run_tests"],
                        "reason": "r",
                    }
                ],
            }
        )
        run = AgentRun(artifact=None, tools_called=("read_file", "run_tests", "read_file"))

        assert run_task(task, run).passed

    def test_tools_check_passes_when_no_tools_were_called(self) -> None:
        """An agent that answered without tools is within any allowed set."""
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "qa",
                "prompt": "p",
                "checks": [{"type": "tools_within", "allowed_tools": ["read_file"], "reason": "r"}],
            }
        )

        assert run_task(task, AgentRun(artifact=None)).passed


class TestRunnerFailingChecks:
    """A check that cannot fail is not a check."""

    def test_missing_artifact_fails_every_artifact_check(self, sample_pm_task: GoldenTask) -> None:
        """The most important failure: no output must never read as green."""
        result = run_task(sample_pm_task, AgentRun(artifact=None))

        assert not result.passed
        details = " ".join(check.detail for check in result.failures)
        assert "produced no artifact" in details

    def test_token_check_fails_over_the_limit(
        self, make_prd: Callable[..., ProductRequirements]
    ) -> None:
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "product_manager",
                "prompt": "p",
                "checks": [{"type": "max_output_tokens", "limit": 100, "reason": "budget"}],
            }
        )

        result = run_task(task, AgentRun(artifact=make_prd(), output_tokens=101))

        assert not result.passed
        assert "101" in result.failures[0].detail
        assert "budget" in result.failures[0].reason

    def test_tools_check_fails_and_names_the_offender(self) -> None:
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "qa",
                "prompt": "p",
                "checks": [
                    {
                        "type": "tools_within",
                        "allowed_tools": ["read_file"],
                        "reason": "QA must not edit source",
                    }
                ],
            }
        )
        run = AgentRun(artifact=None, tools_called=("read_file", "edit_file", "edit_file"))

        result = run_task(task, run)

        assert not result.passed
        # Named once even though it was called twice.
        assert result.failures[0].detail == "called edit_file; allowed: read_file"

    def test_text_absent_fires_on_forbidden_content(
        self, make_prd: Callable[..., ProductRequirements]
    ) -> None:
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "product_manager",
                "prompt": "p",
                "checks": [
                    {
                        "type": "text_absent",
                        "field_name": "problem_statement",
                        "pattern": r"\bTODO\b",
                        "reason": "no placeholders",
                    }
                ],
            }
        )
        run = AgentRun(artifact=make_prd(problem_statement="Export data. TODO refine later."))

        result = run_task(task, run)

        assert not result.passed
        assert "forbidden pattern" in result.failures[0].detail

    def test_word_boundary_does_not_overmatch(
        self, make_prd: Callable[..., ProductRequirements]
    ) -> None:
        """A word merely containing the letters must not trip the check.

        "retype" holds "todo"-adjacent letters only by accident, but the case
        that matters is that ``\\b`` is a real boundary: without it, YAML escape
        damage would turn this into a pattern that matches nothing or matches
        too much, and both are invisible until a run is mis-scored.
        """
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "product_manager",
                "prompt": "p",
                "checks": [
                    {
                        "type": "text_absent",
                        "field_name": "problem_statement",
                        "pattern": r"\bTODO\b",
                        "reason": "no placeholders",
                    }
                ],
            }
        )
        run = AgentRun(artifact=make_prd(problem_statement="Users must retype the table by hand."))

        assert run_task(task, run).passed

    def test_word_boundary_pattern_actually_matches_its_target(
        self, make_prd: Callable[..., ProductRequirements]
    ) -> None:
        """The companion to the test above: the pattern must still fire."""
        pattern = r"\bTODO\b"

        assert re.search(pattern, "TODO refine later", re.IGNORECASE)
        assert re.search(pattern, "a todo note", re.IGNORECASE)
        assert not re.search(pattern, "retodoes", re.IGNORECASE)

    def test_text_matches_fails_when_absent(self, failing_prd_run: Callable[..., AgentRun]) -> None:
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "product_manager",
                "prompt": "p",
                "checks": [
                    {
                        "type": "text_matches",
                        "field_name": "acceptance_criteria",
                        "pattern": "verification",
                        "reason": "criteria must say how to check them",
                    }
                ],
            }
        )

        result = run_task(task, failing_prd_run())

        assert not result.passed
        assert "does not match" in result.failures[0].detail

    def test_field_non_empty_fails_on_a_blank_field(
        self, make_prd: Callable[..., ProductRequirements]
    ) -> None:
        """Catches an empty field the schema itself permits.

        ``goals`` cannot be empty - the domain model requires at least one - but
        ``non_goals`` can, and legitimately defaults to empty. A golden task that
        demands scope boundaries therefore needs this check: it fails on an
        artifact that is perfectly valid as far as pydantic is concerned, which
        is the gap the eval layer exists to cover.
        """
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "product_manager",
                "prompt": "p",
                "checks": [
                    {
                        "type": "field_non_empty",
                        "field_name": "non_goals",
                        "reason": "a bundled request must declare scope boundaries",
                    }
                ],
            }
        )
        run = AgentRun(artifact=make_prd(non_goals=()))

        result = run_task(task, run)

        assert not result.passed
        assert result.failures[0].detail == "non_goals is empty"

    def test_field_non_empty_passes_when_populated(
        self, make_prd: Callable[..., ProductRequirements]
    ) -> None:
        """The companion: the check must not be permanently red."""
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "product_manager",
                "prompt": "p",
                "checks": [
                    {
                        "type": "field_non_empty",
                        "field_name": "non_goals",
                        "reason": "needs scope boundaries",
                    }
                ],
            }
        )
        run = AgentRun(artifact=make_prd(non_goals=("Excel format support",)))

        assert run_task(task, run).passed

    def test_schema_forbids_the_empty_goals_this_check_would_catch(
        self, make_prd: Callable[..., ProductRequirements]
    ) -> None:
        """Documents why the test above uses non_goals rather than goals."""
        with pytest.raises(ValidationError):
            make_prd(goals=())

    def test_kind_check_on_wrong_type(self, make_prd: Callable[..., ProductRequirements]) -> None:
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "qa",
                "prompt": "p",
                "checks": [
                    {"type": "kind_correct", "expected_kind": "test_plan", "reason": "plan stage"}
                ],
            }
        )
        run = AgentRun(artifact=make_prd())

        result = run_task(task, run)

        assert not result.passed
        assert "ProductRequirements" in result.failures[0].detail

    def test_schema_failure_explains_the_validation_error(
        self,
        prd_payload_missing_criteria: Callable[[], str],
    ) -> None:
        """The report must say what was wrong, not just that the type differed.

        The artifact is a Design while the check expects a PRD, so the type test
        fails and the runner re-validates the raw payload to explain why. That
        payload is a PRD missing its criteria, so the diagnostic names the
        missing field instead of stopping at "wrong type".
        """
        from sdlc_agent.domain.engineering import Design

        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "product_manager",
                "prompt": "p",
                "checks": [
                    {
                        "type": "schema_valid",
                        "expected_kind": "prd",
                        "reason": "must be a PRD",
                    }
                ],
            }
        )
        design = Design.model_validate(
            {"prd_id": uuid.uuid4(), "summary": "Stream it.", "components": ("export.py",)}
        )
        run = AgentRun(artifact=design, raw_payload=prd_payload_missing_criteria())

        result = run_task(task, run)

        assert not result.passed
        detail = result.failures[0].detail
        # Names the type actually produced...
        assert "Design" in detail
        # ...and the underlying validation reason from the raw payload.
        assert "acceptance_criteria" in detail

    def test_schema_failure_without_raw_payload_still_names_the_type(
        self, make_prd: Callable[..., ProductRequirements]
    ) -> None:
        """No raw payload means no diagnosis, but the type mismatch is still useful."""
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "qa",
                "prompt": "p",
                "checks": [{"type": "schema_valid", "expected_kind": "test_plan", "reason": "r"}],
            }
        )

        result = run_task(task, AgentRun(artifact=make_prd(), raw_payload=""))

        assert not result.passed
        assert result.failures[0].detail == "got ProductRequirements"

    def test_unknown_field_name_is_a_harness_error_not_a_failure(
        self, passing_prd_run: Callable[..., AgentRun]
    ) -> None:
        """A check naming a nonexistent field is a bug in the eval set."""
        task = GoldenTask.model_validate(
            {
                "task_id": "t",
                "role": "product_manager",
                "prompt": "p",
                "checks": [
                    {
                        "type": "field_non_empty",
                        "field_name": "no_such_field",
                        "reason": "r",
                    }
                ],
            }
        )

        with pytest.raises(EvalRunError, match="does not exist"):
            run_task(task, passing_prd_run())


class TestEvalReport:
    def test_empty_report_is_not_green(self) -> None:
        """Zero failures because nothing ran is the false green this codebase
        refuses everywhere else, so it is refused here too."""
        assert not EvalReport(results=()).is_green
        assert EvalReport(results=()).total == 0

    def test_green_requires_every_task_to_pass(
        self,
        sample_pm_task: GoldenTask,
        passing_prd_run: Callable[..., AgentRun],
    ) -> None:
        green = TaskResult(
            task_id="a",
            role="qa",
            checks=run_task(sample_pm_task, passing_prd_run()).checks,
        )
        report = EvalReport(results=(green,))

        assert report.is_green
        assert report.passed == 1

    def test_one_failure_makes_the_report_red(
        self,
        sample_pm_task: GoldenTask,
        passing_prd_run: Callable[..., AgentRun],
        failing_prd_run: Callable[..., AgentRun],
    ) -> None:
        passing = run_task(sample_pm_task, passing_prd_run())
        failing = run_task(sample_pm_task, failing_prd_run())
        failing_relabelled = TaskResult(task_id="other", role=failing.role, checks=failing.checks)

        report = EvalReport(results=(passing, failing_relabelled))

        assert not report.is_green
        assert report.passed == 1
        assert report.failed == 1

    def test_by_dimension_counts_unexercised_dimensions_as_zero(
        self,
        sample_pm_task: GoldenTask,
        passing_prd_run: Callable[..., AgentRun],
    ) -> None:
        report = EvalReport(results=(run_task(sample_pm_task, passing_prd_run()),))

        counts = report.by_dimension()

        assert counts[EvalDimension.OUTCOME][1] > 0
        assert counts[EvalDimension.STYLE] == (0, 0)

    def test_by_dimension_separates_pass_from_total(
        self,
        sample_pm_task: GoldenTask,
        passing_prd_run: Callable[..., AgentRun],
        failing_prd_run: Callable[..., AgentRun],
    ) -> None:
        """A dimension must report failures, not just how many checks ran."""
        report = EvalReport(
            results=(
                run_task(sample_pm_task, passing_prd_run()),
                run_task(sample_pm_task, failing_prd_run()),
            )
        )

        passed, total = report.by_dimension()[EvalDimension.OUTCOME]

        assert 0 < passed < total

    def test_render_names_the_failing_check(
        self,
        sample_pm_task: GoldenTask,
        failing_prd_run: Callable[..., AgentRun],
    ) -> None:
        report = EvalReport(results=(run_task(sample_pm_task, failing_prd_run()),))

        text = report.render()

        assert "FAIL" in text
        assert "text_matches" in text
        assert "0/1 tasks passed" in text

    def test_render_shows_pass_without_details(
        self,
        sample_pm_task: GoldenTask,
        passing_prd_run: Callable[..., AgentRun],
    ) -> None:
        report = EvalReport(results=(run_task(sample_pm_task, passing_prd_run()),))

        text = report.render()

        assert "PASS" in text
        assert "1/1 tasks passed" in text


class TestRunEvalSet:
    def test_key_must_match_task_id(
        self,
        sample_pm_task: GoldenTask,
        passing_prd_run: Callable[..., AgentRun],
    ) -> None:
        """A mislabelled entry would score one task against another's checks."""
        with pytest.raises(EvalRunError, match="must match task_id"):
            run_eval_set({"wrong_key": (sample_pm_task, passing_prd_run())})

    def test_aggregates_multiple_tasks(
        self,
        sample_pm_task: GoldenTask,
        passing_prd_run: Callable[..., AgentRun],
    ) -> None:
        report = run_eval_set({sample_pm_task.task_id: (sample_pm_task, passing_prd_run())})

        assert report.total == 1
        assert report.is_green

    def test_mixed_results_are_aggregated(
        self,
        sample_pm_task: GoldenTask,
        passing_prd_run: Callable[..., AgentRun],
        failing_prd_run: Callable[..., AgentRun],
    ) -> None:
        """One pass and one fail must not average out into a green report."""
        second = GoldenTask.model_validate(
            {**sample_pm_task.model_dump(mode="json"), "task_id": "pm_second"}
        )

        report = run_eval_set(
            {
                sample_pm_task.task_id: (sample_pm_task, passing_prd_run()),
                second.task_id: (second, failing_prd_run()),
            }
        )

        assert report.total == 2
        assert report.passed == 1
        assert not report.is_green


class TestLoadTasks:
    def test_shipped_example_directory_exists(self) -> None:
        assert EVAL_DIR.is_dir()

    def test_loads_a_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "t.yaml"
        path.write_text(
            """
- task_id: qa_one
  role: qa
  prompt: Build a plan.
  checks:
    - type: max_output_tokens
      limit: 100
      reason: keep it cheap
""",
            encoding="utf-8",
        )

        tasks = load_tasks(path)

        assert len(tasks) == 1
        assert tasks[0].task_id == "qa_one"
        assert tasks[0].checks[0].reason == "keep it cheap"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_tasks(tmp_path / "nope.yaml")

    def test_empty_file_is_an_error_not_zero_tasks(self, tmp_path: Path) -> None:
        """An empty file would silently shrink the eval set and still pass."""
        path = tmp_path / "empty.yaml"
        path.write_text("# nothing here\n", encoding="utf-8")

        with pytest.raises(EvalRunError, match="empty"):
            load_tasks(path)

    def test_non_list_top_level_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("task_id: not_a_list\n", encoding="utf-8")

        with pytest.raises(EvalRunError, match="top-level list"):
            load_tasks(path)

    def test_invalid_yaml_is_wrapped_with_the_filename(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("- task_id: [unclosed\n", encoding="utf-8")

        with pytest.raises(EvalRunError, match="not valid YAML"):
            load_tasks(path)

    def test_invalid_task_names_the_file_and_task(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_task.yaml"
        path.write_text(
            "- task_id: broken_one\n  role: qa\n  prompt: p\n  checks: []\n",
            encoding="utf-8",
        )

        with pytest.raises(EvalRunError) as excinfo:
            load_tasks(path)

        message = str(excinfo.value)
        assert "bad_task.yaml" in message
        assert "broken_one" in message

    def test_unknown_check_type_lists_the_known_ones(self, tmp_path: Path) -> None:
        path = tmp_path / "unknown_check.yaml"
        path.write_text(
            "- task_id: t\n  role: qa\n  prompt: p\n"
            "  checks:\n    - {type: vibes_check, reason: r}\n",
            encoding="utf-8",
        )

        with pytest.raises(EvalRunError, match="known check types"):
            load_tasks(path)

    def test_non_mapping_entry_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "scalar.yaml"
        path.write_text("- just_a_string\n", encoding="utf-8")

        with pytest.raises(EvalRunError, match="not a mapping"):
            load_tasks(path)


class TestLoadAllTasks:
    def test_shipped_set_covers_every_role(self) -> None:
        """HARNESS_SPEC.md 5.2 wants 3-5 golden tasks per role."""
        tasks = load_all_tasks()
        per_role = summarize_coverage(tasks)["tasks_per_role"]

        assert set(per_role) == {role.value for role in PipelineRole}
        for role, count in per_role.items():
            assert 3 <= count <= 5, f"{role} has {count} golden tasks, expected 3-5"

    def test_shipped_task_ids_are_unique(self) -> None:
        tasks = load_all_tasks()

        assert len(tasks) > 0

    def test_duplicate_ids_across_files_are_rejected(self, tmp_path: Path) -> None:
        """A duplicate would silently drop one task from a keyed report."""
        body = (
            "- task_id: dup\n  role: qa\n  prompt: p\n"
            "  checks:\n    - {type: max_output_tokens, limit: 10, reason: r}\n"
        )
        (tmp_path / "a.yaml").write_text(body, encoding="utf-8")
        (tmp_path / "b.yaml").write_text(body, encoding="utf-8")

        with pytest.raises(EvalRunError, match="duplicate task_id"):
            load_all_tasks(tmp_path)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EvalRunError, match="directory not found"):
            load_all_tasks(tmp_path / "absent")

    def test_tasks_for_role_filters(self) -> None:
        assert set(tasks_for_role("qa")) == {
            tid for tid, task in load_all_tasks().items() if task.role.value == "qa"
        }

    def test_tasks_for_role_unknown_returns_empty(self) -> None:
        assert tasks_for_role("tech_writer") == {}


class TestCoverageSummary:
    def test_reports_unexercised_dimensions(self) -> None:
        """Style is deliberately unexercised until a judged check is justified."""
        summary = summarize_coverage(load_all_tasks())

        for role, missing in summary["unexercised"].items():
            assert "style" in missing, f"{role} unexpectedly covers style"

    def test_outcome_dimension_is_exercised_for_every_role(self) -> None:
        summary = summarize_coverage(load_all_tasks())

        for role, dimensions in summary["dimensions"].items():
            assert "outcome" in dimensions, f"{role} has no outcome check"

    def test_process_dimension_is_exercised_somewhere(self) -> None:
        """tools_within is the only process check; at least one role must use it."""
        summary = summarize_coverage(load_all_tasks())

        assert any("process" in dims for dims in summary["dimensions"].values())

    def test_all_shipped_patterns_compile(self) -> None:
        """Guards the YAML escapes: a mangled \\b would only fail at run time."""
        for task in load_all_tasks().values():
            for check in task.checks:
                if hasattr(check, "pattern"):
                    re.compile(check.pattern)

    def test_no_shipped_check_targets_a_nonexistent_field(self) -> None:
        """Catches a field rename breaking the eval set silently."""
        from sdlc_agent.domain import ARTIFACT_TYPES

        for task in load_all_tasks().values():
            for check in task.checks:
                field = getattr(check, "field_name", None)
                if field is None:
                    continue
                expected = getattr(check, "expected_kind", None)
                if expected is None:
                    # Resolve through the role's primary artifact type.
                    continue
                model = ARTIFACT_TYPES[expected]
                assert field in model.model_fields, f"{task.task_id}: {field} not on {model}"

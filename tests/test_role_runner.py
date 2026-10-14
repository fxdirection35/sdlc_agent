"""Tests for the golden-task-to-role bridge.

All offline: roles here are scripted, so these tests pin the bridge's
behaviour - which tasks a role can answer, what context a task becomes, how a
role's result maps to an AgentRun - without a model call. The recorded live
run lives in ``test_llm_golden.py``.
"""

from __future__ import annotations

import pytest

from sdlc_agent.domain.base import Artifact, ArtifactKind, PipelineRole
from sdlc_agent.domain.product import AcceptanceCriterion, ProductRequirements
from sdlc_agent.evals import GoldenTask, load_all_tasks
from sdlc_agent.evals.role_runner import (
    context_for_task,
    exclusion_reason,
    expected_kinds,
    role_addressable_tasks,
    run_golden_task,
    run_role_eval_set,
)
from sdlc_agent.pipeline.roles import (
    ROLE_OUTPUTS,
    FailingRole,
    RoleContext,
    ScriptedRole,
)

#: The real golden set, loaded once. Testing against it rather than a
#: hand-built miniature is the point: these tests must notice when the eval
#: set and the bridge drift apart.
ALL_TASKS: tuple[GoldenTask, ...] = tuple(load_all_tasks().values())

#: The one task today whose expected kind no role produces. It targets the
#: test-execution stage (a TestRun); the sandbox executor now exists and the
#: task is evaluated against it in tests/test_sandbox_executor.py. The
#: exclusion from ROLE evals stays correct: no role produces a TestRun.
KNOWN_EXCLUDED = "qa_reports_not_run_honestly"


def make_prd(context: RoleContext) -> Artifact:
    """A valid PRD factory, in the shape ScriptedRole wants."""
    return ProductRequirements.model_validate(
        {
            "requirement_id": context.requirement.artifact_id,
            "problem_statement": "Report users cannot get data out of the app.",
            "goals": ("Export any filtered report as CSV",),
            "acceptance_criteria": (
                AcceptanceCriterion.model_validate(
                    {"statement": "CSV has one header row", "verification": "pytest t.py"}
                ),
            ),
        }
    )


@pytest.fixture
def pm_role() -> ScriptedRole:
    return ScriptedRole(PipelineRole.PRODUCT_MANAGER, make_prd)


@pytest.fixture
def pm_csv_export() -> GoldenTask:
    return next(task for task in ALL_TASKS if task.task_id == "pm_csv_export")


class TestExpectedKinds:
    def test_collects_kinds_from_both_type_checks(self, sample_pm_task: GoldenTask) -> None:
        """schema_valid and kind_correct both name the kind; miss either and
        an exclusion decision is made on partial evidence."""
        assert expected_kinds(sample_pm_task) == frozenset({ArtifactKind.PRD})

    def test_task_without_type_checks_demands_nothing(self) -> None:
        task = GoldenTask.model_validate(
            {
                "task_id": "kindless",
                "role": "product_manager",
                "prompt": "do the thing",
                "checks": [
                    {
                        "type": "field_non_empty",
                        "field_name": "goals",
                        "reason": "goals matter",
                    }
                ],
            }
        )
        assert expected_kinds(task) == frozenset()


class TestExclusion:
    def test_task_matching_the_roles_output_is_addressable(self, pm_csv_export: GoldenTask) -> None:
        assert exclusion_reason(pm_csv_export) == ""

    def test_excluded_task_names_what_it_wanted(self) -> None:
        task = next(task for task in ALL_TASKS if task.task_id == KNOWN_EXCLUDED)
        reason = exclusion_reason(task)
        assert "test_run" in reason
        assert "test_plan" in reason

    def test_exclusion_matches_kinds_for_every_task(self) -> None:
        """Derived guard: the reason string and the kind sets must agree for
        every task in the set, so neither side can drift alone."""
        for task in ALL_TASKS:
            reason = exclusion_reason(task)
            mismatched = expected_kinds(task) - {ROLE_OUTPUTS[task.role]}
            if mismatched:
                assert reason != "", task.task_id
                for kind in mismatched:
                    assert kind.value in reason, task.task_id
            else:
                assert reason == "", task.task_id

    def test_every_role_keeps_at_least_one_addressable_task(self) -> None:
        """A role whose every task got excluded would silently never be
        evaluated again; that must be a loud change, not a quiet one."""
        for role in PipelineRole:
            assert role_addressable_tasks(role, ALL_TASKS), f"no addressable tasks for {role.value}"

    def test_the_only_excluded_task_today_is_the_known_one(self) -> None:
        excluded = sorted(task.task_id for task in ALL_TASKS if exclusion_reason(task))
        assert excluded == [KNOWN_EXCLUDED]


class TestContextForTask:
    def test_prompt_arrives_verbatim_as_the_description(self, pm_csv_export: GoldenTask) -> None:
        assert context_for_task(pm_csv_export).requirement.description == pm_csv_export.prompt

    def test_title_is_the_first_non_blank_line(self) -> None:
        task = GoldenTask.model_validate(
            {
                "task_id": "t_title",
                "role": "product_manager",
                "prompt": "\n\n    First line wins.   \nSecond line.",
                "checks": [{"type": "field_non_empty", "field_name": "goals", "reason": "r"}],
            }
        )
        title = context_for_task(task).requirement.title
        assert title == "First line wins."

    def test_context_is_pristine(self, pm_csv_export: GoldenTask) -> None:
        """First attempt, no feedback, no upstream - determinism is what makes
        a recorded response replayable for the same task."""
        context = context_for_task(pm_csv_export)
        assert context.attempt == 1
        assert context.feedback == ""
        assert context.upstream == {}

    def test_source_repo_names_the_task(self, pm_csv_export: GoldenTask) -> None:
        assert pm_csv_export.task_id in context_for_task(pm_csv_export).requirement.source_repo


class TestRunGoldenTask:
    def test_maps_role_result_into_an_agent_run(self, pm_csv_export: GoldenTask) -> None:
        role = ScriptedRole(
            PipelineRole.PRODUCT_MANAGER,
            make_prd,
            output_tokens=700,
            tools_called=("read_file",),
        )

        run = run_golden_task(role, pm_csv_export)

        assert run.artifact is not None
        assert run.output_tokens == 700
        assert run.tools_called == ("read_file",)
        assert role.call_count == 1

    def test_a_role_error_becomes_an_artifact_less_run(self, pm_csv_export: GoldenTask) -> None:
        """A role that cannot answer is an evaluable failure, not a broken
        eval: every artifact check must fail on 'produced no artifact'."""
        role = FailingRole(PipelineRole.PRODUCT_MANAGER)

        run = run_golden_task(role, pm_csv_export)

        assert run.artifact is None


class TestRunRoleEvalSet:
    def test_scores_every_addressable_task_of_the_role(self, pm_role: ScriptedRole) -> None:
        report = run_role_eval_set(pm_role, ALL_TASKS)

        addressable = role_addressable_tasks(PipelineRole.PRODUCT_MANAGER, ALL_TASKS)
        assert report.total == len(addressable)
        assert {result.task_id for result in report.results} == {
            task.task_id for task in addressable
        }

    def test_only_the_roles_own_tasks_are_scored(self, pm_role: ScriptedRole) -> None:
        report = run_role_eval_set(pm_role, ALL_TASKS)
        assert all(result.role == PipelineRole.PRODUCT_MANAGER.value for result in report.results)

    def test_a_failing_role_is_not_green(self) -> None:
        """The empty-then-failing path: the report must carry failures with
        reasons, not raise, so a red eval stays diagnosable."""
        report = run_role_eval_set(FailingRole(PipelineRole.PRODUCT_MANAGER), ALL_TASKS)

        assert report.total > 0
        assert not report.is_green
        assert "produced no artifact" in report.render()

    def test_scripted_prd_scores_against_the_real_pm_bar(self, pm_role: ScriptedRole) -> None:
        """The canned PRD is structurally valid; whether it clears the golden
        bar is the eval set's decision. Asserting the exact outcome here
        means a change to either the bridge or the PM tasks is deliberate."""
        report = run_role_eval_set(pm_role, ALL_TASKS)
        assert report.total == 3
        assert report.render().startswith(f"eval: {report.passed}/3 tasks passed")


class TestSanityOfTheLoadedSet:
    def test_the_golden_set_still_has_the_expected_shape(self) -> None:
        """Fails when tasks are added or renamed without revisiting the
        expectations above - especially KNOWN_EXCLUDED."""
        assert len(ALL_TASKS) == 14
        per_role: dict[str, int] = {}
        for task in ALL_TASKS:
            per_role[task.role.value] = per_role.get(task.role.value, 0) + 1
        assert per_role == {
            "product_manager": 3,
            "architect": 3,
            "developer": 4,
            "qa": 4,
        }

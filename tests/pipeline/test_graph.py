"""Tests for the pipeline graph: gates, loop bounds and the budget guard.

The highest-value assertion here is that resuming a gate does not re-run the
role that produced the artifact. LangGraph re-executes a node from the top on
resume, so if a role and its gate shared a node, every human review would bill
the LLM call twice - invisible in the output and visible only on the invoice.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langgraph.types import Command

from sdlc_agent.domain import ArtifactKind, PipelineRole, Verdict
from sdlc_agent.domain.base import ReviewDecision
from sdlc_agent.domain.qa import TestStatus
from sdlc_agent.pipeline import (
    MAX_GATE_REVISIONS,
    GraphConstructionError,
    StubExecutor,
    build_graph,
    initial_state,
)
from sdlc_agent.pipeline.roles import FailingRole, ScriptedRole
from sdlc_agent.pipeline.state import RunStatus, StageName

from .helpers import (
    approve,
    approve_until_terminal,
    artifact_kinds,
    audit_details,
    audit_trail,
    gate_name,
    make_artifact,
    reject,
)


class TestHappyPath:
    def test_stops_at_the_prd_gate_first(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, _ = make_graph(config)

        result = app.invoke(initial_state(requirement), thread)

        assert gate_name(result) == StageName.PRD_REVIEW.value
        assert result["status"] is RunStatus.IN_PROGRESS

    def test_passes_through_exactly_three_gates(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """Three hard gates per HARNESS_SPEC.md: PRD, design and merge."""
        app, _ = make_graph(config)

        result = app.invoke(initial_state(requirement), thread)
        seen = [gate_name(result)]
        for _ in range(3):
            result = app.invoke(approve(), thread)
            name = gate_name(result)
            if not name:
                break
            seen.append(name)

        assert seen == ["prd_review", "design_review", "merge_review"]

    def test_completes_with_every_artifact(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, _ = make_graph(config)

        result = app.invoke(initial_state(requirement), thread)
        for _ in range(3):
            result = app.invoke(approve(), thread)

        assert result["status"] is RunStatus.COMPLETED
        assert artifact_kinds(result) == {
            ArtifactKind.PRD.value,
            ArtifactKind.DESIGN.value,
            ArtifactKind.CODE_CHANGE.value,
            ArtifactKind.TEST_PLAN.value,
            ArtifactKind.TEST_RUN.value,
            ArtifactKind.QUALITY_GATE.value,
        }

    def test_visits_every_stage_in_order(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, _ = make_graph(config)

        result = app.invoke(initial_state(requirement), thread)
        for _ in range(3):
            result = app.invoke(approve(), thread)

        stages = [entry.split(":")[0] for entry in audit_trail(result)]
        assert stages == [
            "pm",
            "prd_review",
            "architect",
            "design_review",
            "developer",
            "qa",
            "verify",
            "merge_review",
            "finalize",
        ]

    def test_each_role_runs_exactly_once(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, roles = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        for _ in range(3):
            app.invoke(approve(), thread)

        assert {role.value: impl.call_count for role, impl in roles.items()} == {
            "product_manager": 1,
            "architect": 1,
            "developer": 1,
            "qa": 1,
        }


class TestGateResumeDoesNotRerunTheRole:
    """The constraint from HARNESS_SPEC.md section 2.4, asserted directly."""

    def test_role_is_not_called_again_after_resume(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, roles = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        assert roles[PipelineRole.PRODUCT_MANAGER].call_count == 1

        app.invoke(approve(), thread)

        # Approving the PRD gate re-enters the graph; if the gate shared a node
        # with the PM role, this would now read 2 and the run would cost double.
        assert roles[PipelineRole.PRODUCT_MANAGER].call_count == 1

    def test_no_role_runs_twice_across_all_gates(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, roles = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        for _ in range(3):
            app.invoke(approve(), thread)

        for role, impl in roles.items():
            assert impl.call_count == 1, f"{role.value} ran {impl.call_count} times"

    def test_gate_interrupt_payload_reports_spend(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """A reviewer deciding whether to continue needs to see the cost so far."""
        app, _ = make_graph(
            config,
            spent_usd={PipelineRole.PRODUCT_MANAGER: 1.25},
        )

        result = app.invoke(initial_state(requirement), thread)

        payload = result["__interrupt__"][0].value
        assert payload["spent_usd"] == 1.25
        assert payload["budget_usd"] == config.budget.per_task_usd
        assert set(payload["allowed_verdicts"]) == {v.value for v in Verdict}


class TestGateRejection:
    def test_rejection_sends_work_back_to_the_producing_role(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, roles = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        result = app.invoke(reject("the PRD is missing scope"), thread)

        # Back at the same gate, meaning the PM ran again and produced a new PRD.
        assert gate_name(result) == StageName.PRD_REVIEW.value
        assert roles[PipelineRole.PRODUCT_MANAGER].call_count == 2

    def test_rejection_comment_reaches_the_role_as_feedback(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """The role must see why it was rejected, or it cannot revise usefully."""
        app, roles = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        app.invoke(reject("add the empty-result case"), thread)

        second_context = roles[PipelineRole.PRODUCT_MANAGER].calls[1]
        assert second_context.feedback == "add the empty-result case"
        assert second_context.attempt == 2

    def test_first_attempt_has_no_feedback(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, roles = make_graph(config)

        app.invoke(initial_state(requirement), thread)

        assert roles[PipelineRole.PRODUCT_MANAGER].calls[0].feedback == ""
        assert roles[PipelineRole.PRODUCT_MANAGER].calls[0].attempt == 1

    def test_rejection_is_recorded_in_the_audit_trail(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, _ = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        result = app.invoke(reject("too vague"), thread)

        assert "prd_review:rejected" in audit_trail(result)
        # The comment lives in the entry's detail, not its stage:event label.
        assert any("too vague" in detail for detail in audit_details(result))

    def test_approval_then_rejection_at_a_later_gate(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, roles = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        app.invoke(approve(), thread)  # PRD approved
        result = app.invoke(reject("no streaming"), thread)  # design rejected

        assert gate_name(result) == StageName.DESIGN_REVIEW.value
        assert roles[PipelineRole.ARCHITECT].call_count == 2
        # The PRD was not re-run: only the rejected role repeats.
        assert roles[PipelineRole.PRODUCT_MANAGER].call_count == 1


class TestGateRevisionLimit:
    def test_escalates_after_too_many_rejections(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """A reviewer who keeps rejecting must not loop forever."""
        app, roles = make_graph(config)

        result = app.invoke(initial_state(requirement), thread)
        for _ in range(MAX_GATE_REVISIONS + 1):
            result = app.invoke(reject("still wrong"), thread)

        assert result["status"] is RunStatus.ESCALATED
        assert gate_name(result) == ""
        # The role ran once initially plus once per rejection that was allowed.
        assert roles[PipelineRole.PRODUCT_MANAGER].call_count == MAX_GATE_REVISIONS + 1

    def test_revision_count_is_per_gate(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """Rejections at the PRD gate must not count against the design gate."""
        app, _ = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        app.invoke(reject("vague"), thread)
        result = app.invoke(approve(), thread)  # PRD approved on the second try

        assert result["gate_revisions"][StageName.PRD_REVIEW.value] == 1
        assert StageName.DESIGN_REVIEW.value not in result["gate_revisions"]

    def test_needs_human_escalates_immediately(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """Not a rejection to retry - the reviewer is asking for a person."""
        app, roles = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        result = app.invoke(
            Command(resume={"verdict": "needs_human", "reviewer": "alice", "comment": "?"}),
            thread,
        )

        assert result["status"] is RunStatus.ESCALATED
        assert roles[PipelineRole.PRODUCT_MANAGER].call_count == 1


class TestQARetryLoop:
    def test_failed_tests_return_to_the_developer(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """The whole retry loop runs without a gate, inside one invoke."""
        app, roles = make_graph(config, executor=StubExecutor(status=TestStatus.FAILED))

        app.invoke(initial_state(requirement), thread)
        app.invoke(approve(), thread)
        result = app.invoke(approve(), thread)

        # 1 initial build plus max_qa_retries revisions, then escalation.
        assert roles[PipelineRole.DEVELOPER].call_count == 1 + config.pipeline.max_qa_retries
        assert result["status"] is RunStatus.ESCALATED
        # No merge gate was ever offered: the suite never passed.
        assert gate_name(result) == ""
        assert ArtifactKind.QUALITY_GATE.value in {str(k) for k in result["artifacts"]}

    def test_retry_loop_is_bounded_by_config(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """An unbounded fix-and-retest loop is a budget fire with no exit."""
        app, roles = make_graph(config, executor=StubExecutor(status=TestStatus.FAILED))

        result = approve_until_terminal(app, thread, initial_state(requirement))

        assert result["status"] is RunStatus.ESCALATED
        assert roles[PipelineRole.DEVELOPER].call_count == 1 + config.pipeline.max_qa_retries
        assert roles[PipelineRole.QA].call_count == 1 + config.pipeline.max_qa_retries

    def test_a_higher_retry_limit_permits_more_attempts(
        self, make_graph: Any, config: Any, requirement: Any
    ) -> None:
        """The bound really comes from config, not a constant in the router."""
        looser = config.model_validate(
            {**config.model_dump(mode="json"), "pipeline": {"max_qa_retries": 4}}
        )

        app, roles = make_graph(looser, executor=StubExecutor(status=TestStatus.FAILED))
        thread = {"configurable": {"thread_id": uuid.uuid4().hex}}

        approve_until_terminal(app, thread, initial_state(requirement))

        assert roles[PipelineRole.DEVELOPER].call_count == 5

    def test_qa_attempts_are_counted(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, _ = make_graph(config, executor=StubExecutor(status=TestStatus.FAILED))

        result = approve_until_terminal(app, thread, initial_state(requirement))

        # One verify per developer attempt.
        assert result["qa_attempts"] == 1 + config.pipeline.max_qa_retries

    def test_passing_tests_reach_the_merge_gate(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, roles = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        app.invoke(approve(), thread)
        result = app.invoke(approve(), thread)

        assert gate_name(result) == StageName.MERGE_REVIEW.value
        assert roles[PipelineRole.DEVELOPER].call_count == 1
        assert result["qa_attempts"] == 1

    def test_not_run_escalates_instead_of_retrying(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """An environment failure is not a code failure; retrying wastes budget."""
        app, roles = make_graph(config, executor=StubExecutor(status=TestStatus.NOT_RUN))

        app.invoke(initial_state(requirement), thread)
        app.invoke(approve(), thread)
        result = app.invoke(approve(), thread)

        assert result["status"] is RunStatus.ESCALATED
        assert roles[PipelineRole.DEVELOPER].call_count == 1
        assert result["artifacts"][ArtifactKind.QUALITY_GATE].verdict is Verdict.NEEDS_HUMAN


class TestBudgetGuard:
    def test_blocks_before_calling_the_model(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """The check must precede the call, or the breaching call is already paid."""
        app, roles = make_graph(config, spent_usd={PipelineRole.PRODUCT_MANAGER: 6.0})

        # The PM runs: spend is zero when its node starts. Its own cost is what
        # breaches the ceiling, so the block lands on the next node.
        result = app.invoke(initial_state(requirement), thread)
        assert roles[PipelineRole.PRODUCT_MANAGER].call_count == 1
        assert gate_name(result) == StageName.PRD_REVIEW.value

        result = app.invoke(approve(), thread)

        assert roles[PipelineRole.ARCHITECT].call_count == 0
        assert gate_name(result) == StageName.BUDGET_REVIEW.value

    def test_records_where_it_stopped(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, _ = make_graph(config, spent_usd={PipelineRole.PRODUCT_MANAGER: 6.0})

        app.invoke(initial_state(requirement), thread)
        result = app.invoke(approve(), thread)

        assert result["blocked_stage"] == StageName.ARCHITECT.value
        assert result["status"] is RunStatus.BUDGET_EXCEEDED
        assert result["spent_usd"] == pytest.approx(6.0)
        assert "architect:blocked_by_budget" in audit_trail(result)

    def test_granting_budget_resumes_the_blocked_role(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, roles = make_graph(config, spent_usd={PipelineRole.PRODUCT_MANAGER: 6.0})

        app.invoke(initial_state(requirement), thread)
        app.invoke(approve(), thread)
        result = app.invoke(Command(resume={"grant_usd": 50.0}), thread)

        assert result["budget_override_usd"] == 50.0
        # The blocked node ran, then the flow advanced to the design gate.
        assert roles[PipelineRole.ARCHITECT].call_count == 1
        assert gate_name(result) == StageName.DESIGN_REVIEW.value

    def test_refusing_budget_ends_the_run(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, _ = make_graph(config, spent_usd={PipelineRole.PRODUCT_MANAGER: 6.0})

        app.invoke(initial_state(requirement), thread)
        app.invoke(approve(), thread)
        result = app.invoke(Command(resume={"grant_usd": 0}), thread)

        assert result["status"] is RunStatus.BUDGET_EXCEEDED
        assert "budget_review:aborted" in audit_trail(result)
        assert any("no additional budget" in detail for detail in audit_details(result))

    def test_exact_ceiling_blocks(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """Spending exactly the ceiling leaves nothing for the next call."""
        app, roles = make_graph(config, spent_usd={PipelineRole.PRODUCT_MANAGER: 5.0})

        app.invoke(initial_state(requirement), thread)
        app.invoke(approve(), thread)

        assert roles[PipelineRole.ARCHITECT].call_count == 0

    def test_under_ceiling_proceeds(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, roles = make_graph(config, spent_usd={PipelineRole.PRODUCT_MANAGER: 4.99})

        app.invoke(initial_state(requirement), thread)
        result = app.invoke(approve(), thread)

        assert roles[PipelineRole.ARCHITECT].call_count == 1
        assert gate_name(result) == StageName.DESIGN_REVIEW.value

    def test_spend_accumulates_across_roles(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, _ = make_graph(
            config,
            spent_usd={
                PipelineRole.PRODUCT_MANAGER: 1.0,
                PipelineRole.ARCHITECT: 2.0,
                PipelineRole.DEVELOPER: 0.5,
                PipelineRole.QA: 0.25,
            },
        )

        result = approve_until_terminal(app, thread, initial_state(requirement))

        assert result["spent_usd"] == pytest.approx(3.75)


class TestRoleFailures:
    def test_provider_error_escalates_without_consuming_a_retry(
        self, make_graph: Any, make_roles: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """An outage must not be recorded as bad work."""
        roles = make_roles()
        roles[PipelineRole.ARCHITECT] = FailingRole(PipelineRole.ARCHITECT, "rate limited")
        app = build_graph(roles, config, StubExecutor())

        app.invoke(initial_state(requirement), thread)
        result = app.invoke(approve(), thread)

        assert result["status"] is RunStatus.ESCALATED
        assert result["gate_revisions"] == {}
        assert any("role_error" in entry for entry in audit_trail(result))

    def test_wrong_artifact_kind_escalates(
        self, make_roles: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """A role returning another role's artifact must not be handed on."""
        roles = make_roles()
        roles[PipelineRole.ARCHITECT] = ScriptedRole(
            PipelineRole.ARCHITECT,
            lambda context: make_artifact(ArtifactKind.PRD, context),
        )
        app = build_graph(roles, config, StubExecutor())

        app.invoke(initial_state(requirement), thread)
        result = app.invoke(approve(), thread)

        assert result["status"] is RunStatus.ESCALATED
        assert any("wrong_artifact_kind" in entry for entry in audit_trail(result))


class TestGraphConstruction:
    def test_missing_role_is_a_construction_error(self, config: Any, make_roles: Any) -> None:
        """A gap is knowable up front, so it must not wait until mid-run."""
        roles = make_roles()
        del roles[PipelineRole.QA]

        with pytest.raises(GraphConstructionError, match="qa"):
            build_graph(roles, config, StubExecutor())

    def test_all_four_roles_are_required(self, config: Any) -> None:
        with pytest.raises(GraphConstructionError, match="no role implementation"):
            build_graph({}, config, StubExecutor())


class TestGatePayloadHandling:
    def test_bare_verdict_string_is_accepted(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """A reviewer answering "approved" should not have to build a document."""
        app, _ = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        result = app.invoke(Command(resume="approved"), thread)

        assert result["reviews"][StageName.PRD_REVIEW.value].verdict is Verdict.APPROVED

    def test_decision_points_at_the_reviewed_artifact(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """The record must name what was looked at, not a placeholder id."""
        app, _ = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        result = app.invoke(approve(), thread)

        decision: ReviewDecision = result["reviews"][StageName.PRD_REVIEW.value]
        assert decision.artifact_id == result["artifacts"][ArtifactKind.PRD].artifact_id

    def test_unknown_verdict_is_refused_not_defaulted(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """Defaulting an unparseable answer to "approved" would pass a gate silently."""
        app, _ = make_graph(config)

        app.invoke(initial_state(requirement), thread)

        with pytest.raises(GraphConstructionError, match="expected one of"):
            app.invoke(Command(resume="looks fine to me"), thread)

    def test_payload_without_verdict_is_refused(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        app, _ = make_graph(config)

        app.invoke(initial_state(requirement), thread)

        with pytest.raises(GraphConstructionError, match="no 'verdict' key"):
            app.invoke(Command(resume={"reviewer": "alice"}), thread)

    def test_rejection_without_comment_gets_a_placeholder(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """The role must still be told something rather than receiving silence."""
        app, roles = make_graph(config)

        app.invoke(initial_state(requirement), thread)
        app.invoke(Command(resume={"verdict": "rejected"}), thread)

        assert roles[PipelineRole.PRODUCT_MANAGER].calls[1].feedback != ""

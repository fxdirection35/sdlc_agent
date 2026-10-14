"""The pipeline graph: four roles, three human gates, one bounded QA loop.

The topology is fixed by HARNESS_SPEC.md section 2.1: PM -> architect ->
developer -> QA is exactly four handoffs, the ceiling Microsoft's data puts on
multi-agent chains. There is deliberately no fifth role node, and adding one
needs evidence rather than a refactor. The ``verify`` node is not a fifth role:
it runs the suite deterministically and holds no agent.

Two rules shape the layout:

- Gates are their own nodes. LangGraph re-executes a node from the top on
  resume, so an LLM call sharing a node with ``interrupt()`` would be billed
  twice per review (section 2.4).
- Every loop is bounded. QA retries and per-gate revisions both escalate to a
  human on exhaustion, because a loop that only the budget guard can stop is a
  loop that always burns the budget first.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from sdlc_agent.config import PlatformConfig
from sdlc_agent.domain import ArtifactKind, PipelineRole, ReviewDecision, Verdict
from sdlc_agent.domain.product import Requirement
from sdlc_agent.domain.qa import QualityGate
from sdlc_agent.pipeline.codeflow import (
    CodeIntegrations,
    make_apply_node,
    make_checkout_node,
    summarize_file_tree,
)
from sdlc_agent.pipeline.roles import ROLE_INPUTS, ROLE_OUTPUTS, Role, RoleContext, RoleError
from sdlc_agent.pipeline.serde import make_memory_checkpointer
from sdlc_agent.pipeline.state import AuditEvent, PipelineState, RunStatus, StageName
from sdlc_agent.pipeline.verify import TestExecutor, make_verify_node

#: Gate -> (the role it sends work back to, the stage it advances to).
GATE_FLOW: dict[str, tuple[str, str]] = {
    StageName.PRD_REVIEW.value: (StageName.PM.value, StageName.ARCHITECT.value),
    StageName.DESIGN_REVIEW.value: (StageName.ARCHITECT.value, StageName.DEVELOPER.value),
    StageName.MERGE_REVIEW.value: (StageName.DEVELOPER.value, StageName.FINALIZE.value),
}

#: The artifact each gate reviews. Used to attach the human's decision to the
#: thing they actually looked at, rather than to a placeholder id.
GATE_ARTIFACT: dict[str, ArtifactKind] = {
    StageName.PRD_REVIEW.value: ArtifactKind.PRD,
    StageName.DESIGN_REVIEW.value: ArtifactKind.DESIGN,
    StageName.MERGE_REVIEW.value: ArtifactKind.CODE_CHANGE,
}

#: Role -> the node that runs it.
ROLE_STAGE: dict[PipelineRole, str] = {
    PipelineRole.PRODUCT_MANAGER: StageName.PM.value,
    PipelineRole.ARCHITECT: StageName.ARCHITECT.value,
    PipelineRole.DEVELOPER: StageName.DEVELOPER.value,
    PipelineRole.QA: StageName.QA.value,
}

#: How many times one gate may reject before escalating.
MAX_GATE_REVISIONS = 2

#: Roles that see the workspace's file tree. The PM works from the requirement
#: alone, and QA plans against the acceptance criteria and the change - the
#: tree is for the roles that must name real paths in code.
TREE_ROLES: frozenset[PipelineRole] = frozenset({PipelineRole.ARCHITECT, PipelineRole.DEVELOPER})


class GraphConstructionError(RuntimeError):
    """The graph could not be assembled, or a human's decision was unusable."""


def _audit(stage: str, event: str, detail: str = "") -> tuple[AuditEvent, ...]:
    return (AuditEvent(stage=stage, event=event, detail=detail),)


def _ceiling(state: PipelineState, config: PlatformConfig) -> float:
    """Effective budget ceiling for this run.

    A per-run override granted at the budget gate wins over the configured
    default. The config stays untouched so the original value remains available
    for auditing.
    """
    override = state.get("budget_override_usd", 0.0)
    return override if override > 0 else config.budget.per_task_usd


def _gate_reviewing(stage: str) -> str | None:
    """The gate that reviews ``stage``'s output, if any.

    Rejection counts are recorded under the gate's name, but the role that must
    revise is addressed by its own stage name. Without this translation an
    attempt counter looked up by role stage would never find the gate's entries
    and every retry would read as a first attempt.
    """
    for gate, (revise, _advance) in GATE_FLOW.items():
        if revise == stage:
            return gate
    return None


def _attempt_for(stage: str, state: PipelineState) -> int:
    """Which attempt this stage is on, from 1.

    The developer counts QA retries; the roles behind a human gate count that
    gate's rejections. Both are attempts to get the same artifact right, they
    just come from different sources.
    """
    if stage == StageName.DEVELOPER.value:
        return state.get("qa_attempts", 0) + 1
    gate = _gate_reviewing(stage)
    if gate is None:
        return 1
    return state.get("gate_revisions", {}).get(gate, 0) + 1


def _context_for(role: PipelineRole, state: PipelineState) -> RoleContext:
    """Assemble the scoped context one role may see.

    Only the artifacts that role declares as inputs are included, not the whole
    mapping (section 4.1): a role that can read everything can also be
    distracted by everything. The workspace's file tree is computed here, once
    per role call, so every implementation of a role sees the same bounded
    view instead of walking the repository itself.
    """
    artifacts = state.get("artifacts", {})
    stage = ROLE_STAGE[role]
    repo_root = state.get("repo_root", "")
    return RoleContext(
        requirement=state["requirement"],
        upstream={kind: artifacts[kind] for kind in ROLE_INPUTS[role] if kind in artifacts},
        feedback=state.get("feedback", ""),
        attempt=_attempt_for(stage, state),
        repo_root=repo_root,
        repo_tree=summarize_file_tree(Path(repo_root)) if repo_root and role in TREE_ROLES else "",
    )


def _make_role_node(role: PipelineRole, impl: Role, config: PlatformConfig) -> Any:
    """Build the node that runs one role.

    Budget is checked BEFORE the call. Checking afterwards means the call that
    breaches the ceiling has already been paid for, and on a looping run the
    breach compounds.
    """
    stage = ROLE_STAGE[role]
    output_kind = ROLE_OUTPUTS[role]

    def node(state: PipelineState) -> dict[str, Any]:
        ceiling = _ceiling(state, config)
        spent = state.get("spent_usd", 0.0)

        if spent >= ceiling:
            # Do not call the model. Record where we stopped so the budget gate
            # can resume this exact node once a human raises the ceiling.
            return {
                "blocked_stage": stage,
                "status": RunStatus.BUDGET_EXCEEDED,
                "audit": _audit(
                    stage, "blocked_by_budget", f"spent ${spent:.4f} of ${ceiling:.4f} ceiling"
                ),
            }

        try:
            result = impl.run(_context_for(role, state))
        except RoleError as exc:
            # A provider failure is not a quality failure: it must not consume a
            # QA retry or a gate revision, or an outage reads as bad work.
            return {
                "status": RunStatus.ESCALATED,
                "audit": _audit(stage, "role_error", str(exc)),
            }

        if result.artifact.kind is not output_kind:
            # The wrong artifact type would be handed on as if it were right.
            return {
                "status": RunStatus.ESCALATED,
                "audit": _audit(
                    stage,
                    "wrong_artifact_kind",
                    f"expected {output_kind.value}, got {result.artifact.kind.value}",
                ),
            }

        return {
            "artifacts": {output_kind: result.artifact},
            "spent_usd": result.spent_usd,
            # Cleared so a later gate does not act on a stale rejection comment.
            "feedback": "",
            "audit": _audit(
                stage,
                "produced",
                f"{output_kind.value} ({result.output_tokens} tokens, ${result.spent_usd:.4f})",
            ),
        }

    return node


def _decision_from(gate: str, raw: Any, state: PipelineState) -> ReviewDecision:
    """Turn a human's resume payload into a ReviewDecision.

    Accepts a bare verdict string or a mapping, because a reviewer answering
    "approved" should not have to construct a document. An unusable payload is
    refused rather than defaulted: defaulting to approval would turn a malformed
    resume into a silently passed gate.

    The artifact id comes from the gate's own artifact when the reviewer did not
    supply one, so the decision points at the thing that was reviewed.
    """
    if isinstance(raw, str):
        payload: dict[str, Any] = {"verdict": raw}
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        raise GraphConstructionError(
            f"gate {gate!r} cannot interpret a resume payload of type {type(raw).__name__}"
        )

    if "verdict" not in payload:
        raise GraphConstructionError(f"gate {gate!r} received a decision with no 'verdict' key")

    artifact_id = payload.get("artifact_id")
    if artifact_id is None:
        reviewed = state.get("artifacts", {}).get(GATE_ARTIFACT[gate])
        if reviewed is None:
            raise GraphConstructionError(
                f"gate {gate!r} has no {GATE_ARTIFACT[gate].value} artifact to review"
            )
        artifact_id = reviewed.artifact_id

    try:
        verdict = Verdict(str(payload["verdict"]).strip().lower())
    except ValueError:
        known = ", ".join(item.value for item in Verdict)
        raise GraphConstructionError(
            f"gate {gate!r} got verdict {payload['verdict']!r}; expected one of: {known}"
        ) from None

    return ReviewDecision(
        artifact_id=artifact_id,
        reviewer=str(payload.get("reviewer", "unspecified")),
        verdict=verdict,
        comment=str(payload.get("comment", "")),
    )


def _make_gate_node(gate: StageName, config: PlatformConfig) -> Any:
    """Build a human-review gate.

    The node only interrupts and records; it does no work of its own, so
    re-execution on resume is harmless.
    """
    gate_name = gate.value

    def node(state: PipelineState) -> dict[str, Any]:
        decision = _decision_from(
            gate_name,
            interrupt(
                {
                    "gate": gate_name,
                    "prompt": f"Review the {gate_name.removesuffix('_review')} and decide.",
                    "spent_usd": state.get("spent_usd", 0.0),
                    "budget_usd": _ceiling(state, config),
                    "allowed_verdicts": [item.value for item in Verdict],
                }
            ),
            state,
        )

        if decision.verdict is Verdict.APPROVED:
            return {
                "reviews": {gate_name: decision},
                "audit": _audit(gate_name, "approved", decision.comment),
            }

        revisions = dict(state.get("gate_revisions", {}))
        revisions[gate_name] = revisions.get(gate_name, 0) + 1

        if decision.verdict is Verdict.NEEDS_HUMAN or revisions[gate_name] > MAX_GATE_REVISIONS:
            return {
                "reviews": {gate_name: decision},
                "gate_revisions": revisions,
                "status": RunStatus.ESCALATED,
                "audit": _audit(
                    gate_name,
                    "escalated",
                    f"{decision.verdict.value} after {revisions[gate_name]} rejection(s)",
                ),
            }

        return {
            "reviews": {gate_name: decision},
            "gate_revisions": revisions,
            "feedback": decision.comment or "Rejected without a comment.",
            "audit": _audit(
                gate_name,
                "rejected",
                f"revision {revisions[gate_name]}/{MAX_GATE_REVISIONS}: {decision.comment}",
            ),
        }

    return node


def _budget_review_node(state: PipelineState) -> dict[str, Any]:
    """Interrupt so a human can raise the ceiling, or let the run end.

    Reached only from the budget guard, so the run is already over its limit.
    """
    raw = interrupt(
        {
            "gate": StageName.BUDGET_REVIEW.value,
            "prompt": "Run exceeded its budget. Raise the ceiling or abort.",
            "spent_usd": state.get("spent_usd", 0.0),
            "blocked_stage": state.get("blocked_stage", ""),
        }
    )
    grant = float(raw.get("grant_usd", 0.0)) if isinstance(raw, Mapping) else 0.0
    if grant > 0:
        return {
            "budget_override_usd": grant,
            # No longer terminal: the blocked node gets another chance.
            "status": RunStatus.IN_PROGRESS,
            "audit": _audit(
                StageName.BUDGET_REVIEW.value, "ceiling_raised", f"new ceiling ${grant:.2f}"
            ),
        }
    return {
        "status": RunStatus.BUDGET_EXCEEDED,
        "audit": _audit(StageName.BUDGET_REVIEW.value, "aborted", "no additional budget granted"),
    }


def _escalate_node(state: PipelineState) -> dict[str, Any]:
    return {
        "status": RunStatus.ESCALATED,
        "audit": _audit(StageName.ESCALATE.value, "escalated", "handed to a human"),
    }


def _finalize_node(state: PipelineState) -> dict[str, Any]:
    return {
        "status": RunStatus.COMPLETED,
        "audit": _audit(
            StageName.FINALIZE.value,
            "completed",
            f"total spend ${state.get('spent_usd', 0.0):.4f}",
        ),
    }


def _route_after_checkout(state: PipelineState) -> str:
    """A failed clone escalates; anything else proceeds to the PM.

    The checkout node also *skips* (synthetic sources) - skipping is a normal,
    audited outcome, not a failure, and the run carries on artifact-only.
    """
    if state.get("status") is RunStatus.ESCALATED:
        return StageName.ESCALATE.value
    return StageName.PM.value


def _route_after_apply(state: PipelineState) -> str:
    """A failed apply escalates; a skipped or applied one proceeds to QA.

    QA plans against the workspace as it now stands, so it must see the code
    the apply node actually wrote, not the claims it was handed.
    """
    if state.get("status") is RunStatus.ESCALATED:
        return StageName.ESCALATE.value
    return StageName.QA.value


def _route_after_role(next_stage: str) -> Any:
    """Onward, to the budget gate, or to escalation."""

    def route(state: PipelineState) -> str:
        status = state.get("status")
        if status is RunStatus.BUDGET_EXCEEDED:
            return StageName.BUDGET_REVIEW.value
        if status is RunStatus.ESCALATED:
            return StageName.ESCALATE.value
        return next_stage

    return route


def _route_after_gate(gate: str) -> Any:
    """Advance, send back for revision, or escalate."""
    revise, advance = GATE_FLOW[gate]

    def route(state: PipelineState) -> str:
        if state.get("status") is RunStatus.ESCALATED:
            return StageName.ESCALATE.value
        decision = state.get("reviews", {}).get(gate)
        if decision is None:
            # A gate that recorded nothing cannot have been approved.
            return StageName.ESCALATE.value
        if decision.verdict is Verdict.APPROVED:
            return advance
        if decision.verdict is Verdict.NEEDS_HUMAN:
            return StageName.ESCALATE.value
        return revise

    return route


def _route_after_verify(max_qa_retries: int) -> Any:
    """Merge gate on a pass, back to the developer on a bounded fail.

    The bound is enforced here, in the flow that owns it, rather than inside the
    QA role - a role implementation cannot forget a limit it does not hold.
    ``NEEDS_HUMAN`` escalates immediately: missing evidence is a decision for a
    person, not something to retry.
    """

    def route(state: PipelineState) -> str:
        status = state.get("status")
        if status is RunStatus.BUDGET_EXCEEDED:
            return StageName.BUDGET_REVIEW.value
        if status is RunStatus.ESCALATED:
            return StageName.ESCALATE.value

        gate = state.get("artifacts", {}).get(ArtifactKind.QUALITY_GATE)
        if not isinstance(gate, QualityGate):
            return StageName.ESCALATE.value

        if gate.verdict is Verdict.APPROVED:
            return StageName.MERGE_REVIEW.value
        if gate.verdict is Verdict.NEEDS_HUMAN:
            return StageName.ESCALATE.value

        # REJECTED. qa_attempts was incremented by the verify node, so after the
        # first failure it is 1; allowing attempts <= max gives exactly
        # max_qa_retries chances to fix it.
        if state.get("qa_attempts", 0) > max_qa_retries:
            return StageName.ESCALATE.value
        return StageName.DEVELOPER.value

    return route


def _route_after_budget_review(state: PipelineState) -> str:
    """Resume the blocked role, or end the run."""
    if state.get("status") is not RunStatus.IN_PROGRESS:
        return END
    blocked = state.get("blocked_stage", "")
    if blocked in set(ROLE_STAGE.values()):
        return blocked
    # Nothing to resume means the run cannot continue.
    return END


def build_graph(
    roles: Mapping[PipelineRole, Role],
    config: PlatformConfig,
    executor: TestExecutor,
    *,
    checkpointer: Any = None,
    code: CodeIntegrations | None = None,
) -> Any:
    """Assemble and compile the pipeline graph.

    Args:
        roles: One implementation per pipeline role. All four are required; a
            gap is a construction error rather than a run that fails halfway,
            because it is knowable up front.
        config: Supplies the budget ceiling and the QA retry limit.
        executor: Runs test plans. P1 uses a stub, P2 a sandbox executor.
        checkpointer: Where state persists across interrupts. Defaults to
            in-memory, which suits tests and single-process runs; a deployed
            service passes the Postgres one so a gate survives a restart.
        code: The real-code bundle. ``None`` builds the artifact-only topology
            of the eval harness and the tests that predate code flow; passing
            it inserts the checkout node ahead of the PM and (in later nodes)
            grounds the run in a real repository workspace.

    Raises:
        GraphConstructionError: If a role implementation is missing.
    """
    missing = [role.value for role in PipelineRole if role not in roles]
    if missing:
        raise GraphConstructionError(f"no role implementation for: {', '.join(missing)}")

    graph = StateGraph(PipelineState)

    for role in PipelineRole:
        graph.add_node(ROLE_STAGE[role], _make_role_node(role, roles[role], config))
    for gate in GATE_FLOW:
        graph.add_node(gate, _make_gate_node(StageName(gate), config))
    graph.add_node(StageName.VERIFY.value, make_verify_node(executor, StageName.VERIFY.value))
    graph.add_node(StageName.BUDGET_REVIEW.value, _budget_review_node)
    graph.add_node(StageName.ESCALATE.value, _escalate_node)
    graph.add_node(StageName.FINALIZE.value, _finalize_node)

    if code is None:
        graph.add_edge(START, StageName.PM.value)
    else:
        graph.add_node(StageName.CHECKOUT.value, make_checkout_node(code, StageName.CHECKOUT.value))
        graph.add_edge(START, StageName.CHECKOUT.value)
        graph.add_conditional_edges(
            StageName.CHECKOUT.value,
            _route_after_checkout,
            [StageName.PM.value, StageName.ESCALATE.value],
        )
        # The developer hands file-edit claims to the patch bridge, which
        # commits them before QA plans against the resulting code.
        graph.add_node(StageName.APPLY.value, make_apply_node(code, StageName.APPLY.value))
        graph.add_conditional_edges(
            StageName.APPLY.value,
            _route_after_apply,
            [StageName.QA.value, StageName.ESCALATE.value],
        )

    role_next = {
        StageName.PM.value: StageName.PRD_REVIEW.value,
        StageName.ARCHITECT.value: StageName.DESIGN_REVIEW.value,
        StageName.DEVELOPER.value: (
            StageName.APPLY.value if code is not None else StageName.QA.value
        ),
    }
    for stage, next_stage in role_next.items():
        graph.add_conditional_edges(
            stage,
            _route_after_role(next_stage),
            [next_stage, StageName.BUDGET_REVIEW.value, StageName.ESCALATE.value],
        )

    for gate, (revise, advance) in GATE_FLOW.items():
        graph.add_conditional_edges(
            gate,
            _route_after_gate(gate),
            [revise, advance, StageName.ESCALATE.value],
        )

    # QA plans; verify executes and derives the verdict.
    graph.add_conditional_edges(
        StageName.QA.value,
        _route_after_role(StageName.VERIFY.value),
        [StageName.VERIFY.value, StageName.BUDGET_REVIEW.value, StageName.ESCALATE.value],
    )
    graph.add_conditional_edges(
        StageName.VERIFY.value,
        _route_after_verify(config.pipeline.max_qa_retries),
        [
            StageName.MERGE_REVIEW.value,
            StageName.DEVELOPER.value,
            StageName.ESCALATE.value,
            StageName.BUDGET_REVIEW.value,
        ],
    )

    graph.add_conditional_edges(
        StageName.BUDGET_REVIEW.value,
        _route_after_budget_review,
        [*ROLE_STAGE.values(), END],
    )
    graph.add_edge(StageName.ESCALATE.value, END)
    graph.add_edge(StageName.FINALIZE.value, END)

    return graph.compile(checkpointer=checkpointer or make_memory_checkpointer())


def initial_state(requirement: Requirement) -> dict[str, Any]:
    """Build the state a run starts from.

    Only the requirement is supplied. Everything else starts empty so a node
    that reads before writing fails loudly instead of seeing a stale value left
    over from a previous run on the same thread.
    """
    return {
        "requirement": requirement,
        "artifacts": {},
        "reviews": {},
        "audit": (),
        "spent_usd": 0.0,
        "qa_attempts": 0,
        "gate_revisions": {},
        "feedback": "",
        "repo_root": "",
        "status": RunStatus.IN_PROGRESS,
        "blocked_stage": "",
        "budget_override_usd": 0.0,
    }

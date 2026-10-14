"""Pipeline state: what flows between nodes, and how updates combine.

State is a TypedDict because that is what LangGraph reduces over, but the
artifacts inside it stay strongly typed - a handoff that degraded to a dict of
unknowns would lose exactly the guarantee HARNESS_SPEC.md section 2.1 exists to
provide.

Artifacts are collected in one mapping keyed by kind rather than eight optional
fields. The QA loop re-runs the developer, and each retry must replace the
previous attempt's artifact instead of accumulating a list nobody reads.
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sdlc_agent.domain import Artifact, ArtifactKind, ReviewDecision
from sdlc_agent.domain.base import reject_blank
from sdlc_agent.domain.product import Requirement


class RunStatus(StrEnum):
    """Terminal or in-progress state of one pipeline run."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    BUDGET_EXCEEDED = "budget_exceeded"


class StageName(StrEnum):
    """Node names in the compiled graph.

    Gates are separate nodes from the roles that produce what they review,
    because LangGraph re-executes a node from the top on resume (see
    HARNESS_SPEC.md section 2.4). Merging them would bill every LLM call twice
    per human review.
    """

    PM = "pm"
    PRD_REVIEW = "prd_review"
    ARCHITECT = "architect"
    DESIGN_REVIEW = "design_review"
    DEVELOPER = "developer"
    #: Deterministic patch bridge: the developer's file-edit claims become
    #: commits in the workspace. Present only when code integrations are on.
    APPLY = "apply"
    QA = "qa"
    #: Deterministic clone of the target repository into a per-run workspace.
    #: Present only when the graph was built with code integrations.
    CHECKOUT = "checkout"
    #: Deterministic test execution, not an agent handoff. Kept separate from
    #: QA so the verdict is evidence produced by running the suite rather than
    #: something the QA agent asserts about its own plan.
    VERIFY = "verify"
    MERGE_REVIEW = "merge_review"
    BUDGET_REVIEW = "budget_review"
    ESCALATE = "escalate"
    FINALIZE = "finalize"


class AuditEvent(BaseModel):
    """One entry in the run's audit trail.

    Transitions that are not human gates are automatic, but they are still
    recorded: an audit trail that only covers approvals cannot explain why a run
    looped three times before it was escalated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str = Field(min_length=1)
    event: str = Field(min_length=1)
    detail: str = ""
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("stage", "event")
    @classmethod
    def _strip(cls, value: str) -> str:
        return reject_blank(value)


def _merge_mappings(left: dict[Any, Any], right: dict[Any, Any]) -> dict[Any, Any]:
    """Reducer that replaces per key rather than replacing the whole mapping.

    Without this, a node returning one artifact would erase the artifacts every
    earlier node had written.
    """
    return {**left, **right}


def _merge_audit(
    left: tuple[AuditEvent, ...], right: tuple[AuditEvent, ...]
) -> tuple[AuditEvent, ...]:
    return (*left, *right)


class PipelineState(TypedDict):
    """Everything the graph carries from requirement to verdict.

    Annotated fields use a reducer, so a node returns only what it changed. The
    rest are replaced wholesale, which is what ``status`` and ``feedback`` want:
    feedback from a rejected gate must not survive into the next approval.
    """

    #: The requirement that started the run. Immutable for the run's lifetime.
    requirement: Requirement
    #: Latest artifact of each kind produced so far.
    artifacts: Annotated[dict[ArtifactKind, Artifact], _merge_mappings]
    #: Human decisions, keyed by which gate made them.
    reviews: Annotated[dict[str, ReviewDecision], _merge_mappings]
    #: Append-only trail of every stage transition.
    audit: Annotated[tuple[AuditEvent, ...], _merge_audit]
    #: Accumulated LLM cost for this run, in USD.
    spent_usd: Annotated[float, operator.add]
    #: How many times the developer has been sent back by QA. Bounded by
    #: ``config.pipeline.max_qa_retries``.
    qa_attempts: int
    #: Rejections per gate, so a reviewer who keeps rejecting escalates to a
    #: human rather than looping forever.
    gate_revisions: Annotated[dict[str, int], _merge_mappings]
    #: The rejection comment handed back to the role that must revise.
    feedback: str
    #: Workspace directory of the cloned target repository, empty when the run
    #: has none (no code integrations, or a synthetic eval source). Carried in
    #: state rather than re-derived so a resumed process finds the same path
    #: the run started with.
    repo_root: str
    #: Terminal outcome once set.
    status: RunStatus
    #: Which role node was blocked by the budget guard, so the budget review
    #: can route back to it after a human raises the ceiling.
    blocked_stage: str
    #: Per-run ceiling granted by a human at the budget gate. Zero means "use
    #: the configured default". Stored in state rather than mutating the config
    #: because the config is frozen and shared across runs; an override is a
    #: property of one run and belongs in its audit trail.
    budget_override_usd: float

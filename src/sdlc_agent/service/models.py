"""API schemas: the wire shapes of the service, separate from the domain.

These models deliberately mirror the domain rather than reusing it. The domain
is frozen by its own tests and the eval set; the wire format is owned by the
SPA and may drift (extra summary fields, flattened enums). Coupling them would
make every API change a domain change and vice versa - the same layering rule
that keeps the pipeline off the eval package.

What each endpoint accepts and returns is validated here, so a malformed
request dies at the boundary with a 422 instead of reaching the graph, where
the equivalent failure is a :class:`GraphConstructionError` nobody asked for.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sdlc_agent.domain.base import reject_blank

#: The two things a run can pause for. Kept as plain strings rather than an
#: enum so the wire format stays open to future pause kinds without a
#: breaking change everywhere ``pending`` appears.
GATE_PAUSE = "gate"
BUDGET_PAUSE = "budget"


class RunCreateRequest(BaseModel):
    """A requester asking for work; becomes the run's :class:`Requirement`."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    requester: str = Field(min_length=1)
    source_repo: str = Field(min_length=1)

    @field_validator("title", "description", "requester", "source_repo")
    @classmethod
    def _strip(cls, value: str) -> str:
        # Same rule as the domain: reject_blank, so the service cannot mint a
        # Requirement the graph would refuse, nor a blank one it would accept
        # only because validation happens to run before the strip.
        return reject_blank(value)


class PendingDecision(BaseModel):
    """What the run is stopped on, presented for a human to answer.

    ``kind`` tells the client which endpoint answers it: ``gate`` goes to
    ``POST /api/runs/{id}/review``, ``budget`` to ``POST /api/runs/{id}/budget``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    gate: str
    prompt: str
    spent_usd: float
    budget_usd: float | None = None
    allowed_verdicts: tuple[str, ...] = ()
    blocked_stage: str = ""


class AuditEventOut(BaseModel):
    """One audit entry, flattened for display."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    event: str
    detail: str
    at: str


class RequirementOut(BaseModel):
    """The requirement the run serves, as entered."""

    model_config = ConfigDict(extra="forbid")

    title: str
    description: str
    requester: str
    source_repo: str


class RunResponse(BaseModel):
    """The state of one run at a point in time.

    This is the response of every state-advancing call (create, review, budget)
    and of ``GET /api/runs/{id}``: the client always learns where the run now
    stands and what, if anything, it waits for.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str
    pending: PendingDecision | None
    spent_usd: float
    budget_usd: float
    requirement: RequirementOut
    #: artifact kind -> artifact id. Full content lives on the review view,
    #: which knows which artifact is under discussion.
    artifacts: dict[str, str]
    audit: tuple[AuditEventOut, ...]


class ReviewRequest(BaseModel):
    """A human's verdict at a gate."""

    model_config = ConfigDict(extra="forbid")

    verdict: str = Field(min_length=1)
    reviewer: str = ""
    comment: str = ""

    @field_validator("verdict")
    @classmethod
    def _verdict(cls, value: str) -> str:
        # Stripped and lowercased here so the graph's own normalization never
        # has to save the request; the API rejects what the graph would reject.
        return reject_blank(value).strip().lower()


class BudgetDecisionRequest(BaseModel):
    """A human's answer at the budget gate.

    ``grant_usd`` is the *additional* ceiling for this run: zero or omitted
    means "no more money", which ends the run as ``budget_exceeded`` - that is
    a decision, not a refusal to answer, so it is a valid 200 payload.
    """

    model_config = ConfigDict(extra="forbid")

    grant_usd: float = Field(default=0.0, ge=0)


class RunSummary(BaseModel):
    """One row of the run list: everything a reviewer scanning for work needs.

    Leaner than :class:`RunResponse` on purpose - no audit trail, no artifact
    ids - because the list is polled and rendered often; the detail view is one
    click away.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str
    #: "" when the run waits on nothing (running or terminal).
    pending_kind: str
    pending_gate: str
    spent_usd: float
    budget_usd: float
    title: str
    requester: str
    #: ISO timestamp of the run's latest checkpoint - the list's sort key.
    updated_at: str


class ReviewView(BaseModel):
    """A gate decision in context: the payload plus the artifact under review."""

    model_config = ConfigDict(extra="forbid")

    pending: PendingDecision
    artifact_kind: str
    artifact: dict[str, Any]


__all__ = [
    "BUDGET_PAUSE",
    "GATE_PAUSE",
    "AuditEventOut",
    "BudgetDecisionRequest",
    "PendingDecision",
    "RequirementOut",
    "ReviewRequest",
    "ReviewView",
    "RunCreateRequest",
    "RunResponse",
    "RunSummary",
]

"""Common building blocks shared by every pipeline artifact.

Every handoff in the pipeline carries a strongly typed artifact rather than
free text (HARNESS_SPEC.md section 2.1). The base classes here give all
artifacts the identity, versioning and serialization behavior they need to be
written to files inside the sandbox repository and indexed in the database
(HARNESS_SPEC.md section 6).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


def reject_blank(value: str) -> str:
    """Reject strings that are empty or whitespace only.

    ``Field(min_length=1)`` alone accepts ``"   "``, which would silently
    produce an empty artifact several steps downstream. Every text field in the
    domain layer runs its value through here so the failure happens at
    construction, close to its cause.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError("must contain non-whitespace characters")
    return stripped


class ArtifactKind(StrEnum):
    """The artifact types that flow between pipeline roles.

    The order matches the pipeline: each kind is produced by one role and
    consumed by the next. Keep this list in sync with ``PipelineStage``.
    """

    REQUIREMENT = "requirement"
    PRD = "prd"
    DESIGN = "design"
    TASK_BREAKDOWN = "task_breakdown"
    CODE_CHANGE = "code_change"
    TEST_PLAN = "test_plan"
    TEST_RUN = "test_run"
    QUALITY_GATE = "quality_gate"


class Artifact(BaseModel):
    """Base class for every pipeline artifact.

    Attributes:
        artifact_id: Stable identifier that survives serialization, so the
            database index can point at the file copy across restarts.
        kind: Discriminator used when loading an artifact whose concrete type
            is not known up front.
        created_at: UTC timestamp set at construction time.
        schema_version: Bumped when the artifact shape changes in a way that
            older readers cannot parse; enables forward-compatible storage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    kind: ArtifactKind
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: int = 1

    def to_json(self) -> str:
        """Serialize for writing to a markdown front matter block or the DB."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Round-trip counterpart of :meth:`to_json`.

        Returns ``Self`` rather than ``Artifact`` so that calling this on a
        concrete subclass keeps the subclass's fields visible to type checkers:
        ``Requirement.from_json(...)`` yields a ``Requirement``, not a bare
        artifact that has to be narrowed before use.
        """
        return cls.model_validate_json(payload)


class Verdict(StrEnum):
    """Outcome of a review or verification step."""

    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_HUMAN = "needs_human"


class ReviewDecision(BaseModel):
    """A recorded decision at one of the three human gates.

    The pipeline has exactly three hard gates (PRD review, design review, code
    merge); every other transition is automatic but still logged for audit.
    This model is what the audit log stores.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: uuid.UUID
    reviewer: str
    verdict: Verdict
    comment: str = ""
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

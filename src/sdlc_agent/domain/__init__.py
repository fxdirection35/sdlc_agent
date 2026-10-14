"""Pipeline artifacts: the typed contracts that flow between role agents.

Each artifact is produced by exactly one role and consumed by the next, so the
module layout mirrors the pipeline itself:

    Requirement -> ProductRequirements -> Design -> Task -> CodeChange
                -> TestPlan -> TestRun -> QualityGate

Handoffs carry these models rather than free text (HARNESS_SPEC.md
section 2.1). The concrete artifact a run holds is recoverable from
:data:`ARTIFACT_TYPES` via the ``kind`` discriminator stored on every artifact.
"""

from __future__ import annotations

import json

from sdlc_agent.domain.base import (
    Artifact,
    ArtifactKind,
    PipelineRole,
    ReviewDecision,
    Verdict,
    reject_blank,
)
from sdlc_agent.domain.engineering import CodeChange, Design, FileEdit, Task
from sdlc_agent.domain.product import AcceptanceCriterion, ProductRequirements, Requirement
from sdlc_agent.domain.qa import (
    QualityGate,
    TestCaseResult,
    TestPlan,
    TestPlanItem,
    TestRun,
    TestStatus,
)

ARTIFACT_TYPES: dict[ArtifactKind, type[Artifact]] = {
    ArtifactKind.REQUIREMENT: Requirement,
    ArtifactKind.PRD: ProductRequirements,
    ArtifactKind.DESIGN: Design,
    ArtifactKind.TASK_BREAKDOWN: Task,
    ArtifactKind.CODE_CHANGE: CodeChange,
    ArtifactKind.TEST_PLAN: TestPlan,
    ArtifactKind.TEST_RUN: TestRun,
    ArtifactKind.QUALITY_GATE: QualityGate,
}


def load_artifact(payload: str) -> Artifact:
    """Deserialize an artifact whose concrete type is not known up front.

    Artifacts live as files inside the sandbox repository while the database
    only holds pointers to them (HARNESS_SPEC.md section 6), so a reader often
    has JSON in hand without knowing which model produced it. The ``kind``
    field stored on every artifact resolves that.

    Dispatch reads ``kind`` out of the raw JSON instead of validating against
    :class:`Artifact` first. The base class forbids extra fields, so probing
    through it would reject every field belonging to a concrete subclass.

    Args:
        payload: JSON produced by :meth:`Artifact.to_json`.

    Returns:
        The artifact as its concrete subclass, fully validated.

    Raises:
        ValueError: If the payload is not a JSON object, carries no ``kind``,
            or names a kind this version does not know. The message names the
            offending value so a schema-version mismatch is diagnosable from
            the error alone.
        pydantic.ValidationError: If the payload does not match the schema of
            the type its ``kind`` selects.
    """
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"artifact payload is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"artifact payload must be a JSON object, got {type(raw).__name__}")

    kind_value = raw.get("kind")
    if kind_value is None:
        raise ValueError("artifact payload has no 'kind' field; cannot resolve its type")

    try:
        kind = ArtifactKind(kind_value)
    except ValueError:
        known = ", ".join(member.value for member in ArtifactKind)
        raise ValueError(f"unknown artifact kind {kind_value!r}; known kinds: {known}") from None

    artifact_type = ARTIFACT_TYPES.get(kind)
    if artifact_type is None:
        raise ValueError(f"no artifact type registered for kind {kind!r}")

    return artifact_type.model_validate(raw)


__all__ = [
    "ARTIFACT_TYPES",
    "AcceptanceCriterion",
    "Artifact",
    "ArtifactKind",
    "CodeChange",
    "Design",
    "FileEdit",
    "PipelineRole",
    "ProductRequirements",
    "QualityGate",
    "Requirement",
    "ReviewDecision",
    "Task",
    "TestCaseResult",
    "TestPlan",
    "TestPlanItem",
    "TestRun",
    "TestStatus",
    "Verdict",
    "load_artifact",
    "reject_blank",
]

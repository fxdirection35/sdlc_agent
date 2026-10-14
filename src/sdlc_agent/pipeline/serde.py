"""Checkpoint serialization: which types may be deserialized from a checkpoint.

LangGraph stores pipeline state as msgpack and reconstructs it on every resume.
Two problems follow from that, and both are solved here rather than left to
default behaviour.

**Future compatibility.** By default LangGraph allows every type but warns that
unregistered types "will be blocked in a future version". Our artifacts are
pydantic models, so the warning fires on every resume and a LangGraph upgrade
would turn checkpointing into a hard failure - taking the human gates down with
it, since a gate only survives a restart because of its checkpoint.

**Security.** The serializer's own docs warn that if an attacker can write to
the checkpoint store they may trigger code execution during deserialization.
Declaring an explicit allowlist narrows that to the types the pipeline actually
stores, instead of anything importable.

Note ``with_msgpack_allowlist()`` cannot be used for this: it returns the
serializer unchanged when the base allowlist is the permissive ``True`` default.
The allowlist must be passed to the constructor.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from sdlc_agent.domain import ARTIFACT_TYPES
from sdlc_agent.domain.base import ArtifactKind, PipelineRole, ReviewDecision, Verdict
from sdlc_agent.domain.product import AcceptanceCriterion
from sdlc_agent.domain.qa import TestCaseResult, TestPlanItem, TestStatus
from sdlc_agent.pipeline.state import AuditEvent, RunStatus, StageName

#: Every type the pipeline state can contain.
#:
#: Derived from ARTIFACT_TYPES where possible so a new artifact kind is
#: registered automatically; the supporting value types are listed explicitly.
#: Standard-library types that appear in state (``uuid.UUID``, ``datetime``,
#: ``Decimal``) are handled by the serializer itself and must not be listed.
_STATE_TYPES: tuple[type, ...] = (
    *ARTIFACT_TYPES.values(),
    # Nested value objects inside artifacts.
    AcceptanceCriterion,
    TestCaseResult,
    TestPlanItem,
    # Enums used as state values and as artifact dict keys.
    ArtifactKind,
    PipelineRole,
    Verdict,
    TestStatus,
    RunStatus,
    StageName,
    # State-level records.
    AuditEvent,
    ReviewDecision,
)


def msgpack_allowlist() -> list[tuple[str, str]]:
    """The ``(module, qualname)`` pairs LangGraph needs to rebuild our state.

    Returns a sorted list rather than a set because the serializer normalizes
    either, and a stable order makes diffs and log output readable.
    """
    return sorted({(type_.__module__, type_.__qualname__) for type_ in _STATE_TYPES})


def make_serializer() -> JsonPlusSerializer:
    """Build the serializer used for every checkpointer in this project."""
    return JsonPlusSerializer(allowed_msgpack_modules=msgpack_allowlist())


def make_memory_checkpointer() -> InMemorySaver:
    """An in-memory checkpointer with the allowlist applied.

    For tests and single-process runs. A deployed service builds a Postgres
    checkpointer with the same serializer so gates survive a restart.
    """
    return InMemorySaver(serde=make_serializer())


__all__ = [
    "make_memory_checkpointer",
    "make_serializer",
    "msgpack_allowlist",
]

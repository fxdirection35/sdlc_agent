"""The pipeline: role interface, graph topology and test execution.

P1 runs on stub roles so the flow can be verified without an API key. The graph
itself is the product; swapping stubs for litellm-backed roles in P2 does not
change it, because nothing in the topology depends on which implementation
produced an artifact.
"""

from __future__ import annotations

from sdlc_agent.pipeline.graph import (
    GATE_ARTIFACT,
    GATE_FLOW,
    MAX_GATE_REVISIONS,
    ROLE_STAGE,
    GraphConstructionError,
    build_graph,
    initial_state,
)
from sdlc_agent.pipeline.postgres import postgres_checkpointer
from sdlc_agent.pipeline.roles import (
    ROLE_INPUTS,
    ROLE_OUTPUTS,
    FailingRole,
    Role,
    RoleContext,
    RoleError,
    RoleResult,
    ScriptedRole,
)
from sdlc_agent.pipeline.serde import (
    make_memory_checkpointer,
    make_serializer,
    msgpack_allowlist,
)
from sdlc_agent.pipeline.state import AuditEvent, PipelineState, RunStatus, StageName
from sdlc_agent.pipeline.verify import StubExecutor, TestExecutor, make_verify_node

__all__ = [
    "GATE_ARTIFACT",
    "GATE_FLOW",
    "MAX_GATE_REVISIONS",
    "ROLE_INPUTS",
    "ROLE_OUTPUTS",
    "ROLE_STAGE",
    "AuditEvent",
    "FailingRole",
    "GraphConstructionError",
    "PipelineState",
    "Role",
    "RoleContext",
    "RoleError",
    "RoleResult",
    "RunStatus",
    "ScriptedRole",
    "StageName",
    "StubExecutor",
    "TestExecutor",
    "build_graph",
    "initial_state",
    "make_memory_checkpointer",
    "make_serializer",
    "make_verify_node",
    "msgpack_allowlist",
    "postgres_checkpointer",
]

"""Fixtures for the pipeline graph tests.

Roles are scripted rather than LLM-backed on purpose: these tests exercise state
transitions, gate behaviour and loop bounds, none of which should depend on a
model call. That is also what makes them free, fast and non-flaky enough to run
on every commit.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from functools import partial
from typing import Any

import pytest

from sdlc_agent.config import PlatformConfig
from sdlc_agent.domain import PipelineRole
from sdlc_agent.pipeline import ScriptedRole, StubExecutor, build_graph

from .helpers import ROLE_KINDS, make_artifact, make_requirement


@pytest.fixture
def requirement() -> Any:
    return make_requirement()


@pytest.fixture
def config() -> PlatformConfig:
    """A config with small limits, so bounds can be exercised in a few steps."""
    return PlatformConfig.model_validate(
        {
            "providers": {"anthropic": {"api_key_env": "ANTHROPIC_API_KEY"}},
            "roles": {
                role.value: {"provider": "anthropic", "model": "claude-opus-5-5"}
                for role in PipelineRole
            },
            "budget": {"per_task_usd": 5.0},
            "pipeline": {"max_qa_retries": 2},
        }
    )


@pytest.fixture
def make_roles() -> Callable[..., dict[PipelineRole, ScriptedRole]]:
    """Build the four scripted roles, with per-role overrides.

    Overrides exist so a test can make one role cost money, call an unexpected
    tool, or emit more tokens, without rebuilding the others.
    """

    def factory(
        *,
        spent_usd: dict[PipelineRole, float] | None = None,
        tools: dict[PipelineRole, tuple[str, ...]] | None = None,
        output_tokens: dict[PipelineRole, int] | None = None,
    ) -> dict[PipelineRole, ScriptedRole]:
        costs = spent_usd or {}
        tool_sets = tools or {}
        token_counts = output_tokens or {}
        return {
            role: ScriptedRole(
                role,
                # partial over an untyped lambda: it carries make_artifact's real
                # signature, so mypy can check the factory contract.
                partial(make_artifact, kind),
                spent_usd=costs.get(role, 0.0),
                tools_called=tool_sets.get(role, ("read_file",)),
                output_tokens=token_counts.get(role, 500),
            )
            for role, kind in ROLE_KINDS.items()
        }

    return factory


@pytest.fixture
def make_graph(
    make_roles: Callable[..., dict[PipelineRole, ScriptedRole]],
) -> Callable[..., tuple[Any, dict[PipelineRole, ScriptedRole]]]:
    """Compile a graph and hand back the roles driving it."""

    def factory(
        config: PlatformConfig,
        executor: StubExecutor | None = None,
        **role_kwargs: Any,
    ) -> tuple[Any, dict[PipelineRole, ScriptedRole]]:
        roles = make_roles(**role_kwargs)
        return build_graph(roles, config, executor or StubExecutor()), roles

    return factory


@pytest.fixture
def thread() -> dict[str, Any]:
    """A fresh checkpointer thread per test, so runs cannot interfere."""
    return {"configurable": {"thread_id": uuid.uuid4().hex}}

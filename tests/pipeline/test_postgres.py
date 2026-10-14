"""Tests for the Postgres checkpointer wrapper.

Two layers, because what can be tested without a database and what the module
exists for are different things:

- Unit tests swap ``PostgresSaver`` for a fake and pin down *our* wiring: the
  allowlisted serializer lands on the saver, ``setup()`` runs before the saver
  is handed out, and the connection's lifetime is honoured.
- The restart test runs against a real Postgres (``SDLC_TEST_DATABASE_URL``)
  and proves the property the module exists for: an interrupt at a human gate
  survives the saver, the graph and the app all being thrown away. It skips
  when no database is offered - a skipped test is honest; a green one against
  a fake would not be.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from sdlc_agent.config import PlatformConfig
from sdlc_agent.domain import PipelineRole
from sdlc_agent.pipeline import StubExecutor, build_graph, initial_state
from sdlc_agent.pipeline import postgres as postgres_module
from sdlc_agent.pipeline.postgres import postgres_checkpointer
from sdlc_agent.pipeline.state import AuditEvent

from .helpers import approve, gate_name, make_requirement

TEST_DATABASE_URL = os.environ.get("SDLC_TEST_DATABASE_URL", "")


class FakeSaver:
    """Records what the wrapper does to it; stands in for ``PostgresSaver``."""

    def __init__(self) -> None:
        self.serde: Any = None
        self.setup_calls = 0

    def setup(self) -> None:
        self.setup_calls += 1


def install_fake_saver(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the module's ``PostgresSaver`` and return its event log."""
    events: dict[str, Any] = {"order": [], "conn_string": None}

    @contextmanager
    def fake_from_conn_string(cls: type, conn_string: str) -> Iterator[FakeSaver]:
        events["conn_string"] = conn_string
        events["order"].append("open")
        saver = FakeSaver()
        events["saver"] = saver
        try:
            yield saver
        finally:
            events["order"].append("close")

    fake_class = type(
        "FakePostgresSaver",
        (),
        {"from_conn_string": classmethod(fake_from_conn_string)},
    )
    monkeypatch.setattr(postgres_module, "PostgresSaver", fake_class)
    return events


class TestWrapperWiring:
    def test_yields_a_saver_with_the_allowlisted_serializer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A default serializer would silently re-widen the allowlist."""
        events = install_fake_saver(monkeypatch)

        with postgres_checkpointer("postgres://localhost/test"):
            serde = events["saver"].serde
            assert serde is not None
            # Functional check, not identity: the attached serializer round-trips
            # a registered type through the typed protocol the saver uses.
            dump = serde.dumps_typed(AuditEvent(stage="qa", event="verified", detail="2 passed"))
            restored = serde.loads_typed(dump)
            assert isinstance(restored, AuditEvent)
            assert restored.event == "verified"

    def test_setup_runs_before_the_saver_is_handed_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wrapper, not the caller, owns setup - callers cannot forget it."""
        events = install_fake_saver(monkeypatch)

        with postgres_checkpointer("postgres://localhost/test"):
            assert events["saver"].setup_calls == 1
            assert events["order"] == ["open"]

    def test_connection_closes_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Leaking connections would exhaust the pool one restart at a time."""
        events = install_fake_saver(monkeypatch)

        with postgres_checkpointer("postgres://localhost/test"):
            assert "close" not in events["order"]

        assert events["order"] == ["open", "close"]

    def test_connection_string_is_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The wrapper keeps no default URL; what the caller gives is what opens."""
        events = install_fake_saver(monkeypatch)

        with postgres_checkpointer("postgresql://db:5432/pipelines"):
            pass

        assert events["conn_string"] == "postgresql://db:5432/pipelines"


@pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set SDLC_TEST_DATABASE_URL to a Postgres the test may write to",
)
class TestGateSurvivesRestart:
    def test_an_interrupted_gate_resumes_on_a_new_connection(
        self, config: PlatformConfig, make_roles: Any
    ) -> None:
        """The property the module exists for: the gate outlives the process.

        The run stops at the first gate, then the saver, the graph and the app
        are all discarded - the closest a test can get to a process restart.
        A second checkpointer on a fresh connection must resume at the gate,
        and the PM must not run twice for it (HARNESS_SPEC.md section 2.4).
        """
        roles = make_roles()
        thread = {"configurable": {"thread_id": uuid.uuid4().hex}}

        with postgres_checkpointer(TEST_DATABASE_URL) as checkpointer:
            app = build_graph(roles, config, StubExecutor(), checkpointer=checkpointer)
            result = app.invoke(initial_state(make_requirement()), thread)
        assert gate_name(result) == "prd_review"
        assert roles[PipelineRole.PRODUCT_MANAGER].call_count == 1

        with postgres_checkpointer(TEST_DATABASE_URL) as checkpointer:
            rebuilt = build_graph(roles, config, StubExecutor(), checkpointer=checkpointer)
            resumed = rebuilt.invoke(approve(), thread)

        assert gate_name(resumed) == "design_review"
        assert roles[PipelineRole.PRODUCT_MANAGER].call_count == 1

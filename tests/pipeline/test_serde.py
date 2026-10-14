"""Tests for the checkpoint serializer's msgpack allowlist.

Every type the pipeline state can hold must be registered (see
``sdlc_agent.pipeline.serde``). An unregistered type still serializes but
LangGraph refuses to rebuild it on resume - which would take the human gates
down with it, since a gate only survives a restart because of its checkpoint.
"""

from __future__ import annotations

import uuid

from sdlc_agent.domain import CodeChange, FileEdit
from sdlc_agent.pipeline.serde import make_serializer


def test_a_code_change_with_edits_survives_a_checkpoint_round_trip() -> None:
    """FileEdit joined the state as a nested value inside CodeChange; if it
    were missing from the allowlist, a resumed run would fail to rebuild the
    artifact the merge gate is about to show."""
    change = CodeChange.model_validate(
        {
            "task_id": uuid.uuid4(),
            "branch": "sdlc-agent/abc123",
            "commit_sha": "a1b2c3d4e5f6",
            "edits": (FileEdit.model_validate({"path": "a.py", "content": "x = 1"}),),
            "summary": "Added the writer.",
        }
    )

    serde = make_serializer()
    restored = serde.loads_typed(serde.dumps_typed(change))

    assert isinstance(restored, CodeChange)
    assert restored.edits == change.edits

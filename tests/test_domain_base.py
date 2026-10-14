"""Tests for the shared artifact base classes and validation helper."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from sdlc_agent.domain.base import (
    Artifact,
    ArtifactKind,
    ReviewDecision,
    Verdict,
    reject_blank,
)


class TestRejectBlank:
    def test_returns_stripped_value(self) -> None:
        assert reject_blank("  ship it  ") == "ship it"

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="non-whitespace"):
            reject_blank("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="non-whitespace"):
            reject_blank("   \t\n ")


class TestArtifact:
    def test_assigns_identity_and_timestamp(self) -> None:
        artifact = Artifact(kind=ArtifactKind.PRD)

        assert isinstance(artifact.artifact_id, uuid.UUID)
        assert artifact.kind is ArtifactKind.PRD
        assert artifact.schema_version == 1
        assert artifact.created_at.tzinfo is not None

    def test_created_at_is_utc_and_recent(self) -> None:
        before = datetime.now(UTC)
        artifact = Artifact(kind=ArtifactKind.DESIGN)
        after = datetime.now(UTC)

        assert before <= artifact.created_at <= after

    def test_each_instance_gets_a_distinct_id(self) -> None:
        first = Artifact(kind=ArtifactKind.REQUIREMENT)
        second = Artifact(kind=ArtifactKind.REQUIREMENT)

        assert first.artifact_id != second.artifact_id

    def test_json_round_trip_preserves_fields(self) -> None:
        original = Artifact(kind=ArtifactKind.TEST_RUN)

        restored = Artifact.from_json(original.to_json())

        assert restored.artifact_id == original.artifact_id
        assert restored.kind is original.kind
        assert restored.created_at == original.created_at
        assert restored.schema_version == original.schema_version

    def test_is_frozen(self) -> None:
        """Artifacts are immutable so a handoff cannot be altered in flight."""
        artifact = Artifact(kind=ArtifactKind.PRD)

        with pytest.raises(ValidationError):
            artifact.kind = ArtifactKind.DESIGN  # type: ignore[misc]

    def test_rejects_unknown_fields(self) -> None:
        """extra="forbid" catches typos in agent-produced payloads early."""
        with pytest.raises(ValidationError, match="surprise"):
            Artifact.model_validate({"kind": "prd", "surprise": 1})

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError):
            Artifact.model_validate({"kind": "not_a_kind"})


class TestArtifactKind:
    def test_covers_the_full_pipeline(self) -> None:
        """Every stage of PM -> architect -> developer -> QA has an artifact."""
        assert [member.value for member in ArtifactKind] == [
            "requirement",
            "prd",
            "design",
            "task_breakdown",
            "code_change",
            "test_plan",
            "test_run",
            "quality_gate",
        ]

    def test_values_are_unique(self) -> None:
        values = [member.value for member in ArtifactKind]
        assert len(values) == len(set(values))


class TestReviewDecision:
    def test_records_a_gate_decision(self) -> None:
        artifact_id = uuid.uuid4()

        decision = ReviewDecision(
            artifact_id=artifact_id,
            reviewer="alice",
            verdict=Verdict.APPROVED,
            comment="looks good",
        )

        assert decision.artifact_id == artifact_id
        assert decision.reviewer == "alice"
        assert decision.verdict is Verdict.APPROVED
        assert decision.comment == "looks good"
        assert decision.decided_at.tzinfo is not None

    def test_comment_defaults_to_empty(self) -> None:
        decision = ReviewDecision(
            artifact_id=uuid.uuid4(),
            reviewer="bob",
            verdict=Verdict.NEEDS_HUMAN,
        )

        assert decision.comment == ""

    def test_rejects_unknown_verdict(self) -> None:
        with pytest.raises(ValidationError):
            ReviewDecision(
                artifact_id=uuid.uuid4(),
                reviewer="bob",
                verdict="maybe",  # type: ignore[arg-type]
            )

    def test_is_frozen(self) -> None:
        decision = ReviewDecision(
            artifact_id=uuid.uuid4(),
            reviewer="bob",
            verdict=Verdict.REJECTED,
        )

        with pytest.raises(ValidationError):
            decision.verdict = Verdict.APPROVED  # type: ignore[misc]

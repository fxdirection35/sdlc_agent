"""Tests for the domain package surface: type registry and kind-based loading."""

from __future__ import annotations

import json
import uuid

import pytest
from pydantic import ValidationError

from sdlc_agent.domain import (
    ARTIFACT_TYPES,
    Artifact,
    ArtifactKind,
    CodeChange,
    Design,
    ProductRequirements,
    QualityGate,
    Requirement,
    Task,
    TestPlan,
    TestRun,
    TestStatus,
    load_artifact,
)
from sdlc_agent.domain.product import AcceptanceCriterion


class TestArtifactTypeRegistry:
    def test_every_kind_is_registered(self) -> None:
        """A kind with no registered type would be unloadable from disk."""
        assert set(ARTIFACT_TYPES) == set(ArtifactKind)

    def test_registered_types_declare_matching_kinds(self) -> None:
        """Guards against a copy-paste slip mapping the wrong class to a kind."""
        for kind, artifact_type in ARTIFACT_TYPES.items():
            instance_kind = artifact_type.model_fields["kind"].default
            assert instance_kind is kind, (
                f"{artifact_type.__name__} declares {instance_kind}, not {kind}"
            )


class TestLoadArtifact:
    def test_loads_each_concrete_type(self) -> None:
        requirement = Requirement.model_validate(
            {
                "title": "Add CSV export",
                "description": "Users need CSV export.",
                "requester": "alice",
                "source_repo": "https://github.com/example/repo.git",
            }
        )

        loaded = load_artifact(requirement.to_json())

        assert isinstance(loaded, Requirement)
        assert loaded.artifact_id == requirement.artifact_id
        assert loaded.title == "Add CSV export"

    def test_loads_nested_artifact(self) -> None:
        prd = ProductRequirements.model_validate(
            {
                "requirement_id": uuid.uuid4(),
                "problem_statement": "Data cannot leave the app.",
                "goals": ("CSV export for any filtered report",),
                "acceptance_criteria": (
                    AcceptanceCriterion.model_validate(
                        {"statement": "CSV has a header row", "verification": "pytest tests/x.py"}
                    ),
                ),
            }
        )

        loaded = load_artifact(prd.to_json())

        assert isinstance(loaded, ProductRequirements)
        assert loaded.acceptance_criteria[0].statement == "CSV has a header row"

    def test_distinguishes_sibling_types(self) -> None:
        """Design and Task are both engineering artifacts; kind must separate them."""
        design = Design.model_validate(
            {"prd_id": uuid.uuid4(), "summary": "Stream the CSV.", "components": ("export.py",)}
        )
        task = Task.model_validate({"design_id": uuid.uuid4(), "title": "Write the writer"})

        assert isinstance(load_artifact(design.to_json()), Design)
        assert isinstance(load_artifact(task.to_json()), Task)

    def test_loads_qa_artifacts(self) -> None:
        plan = TestPlan.model_validate(
            {
                "prd_id": uuid.uuid4(),
                "code_change_id": uuid.uuid4(),
                "items": (
                    {
                        "criterion": "CSV has a header row",
                        "description": "Assert header",
                        "test_path": "tests/test_export.py",
                    },
                ),
            }
        )
        change = CodeChange.model_validate(
            {
                "task_id": uuid.uuid4(),
                "branch": "feat/csv",
                "commit_sha": "abc1234",
                "summary": "Added the writer.",
            }
        )
        run = TestRun.model_validate(
            {
                "test_plan_id": uuid.uuid4(),
                "status": TestStatus.PASSED,
                "results": (
                    {
                        "test_path": "tests/test_export.py",
                        "test_name": "test_header",
                        "status": "passed",
                    },
                ),
                "command": "pytest -q",
                "exit_code": 0,
            }
        )

        assert isinstance(load_artifact(plan.to_json()), TestPlan)
        assert isinstance(load_artifact(change.to_json()), CodeChange)

        loaded_run = load_artifact(run.to_json())
        assert isinstance(loaded_run, TestRun)
        assert loaded_run.status is TestStatus.PASSED

        gate = QualityGate.from_evidence(run)
        loaded_gate = load_artifact(gate.to_json())
        assert isinstance(loaded_gate, QualityGate)
        assert loaded_gate.verdict is gate.verdict

    def test_rejects_payload_without_kind(self) -> None:
        with pytest.raises(ValueError, match="no 'kind' field"):
            load_artifact(json.dumps({"title": "no kind here"}))

    def test_rejects_malformed_json(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            load_artifact("{not json")

    def test_rejects_non_object_payload(self) -> None:
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_artifact(json.dumps(["kind", "requirement"]))

    def test_rejects_unknown_kind(self) -> None:
        """A kind outside the enum fails before dispatch, naming what it got."""
        with pytest.raises(ValueError, match="unknown artifact kind 'deployment'"):
            load_artifact(json.dumps({"kind": "deployment", "title": "not a real kind"}))

    def test_rejects_extra_fields(self) -> None:
        """Artifacts forbid unknown keys, so a hallucinated field is caught."""
        payload = Requirement.model_validate(
            {"title": "t", "description": "d", "requester": "r", "source_repo": "s"}
        ).model_dump(mode="json")
        payload["priority"] = "high"

        with pytest.raises(ValidationError, match="priority"):
            load_artifact(json.dumps(payload))

    def test_dispatches_on_kind_then_validates_fields(self) -> None:
        """Dispatch resolves the concrete type, which then enforces its schema."""
        bare = Artifact(kind=ArtifactKind.PRD).to_json()

        with pytest.raises(ValidationError, match="requirement_id"):
            load_artifact(bare)

"""Tests for the engineering-role artifacts: Design, Task and CodeChange."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from sdlc_agent.domain.engineering import CodeChange, Design, Task


def make_design(**overrides: object) -> Design:
    payload: dict[str, object] = {
        "prd_id": uuid.uuid4(),
        "summary": "Add a streaming CSV writer behind the existing report endpoint.",
        "components": ("reports/export.py", "reports/views.py"),
        "api_changes": ("GET /reports/{id}/export?format=csv",),
    }
    payload.update(overrides)
    return Design.model_validate(payload)


class TestDesign:
    def test_valid_design(self) -> None:
        design = make_design()

        assert design.kind.value == "design"
        assert len(design.components) == 2
        assert design.risks == ()
        assert design.alternatives_considered == ()

    def test_requires_at_least_one_component(self) -> None:
        """A design naming no component gives the developer nothing to build."""
        with pytest.raises(ValidationError):
            make_design(components=())

    def test_rejects_blank_component(self) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_design(components=("reports/export.py", "  "))

    def test_rejects_blank_summary(self) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_design(summary="   ")

    def test_records_alternatives(self) -> None:
        design = make_design(
            alternatives_considered=("Client-side export: rejected, row count unbounded",),
            risks=("Large exports may time out",),
        )

        assert design.alternatives_considered[0].startswith("Client-side")
        assert design.risks[0].startswith("Large exports")

    def test_json_round_trip(self) -> None:
        original = make_design()

        restored = Design.from_json(original.to_json())

        assert restored.prd_id == original.prd_id
        assert restored.components == original.components


def make_task(**overrides: object) -> Task:
    payload: dict[str, object] = {
        "design_id": uuid.uuid4(),
        "title": "Implement streaming CSV writer",
        "acceptance_criteria": ("Handles >10k rows without buffering all in memory",),
    }
    payload.update(overrides)
    return Task.model_validate(payload)


class TestTask:
    def test_valid_task(self) -> None:
        task = make_task()

        assert task.kind.value == "task_breakdown"
        assert task.depends_on == ()

    def test_rejects_blank_title(self) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_task(title="  ")

    def test_rejects_overlong_title(self) -> None:
        with pytest.raises(ValidationError):
            make_task(title="x" * 301)

    def test_strips_criteria_whitespace(self) -> None:
        task = make_task(acceptance_criteria=("  Padded criterion  ",))

        assert task.acceptance_criteria == ("Padded criterion",)

    def test_dependency_edges_are_uuids(self) -> None:
        first = uuid.uuid4()
        second = uuid.uuid4()

        task = make_task(depends_on=(first, second))

        assert task.depends_on == (first, second)

    def test_rejects_malformed_dependency(self) -> None:
        with pytest.raises(ValidationError):
            make_task(depends_on=("not-a-uuid",))

    def test_json_round_trip_preserves_dependencies(self) -> None:
        dependency = uuid.uuid4()
        original = make_task(depends_on=(dependency,))

        restored = Task.from_json(original.to_json())

        assert restored.depends_on == (dependency,)


def make_code_change(**overrides: object) -> CodeChange:
    payload: dict[str, object] = {
        "task_id": uuid.uuid4(),
        "branch": "feat/csv-export",
        "commit_sha": "a1b2c3d4e5f6",
        "files_changed": ("src/reports/export.py", "tests/test_export.py"),
        "summary": "Added streaming CSV writer with row-chunked output.",
    }
    payload.update(overrides)
    return CodeChange.model_validate(payload)


class TestCodeChange:
    def test_valid_code_change(self) -> None:
        change = make_code_change()

        assert change.kind.value == "code_change"
        assert change.branch == "feat/csv-export"
        assert len(change.files_changed) == 2

    @pytest.mark.parametrize("sha", ["a1b2c3", "x" * 41])
    def test_rejects_out_of_range_commit_sha(self, sha: str) -> None:
        """Short shas are allowed down to 7 chars; anything longer than a full sha is not."""
        with pytest.raises(ValidationError):
            make_code_change(commit_sha=sha)

    def test_accepts_full_length_sha(self) -> None:
        change = make_code_change(commit_sha="a" * 40)

        assert len(change.commit_sha) == 40

    def test_rejects_blank_branch(self) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_code_change(branch="   ")

    def test_rejects_blank_file_path(self) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_code_change(files_changed=("src/ok.py", ""))

    def test_files_changed_may_be_empty(self) -> None:
        """A no-op change is unusual but valid; the reviewer decides if it is wrong."""
        change = make_code_change(files_changed=())

        assert change.files_changed == ()

    def test_json_round_trip(self) -> None:
        original = make_code_change()

        restored = CodeChange.from_json(original.to_json())

        assert restored.branch == original.branch
        assert restored.commit_sha == original.commit_sha
        assert restored.files_changed == original.files_changed

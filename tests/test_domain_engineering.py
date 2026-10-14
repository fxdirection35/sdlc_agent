"""Tests for the engineering-role artifacts: Design, Task, FileEdit and CodeChange."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from sdlc_agent.domain.engineering import CodeChange, Design, FileEdit, Task


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


def make_file_edit(**overrides: object) -> FileEdit:
    payload: dict[str, object] = {
        "path": "src/reports/export.py",
        "content": "def export(rows):\n    ...\n",
        "op": "write",
    }
    payload.update(overrides)
    return FileEdit.model_validate(payload)


class TestFileEdit:
    def test_valid_write_edit(self) -> None:
        edit = make_file_edit()

        assert edit.op == "write"
        assert edit.path == "src/reports/export.py"

    def test_write_is_the_default_operation(self) -> None:
        edit = FileEdit.model_validate({"path": "README.md", "content": "hello"})

        assert edit.op == "write"

    def test_valid_delete_edit(self) -> None:
        edit = make_file_edit(op="delete", content="")

        assert edit.op == "delete"

    def test_delete_must_not_carry_content(self) -> None:
        with pytest.raises(ValidationError, match="delete edit must not carry content"):
            make_file_edit(op="delete")

    @pytest.mark.parametrize(
        "path",
        [
            "/etc/passwd",  # POSIX absolute
            "C:\\Windows\\system32",  # Windows absolute, backslashes
            "src\\reports\\export.py",  # backslashes even when relative
            "../outside.py",  # upward traversal
            "src/../../outside.py",  # traversal hidden mid-path
            "  ",
        ],
    )
    def test_rejects_paths_that_escape_the_workspace(self, path: str) -> None:
        """An edit that can point outside the workspace is a write-anywhere
        primitive wearing an edit's clothes."""
        with pytest.raises(ValidationError):
            make_file_edit(path=path)

    def test_json_round_trip(self) -> None:
        original = make_file_edit()

        restored = FileEdit.model_validate(original.model_dump(mode="json"))

        assert restored == original


def make_code_change(**overrides: object) -> CodeChange:
    payload: dict[str, object] = {
        "task_id": uuid.uuid4(),
        "branch": "feat/csv-export",
        "commit_sha": "a1b2c3d4e5f6",
        "files_changed": ("src/reports/export.py", "tests/test_export.py"),
        "edits": (make_file_edit(),),
        "summary": "Added streaming CSV writer with row-chunked output.",
        "pr_url": "https://github.com/example/reports/pull/7",
    }
    payload.update(overrides)
    return CodeChange.model_validate(payload)


class TestCodeChange:
    def test_valid_code_change(self) -> None:
        change = make_code_change()

        assert change.kind.value == "code_change"
        assert change.branch == "feat/csv-export"
        assert len(change.files_changed) == 2
        assert change.edits[0].path == "src/reports/export.py"

    def test_the_pre_apply_state_is_all_empty(self) -> None:
        """Before the apply step runs, branch/sha/pr_url are unknown and the
        edits are the whole claim - this is the shape the developer role
        actually emits."""
        change = CodeChange.model_validate(
            {"task_id": uuid.uuid4(), "summary": "Will add the writer."}
        )

        assert change.branch == ""
        assert change.commit_sha == ""
        assert change.pr_url == ""
        assert change.edits == ()

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
        assert restored.edits == original.edits
        assert restored.pr_url == original.pr_url

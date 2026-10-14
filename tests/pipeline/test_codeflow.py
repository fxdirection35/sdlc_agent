"""Tests for the code-flow primitives in pipeline.codeflow.

The tree summarizer feeds prompts, so its output must be bounded (a whole
repository would blow the context window) and deterministic (same tree in,
same bytes out, so prompt diffs stay reviewable). The checkout node tests
drive the node directly; graph-level wiring lives in test_graph.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from sdlc_agent.pipeline.codeflow import (
    SKIP_DIRS,
    CodeIntegrations,
    make_checkout_node,
    summarize_file_tree,
)
from sdlc_agent.pipeline.graph import initial_state
from sdlc_agent.pipeline.state import PipelineState, RunStatus

from .helpers import FakeRepoClient, make_requirement


def make_tree(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "deep").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "README.md").write_text("readme", encoding="utf-8")
    (repo / "src" / "app.py").write_text("x = 1", encoding="utf-8")
    (repo / "src" / "deep" / "leaf.py").write_text("y = 2", encoding="utf-8")
    return repo


class TestSummarizeFileTree:
    def test_renders_a_small_tree_exactly(self, tmp_path: Path) -> None:
        """Sorted, two-space-indented, directories trailing-slash - pinned
        byte for byte because this text lands inside prompts."""
        repo = make_tree(tmp_path)

        assert summarize_file_tree(repo) == "\n".join(
            [
                "README.md",
                "src/",
                "  app.py",
                "  deep/",
                "    leaf.py",
                "tests/",
            ]
        )

    def test_missing_root_renders_empty(self, tmp_path: Path) -> None:
        """No workspace is a legitimate state (evals, degraded runs): the
        caller gets "nothing to show", not an error."""
        assert summarize_file_tree(tmp_path / "absent") == ""

    def test_depth_limit_ellipsizes_non_empty_directories(self, tmp_path: Path) -> None:
        """A directory at the limit must not look empty when it is not."""
        repo = tmp_path / "repo"
        (repo / "src" / "a" / "b" / "c").mkdir(parents=True)
        (repo / "src" / "a" / "b" / "c" / "file.txt").write_text("x", encoding="utf-8")
        (repo / "src" / "empty").mkdir()

        text = summarize_file_tree(repo, max_depth=2)

        assert "b/ …" in text
        assert "file.txt" not in text
        assert "empty/" in text  # an empty dir stays visibly empty

    def test_skips_uninteresting_directories(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        for name in sorted(SKIP_DIRS):
            (repo / name).mkdir(parents=True)
            (repo / name / "junk.bin").write_text("junk", encoding="utf-8")
        (repo / "app.py").write_text("x = 1", encoding="utf-8")

        assert summarize_file_tree(repo) == "app.py"

    def test_entry_budget_counts_the_omission(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        for index in range(5):
            (repo / f"file{index}.txt").write_text("x", encoding="utf-8")

        text = summarize_file_tree(repo, max_entries=3)

        assert "… (2 more entries omitted)" in text
        assert "file4.txt" not in text

    def test_char_budget_truncates_with_a_marker(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a" / "b").mkdir(parents=True)
        (repo / "a" / "long-name.py").write_text("x", encoding="utf-8")
        (repo / "z.py").write_text("x", encoding="utf-8")

        text = summarize_file_tree(repo, max_chars=10)

        assert "… (truncated at 10 characters)" in text
        assert "z.py" not in text

    def test_output_respects_the_character_budget(self, tmp_path: Path) -> None:
        """The marker is the only thing allowed to push past the budget."""
        repo = tmp_path / "repo"
        repo.mkdir()
        for index in range(50):
            (repo / f"module_{index:02d}.py").write_text("x", encoding="utf-8")

        text = summarize_file_tree(repo, max_chars=200)

        assert len(text.split("… (truncated at 200 characters)")[0]) <= 200


def run_checkout(
    code: CodeIntegrations,
    source_repo: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke the checkout node directly with an optional config override."""
    requirement = make_requirement().model_copy(update={"source_repo": source_repo})
    node = make_checkout_node(code, "checkout")
    result: dict[str, Any] = node(cast(PipelineState, initial_state(requirement)), config or {})
    return result


class TestCheckoutNode:
    def test_clones_an_https_remote_into_a_per_run_workspace(self, tmp_path: Path) -> None:
        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, token="ghp_sentinel_x")

        result = run_checkout(
            code, "https://github.com/example/reports.git", {"configurable": {"thread_id": "t-9"}}
        )

        assert result["repo_root"] == str(tmp_path / "t-9")
        assert repo.clones == [
            ("https://github.com/example/reports.git", tmp_path / "t-9", "ghp_sentinel_x")
        ]

    def test_without_a_thread_it_falls_back_to_the_requirement_id(self, tmp_path: Path) -> None:
        """Direct node invocation (no compiled graph) still names the workspace
        deterministically - by the only durable id the run has."""
        requirement = make_requirement()
        code = CodeIntegrations(repo=FakeRepoClient(), workspaces_dir=tmp_path)
        node = make_checkout_node(code, "checkout")

        result: dict[str, Any] = node(cast(PipelineState, initial_state(requirement)), {})

        assert result["repo_root"] == str(tmp_path / str(requirement.artifact_id))

    def test_a_synthetic_source_skips_without_touching_the_client(self, tmp_path: Path) -> None:
        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, token="ghp_sentinel_x")

        result = run_checkout(code, "eval://golden/dev_implement_writer")

        assert result["repo_root"] == ""
        assert repo.clones == []
        events = result["audit"]
        assert events[0].event == "checkout_skipped"
        assert "not an http(s) remote" in events[0].detail

    def test_an_existing_workspace_is_reused_without_cloning(self, tmp_path: Path) -> None:
        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path)
        workspace = tmp_path / "t-9"
        workspace.mkdir()
        (workspace / ".git").mkdir()

        result = run_checkout(
            code, "https://github.com/example/reports.git", {"configurable": {"thread_id": "t-9"}}
        )

        assert result["repo_root"] == str(workspace)
        assert repo.clones == []

    def test_a_partial_clone_leftover_is_redone(self, tmp_path: Path) -> None:
        """A directory without .git is the corpse of a failed clone; it must
        not poison the retry with 'destination already exists'."""
        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path)
        workspace = tmp_path / "t-9"
        workspace.mkdir()
        (workspace / "stray.tmp").write_text("junk", encoding="utf-8")

        result = run_checkout(
            code, "https://github.com/example/reports.git", {"configurable": {"thread_id": "t-9"}}
        )

        assert result["repo_root"] == str(workspace)
        assert len(repo.clones) == 1
        assert (workspace / ".git").exists()

    def test_a_failed_clone_escalates_without_leaking_the_token(self, tmp_path: Path) -> None:
        from sdlc_agent.integrations import GitCommandError

        repo = FakeRepoClient(
            fail_with=GitCommandError(["git", "clone", "https://x"], 128, "fatal: no access")
        )
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, token="ghp_sentinel_x")

        result = run_checkout(code, "https://github.com/example/reports.git")

        assert result["status"] is RunStatus.ESCALATED
        events = result["audit"]
        assert events[0].event == "checkout_failed"
        assert "ghp_sentinel_x" not in events[0].detail

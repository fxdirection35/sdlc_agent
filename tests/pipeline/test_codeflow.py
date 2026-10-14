"""Tests for the repo-context primitives in pipeline.codeflow.

The tree summarizer feeds prompts, so its output must be bounded (a whole
repository would blow the context window) and deterministic (same tree in,
same bytes out, so prompt diffs stay reviewable).
"""

from __future__ import annotations

from pathlib import Path

from sdlc_agent.pipeline.codeflow import SKIP_DIRS, summarize_file_tree


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

"""Code-flow helpers: bridging the pipeline's artifacts to a real repository.

This module grows into the checkout/apply/deliver nodes. It currently holds
the repo-context primitives: the bounded tree renderer for prompts, and the
:class:`RepoClient` protocol the deterministic nodes will drive. Real
implementations (subprocess git, GitHub REST) live in
:mod:`sdlc_agent.integrations`; tests use fakes against this protocol, the
same seam :class:`~sdlc_agent.pipeline.verify.TestExecutor` established.

Everything here is deterministic on purpose (HARNESS_SPEC.md section 5.2):
same tree in, same text out, so prompt diffs stay reviewable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sdlc_agent.domain.engineering import FileEdit

#: Directories that are never interesting to a role reading a repository:
#: build output, dependency trees, caches. Skipping them keeps the summary
#: inside its budget in real repos.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".sandbox",
        ".venv",
        "__pycache__",
        "dist",
        "node_modules",
    }
)


def summarize_file_tree(
    root: Path,
    *,
    max_depth: int = 3,
    max_entries: int = 200,
    max_chars: int = 8000,
    skip: frozenset[str] = SKIP_DIRS,
) -> str:
    """Render a bounded file-tree summary of ``root`` for a role prompt.

    Args:
        root: The workspace directory. A missing or non-directory root yields
            an empty string - "no repo to show" rather than an error, because
            runs without a workspace (evals, degraded mode) are legitimate.
        max_depth: Levels shown below ``root``. Deeper directories render as
            ``name/ …`` when they are non-empty, so the tree never lies by
            implying a directory is empty.
        max_entries: Maximum lines emitted; further entries are counted and
            reported as ``… (N more entries omitted)``.
        max_chars: Maximum total characters; on overflow the output is cut and
            reported as ``… (truncated at N characters)``.
        skip: Directory names excluded from the walk entirely.

    Returns:
        The rendered tree, empty for an absent root or a tree whose visible
        entries are all skipped.
    """

    if not root.is_dir():
        return ""

    lines: list[str] = []
    chars = 0
    omitted_entries = 0
    truncated_chars = False

    def emit(depth: int, text: str) -> None:
        nonlocal chars, omitted_entries, truncated_chars
        if truncated_chars:
            omitted_entries += 1
            return
        line = "  " * depth + text
        if len(lines) >= max_entries:
            omitted_entries += 1
            return
        # +1 for the newline that will join this line to the rest; without it
        # the rendered text overshoots the budget by one char per line.
        if chars + len(line) + 1 > max_chars:
            truncated_chars = True
            return
        lines.append(line)
        chars += len(line) + 1

    def walk(directory: Path, depth: int) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            return
        for entry in entries:
            if entry.name in skip:
                continue
            try:
                is_dir = entry.is_dir()
            except OSError:
                continue
            if not is_dir:
                emit(depth, entry.name)
            elif depth >= max_depth:
                # At the depth limit a directory renders with an ellipsis iff
                # it has children, so an empty dir never looks truncated-deep.
                try:
                    has_children = bool(list(entry.iterdir()))
                except OSError:
                    has_children = False
                emit(depth, f"{entry.name}/ …" if has_children else f"{entry.name}/")
            else:
                emit(depth, f"{entry.name}/")
                walk(entry, depth + 1)

    walk(root, 0)

    if truncated_chars:
        # The char marker subsumes the entry count: everything past the cut is
        # omitted by the same cut, and two ellipsis lines are just noise.
        lines.append(f"… (truncated at {max_chars} characters)")
    elif omitted_entries:
        lines.append(f"… ({omitted_entries} more entries omitted)")
    return "\n".join(lines)


#: Marker tag the clone step leaves in the workspace, pointing at the commit
#: the run started from. The apply step diffs against it, so a revision loop
#: keeps reporting the full change the PR will carry, not just its own delta.
BASE_TAG = "sdlc-agent/base"


@dataclass(frozen=True, slots=True)
class CommitResult:
    """What one successful apply recorded in the workspace.

    Attributes:
        sha: Full HEAD sha after the commit.
        files_changed: Paths that differ between the clone base and HEAD -
            everything this run's applies touched, across revision loops.
    """

    sha: str
    files_changed: tuple[str, ...]


class RepoClient(Protocol):
    """The git operations the deterministic nodes need, as one seam.

    Implementations must keep credentials out of argv and out of on-disk git
    configuration; the subprocess implementation documents how (see
    :mod:`sdlc_agent.integrations.git_subprocess`).
    """

    def clone(self, url: str, destination: Path, *, token: str = "") -> None:
        """Clone ``url`` into ``destination`` (created), tagging the base."""
        ...

    def apply_and_commit(
        self,
        root: Path,
        edits: Sequence[FileEdit],
        branch: str,
        message: str,
        *,
        token: str = "",
    ) -> CommitResult:
        """Write ``edits`` into the workspace and commit them on ``branch``."""
        ...

    def push(self, root: Path, branch: str, *, token: str = "") -> None:
        """Push ``branch`` to the workspace's origin."""
        ...


__all__ = ["BASE_TAG", "SKIP_DIRS", "CommitResult", "RepoClient", "summarize_file_tree"]

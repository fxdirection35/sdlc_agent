"""Code-flow helpers: bridging the pipeline's artifacts to a real repository.

This module grows into the checkout/apply/deliver nodes; it starts with the
one pure function they all need - rendering a bounded view of the workspace
for prompts. Bounded matters twice over: a whole-repo listing would blow the
context window, and an unbounded walk could hang on a pathological tree.

Everything here is deterministic on purpose (HARNESS_SPEC.md section 5.2):
same tree in, same text out, so prompt diffs stay reviewable.
"""

from __future__ import annotations

from pathlib import Path

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


__all__ = ["SKIP_DIRS", "summarize_file_tree"]

"""Code-flow helpers: bridging the pipeline's artifacts to a real repository.

This module holds the deterministic code-flow nodes - checkout, apply and
deliver - plus the repo-context primitives they share: the bounded
tree renderer for prompts and the :class:`RepoClient` protocol the nodes
drive. Real implementations (subprocess git, GitHub REST) live in
:mod:`sdlc_agent.integrations`; tests use fakes against this protocol, the
same seam :class:`~sdlc_agent.pipeline.verify.TestExecutor` established.

Everything here is deterministic on purpose (HARNESS_SPEC.md section 5.2):
same tree in, same text out, so prompt diffs stay reviewable.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

# RunnableConfig is re-exported by langgraph's own docs from here; importing
# it from langgraph.config trips mypy's explicit-re-export check.
from langchain_core.runnables import RunnableConfig

from sdlc_agent.domain import ArtifactKind
from sdlc_agent.domain.engineering import CodeChange, FileEdit
from sdlc_agent.domain.qa import QualityGate
from sdlc_agent.pipeline.state import AuditEvent, PipelineState, RunStatus

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

    def head_sha(self, root: Path, *, token: str = "") -> str:
        """Current HEAD of the workspace, for the deliver guard's check that
        what would be pushed is what was verified."""
        ...


class PullRequestOpener(Protocol):
    """Opens (or finds) the pull request a delivered branch is reviewed in."""

    def ensure_open(self, *, repo: str, branch: str, title: str, body: str) -> str:
        """Return the PR's ``html_url``, opening one only if none is open.

        Must be idempotent per branch: a revision loop re-delivers the same
        branch and must find its own PR, not open a sibling.
        """
        ...

    def slug_for(self, remote: str) -> str:
        """``owner/name`` for a git remote this opener serves.

        Raises when the remote is not served by this opener (wrong host,
        wrong scheme): a remote the opener cannot serve simply gets no PR.
        """
        ...


@dataclass(frozen=True, slots=True)
class CodeIntegrations:
    """The real-code bundle the graph needs, injected as one seam.

    Presence is the feature switch: ``build_graph(code=None)`` builds the old
    artifact-only topology; a ``CodeIntegrations`` adds the checkout/apply/
    deliver nodes. The token lives in memory only - resolved once at startup,
    passed through to the client per call, never written to state, audit,
    checkpoints or disk.

    Attributes:
        repo: The git seam. Implementations keep credentials out of argv and
            out of on-disk git configuration.
        workspaces_dir: Parent of the per-run clone directories. One run's
            workspace is ``workspaces_dir / run_id``, so gates that sit for
            days find the same clone on resume.
        token: Remote credentials for https origins; empty for local or
            public-write setups.
        pulls: PR seam for the deliver node. ``None`` delivers by push only -
            the branch lands on the remote but no PR is opened or linked.
    """

    repo: RepoClient
    workspaces_dir: Path
    token: str = ""
    pulls: PullRequestOpener | None = None


def _run_id(config: RunnableConfig, state: PipelineState) -> str:
    """Identify the run for workspace naming: the LangGraph thread id.

    The thread id is what the checkpointer keys resumes on, so it is already
    the run's durable identity. Falls back to the requirement's id for direct
    node invocation outside a compiled graph (tests), which is unique per run
    but not resumable - exactly the situation the fallback fits.
    """
    configurable = (config or {}).get("configurable") or {}
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id else str(state["requirement"].artifact_id)


def _branch_for(config: RunnableConfig, state: PipelineState) -> str:
    """Deterministic branch name: one branch per run, stable across revision
    loops, so a pushed PR updates in place instead of spawning siblings."""
    return f"sdlc-agent/{_run_id(config, state)}"


def _audit(stage: str, event: str, detail: str = "") -> tuple[AuditEvent, ...]:
    return (AuditEvent(stage=stage, event=event, detail=detail),)


def make_checkout_node(code: CodeIntegrations, stage: str) -> Any:
    """Build the node that gives a run its workspace, or honestly skips.

    A run whose requirement names an http(s) remote exists to change code, so
    a failed clone escalates rather than degrading to artifact-only mode -
    that degradation would be a green run that cannot deliver. Everything
    else skips with an audit trail: synthetic eval sources, and local paths
    the auth layer deliberately does not scope credentials to.

    Re-execution is safe: an existing workspace with a ``.git`` is reused, a
    leftover directory without one (a clone that died halfway) is removed
    before cloning again.
    """

    def node(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        requirement = state["requirement"]
        scheme = urlsplit(requirement.source_repo).scheme
        if scheme not in ("http", "https"):
            named = scheme if scheme else "a local path"
            return {
                "repo_root": "",
                "audit": _audit(
                    stage, "checkout_skipped", f"source is not an http(s) remote: {named}"
                ),
            }

        workspace = code.workspaces_dir / _run_id(config, state)
        if (workspace / ".git").exists():
            return {
                "repo_root": str(workspace),
                "audit": _audit(stage, "checkout_reused", str(workspace)),
            }
        if workspace.exists():
            # Not a repository, so nothing of value: the corpse of a failed
            # clone. The directory belongs to this pipeline alone.
            shutil.rmtree(workspace, ignore_errors=True)
        try:
            code.repo.clone(requirement.source_repo, workspace, token=code.token)
        except (RuntimeError, OSError) as exc:
            # RuntimeError covers GitCommandError without making the pipeline
            # depend on the integrations layer; the message carries no token.
            return {
                "status": RunStatus.ESCALATED,
                "audit": _audit(stage, "checkout_failed", str(exc)),
            }
        return {
            "repo_root": str(workspace),
            "audit": _audit(stage, "checkout_cloned", f"{requirement.source_repo} -> {workspace}"),
        }

    return node


def make_apply_node(code: CodeIntegrations, stage: str) -> Any:
    """Build the patch-bridge node: developer claims -> repository commits.

    The developer's ``CodeChange`` carries typed :class:`FileEdit` claims; this
    node is the only place they become files. Without a workspace the node is a
    pass-through so eval-shaped runs keep their exact topology. With one, a
    code change that claims no edits escalates - today that would slide
    through as a fabricated branch name, and a green run that changed no code
    is the fakest green there is.

    The artifact is rebuilt via ``model_validate(dump | updates)`` so the
    ``artifact_id`` survives: ``TestPlan.code_change_id`` back-pointers must
    not break because the platform filled in what only it could know.
    """

    def node(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        repo_root = state.get("repo_root", "")
        if not repo_root:
            return {"audit": _audit(stage, "apply_skipped", "run has no workspace")}

        change = state.get("artifacts", {}).get(ArtifactKind.CODE_CHANGE)
        if not isinstance(change, CodeChange):
            return {
                "status": RunStatus.ESCALATED,
                "audit": _audit(stage, "apply_failed", "no code change to apply"),
            }
        if not change.edits:
            return {
                "status": RunStatus.ESCALATED,
                "audit": _audit(
                    stage,
                    "apply_failed",
                    "developer produced no file edits for a real repository",
                ),
            }

        branch = _branch_for(config, state)
        try:
            commit = code.repo.apply_and_commit(
                Path(repo_root),
                change.edits,
                branch,
                f"sdlc-agent: {state['requirement'].title}",
                token=code.token,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            return {
                "status": RunStatus.ESCALATED,
                "audit": _audit(stage, "apply_failed", str(exc)),
            }

        updated = CodeChange.model_validate(
            {
                **change.model_dump(),
                "branch": branch,
                "commit_sha": commit.sha,
                "files_changed": commit.files_changed,
            }
        )
        detail = f"{branch} @ {commit.sha[:12]}, {len(commit.files_changed)} file(s)"
        claimed = {edit.path for edit in change.edits}
        missing = claimed - set(commit.files_changed)
        if missing:
            # The commit landed; some claimed paths made no actual difference
            # (e.g. content identical to what is already there). Informative,
            # not fatal - the PR diff is the truth the reviewer will see.
            detail += f"; claimed but unchanged: {', '.join(sorted(missing))}"
        return {
            "artifacts": {ArtifactKind.CODE_CHANGE: updated},
            "audit": _audit(stage, "applied", detail),
        }

    return node


def make_deliver_node(code: CodeIntegrations, stage: str) -> Any:
    """Build the node that puts a verified change in front of a human.

    Runs after verify approves, before the merge gate: push the branch, open
    (or find) the PR, and put its URL on the artifacts the gate displays - so
    the human approves or rejects a *diff they can click*, not a description
    of one.

    Failure policy is asymmetric on purpose. A workspace whose HEAD no longer
    matches the verified ``CodeChange.commit_sha`` escalates: the volume was
    lost or the workspace was rebuilt, and pushing whatever is there now
    would publish code nobody verified. A push or PR call that merely fails
    (GitHub down, network blip) is audited and the run continues to the gate
    - a transient outage must not kill a change the suite already passed; the
    human gate remains the safety net, and delivery can be re-driven.
    """

    def node(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        repo_root = state.get("repo_root", "")
        if not repo_root:
            return {"audit": _audit(stage, "deliver_skipped", "run has no workspace")}

        change = state.get("artifacts", {}).get(ArtifactKind.CODE_CHANGE)
        if not isinstance(change, CodeChange):
            return {
                "status": RunStatus.ESCALATED,
                "audit": _audit(stage, "deliver_failed", "no code change to deliver"),
            }

        try:
            head = code.repo.head_sha(Path(repo_root), token=code.token)
        except (RuntimeError, OSError) as exc:
            return {
                "status": RunStatus.ESCALATED,
                "audit": _audit(stage, "deliver_diverged", f"could not read workspace HEAD: {exc}"),
            }
        if head != change.commit_sha:
            return {
                "status": RunStatus.ESCALATED,
                "audit": _audit(
                    stage,
                    "deliver_diverged",
                    f"workspace HEAD {head[:12]} != verified {change.commit_sha[:12]}; "
                    "refusing to push unverified code",
                ),
            }

        branch = _branch_for(config, state)
        # Deliberately no "PR already open" short-circuit. After a merge-review
        # rejection the loop re-applies, and by the time deliver runs again
        # head == commit_sha once more - while the open PR still points at the
        # *previous* sha. Skipping the push would leave the human approving a
        # stale diff. Both operations below are idempotent instead: re-pushing
        # an unchanged branch is a no-op, and ensure_open finds the existing
        # PR rather than opening a sibling.
        audit: list[AuditEvent] = []
        try:
            code.repo.push(Path(repo_root), branch, token=code.token)
            audit.append(
                AuditEvent(stage=stage, event="delivered", detail=f"pushed {branch} @ {head[:12]}")
            )
        except (RuntimeError, OSError) as exc:
            audit.append(AuditEvent(stage=stage, event="deliver_failed", detail=f"push: {exc}"))
            return {"audit": tuple(audit)}

        updates: dict[str, Any] = {}
        if code.pulls is not None:
            try:
                slug = code.pulls.slug_for(state["requirement"].source_repo)
                url = code.pulls.ensure_open(
                    repo=slug,
                    branch=branch,
                    title=f"sdlc-agent: {state['requirement'].title}",
                    body=(
                        f"{change.summary}\n\n"
                        f"Branch: {branch}\n"
                        f"Files changed: {len(change.files_changed)}\n\n"
                        "Opened by the SDLC agent pipeline; the merge gate reviews it."
                    ),
                )
            except (RuntimeError, ValueError) as exc:
                audit.append(AuditEvent(stage=stage, event="deliver_failed", detail=f"PR: {exc}"))
            else:
                audit.append(AuditEvent(stage=stage, event="delivered", detail=f"PR {url}"))
                updates = {
                    "artifacts": {
                        ArtifactKind.CODE_CHANGE: CodeChange.model_validate(
                            {**change.model_dump(), "pr_url": url}
                        ),
                    }
                }
                gate = state.get("artifacts", {}).get(ArtifactKind.QUALITY_GATE)
                if isinstance(gate, QualityGate):
                    updates["artifacts"][ArtifactKind.QUALITY_GATE] = gate.model_copy(
                        update={"pr_url": url}
                    )

        return {"audit": tuple(audit), **updates}

    return node


__all__ = [
    "BASE_TAG",
    "SKIP_DIRS",
    "CodeIntegrations",
    "CommitResult",
    "PullRequestOpener",
    "RepoClient",
    "make_apply_node",
    "make_checkout_node",
    "make_deliver_node",
    "summarize_file_tree",
]

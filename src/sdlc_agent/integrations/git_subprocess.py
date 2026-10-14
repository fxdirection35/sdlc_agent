"""Git over subprocess: the RepoClient implementation behind the code-flow nodes.

Credentials never travel where they could be observed or persisted:

- **Never in argv.** The process list is readable by other users on a host;
  auth rides in the environment using git's ``GIT_CONFIG_COUNT`` mechanism
  (git >= 2.31): ``http.<origin>/.extraheader`` with a basic-auth header.
- **Never on disk.** The remote stays a clean https URL - no token ever
  appears in ``.git/config``, so a workspace or its config can be inspected,
  archived or shipped without scrubbing.

Local-path origins (tests, local development) carry no scheme and get no auth
environment at all.
"""

from __future__ import annotations

import base64
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from sdlc_agent.domain.engineering import FileEdit
from sdlc_agent.pipeline.codeflow import BASE_TAG, CommitResult


class GitCommandError(RuntimeError):
    """A git invocation failed.

    The message carries the command (never the environment, which is where
    auth lives) and the first line of stderr, truncated - stderr can be
    arbitrarily long and has been known to echo configuration values.
    """

    def __init__(self, argv: Sequence[str], returncode: int, stderr: str) -> None:
        command = " ".join(argv)
        first_line = stderr.strip().splitlines()[0] if stderr.strip() else "(no stderr)"
        if len(first_line) > 200:
            first_line = first_line[:200] + "…"
        status = "failed" if returncode else "could not run"
        super().__init__(f"git {status} ({command}): {first_line}")
        self.argv = tuple(argv)
        self.returncode = returncode


def _default_runner(
    argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), cwd=cwd, env=dict(env), timeout=timeout, capture_output=True, text=True
    )


class _Runner(Protocol):
    """The subprocess seam: fixed binary, list argv, no shell, per-call env."""

    def __call__(
        self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]: ...


def _auth_env(url: str, token: str) -> dict[str, str]:
    """Environment entries that authenticate https remotes without touching disk.

    Scoped to the origin's scheme+host so a malicious ``url`` in a request
    cannot redirect the header at an attacker host... and vice versa: the
    header is only attached to the host the run actually named.
    """
    if not token:
        return {}
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return {}
    scope = f"http.{parts.scheme}://{parts.netloc}/.extraheader"
    secret = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": scope,
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {secret}",
    }


@dataclass(frozen=True, slots=True)
class SubprocessGit:
    """RepoClient driven by the ``git`` binary.

    Attributes:
        git_bin: The git executable; overridable for tests and odd images.
        committer_name/email: Identity for commits the pipeline itself makes.
            A workspace has no user configuration of its own, and git refuses
            to commit without an identity.
        clone_timeout/push_timeout/op_timeout: Per-command ceilings; a hung
            network git must not hang a graph node with it.
        runner: The subprocess seam. Tests inject a recorder or a stub; the
            default runs real subprocesses.
    """

    git_bin: str = "git"
    committer_name: str = "sdlc-agent"
    committer_email: str = "agent@sdlc.invalid"
    clone_timeout: float = 120.0
    push_timeout: float = 60.0
    op_timeout: float = 30.0
    runner: _Runner = field(default=_default_runner, repr=False)

    def _git(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        token: str = "",
        url: str = "",
        timeout: float | None = None,
    ) -> str:
        argv = [self.git_bin, "-c", "core.autocrlf=false", *args]
        env = {**os.environ, **_auth_env(url, token)}
        ceiling = timeout if timeout is not None else self.op_timeout
        completed = self.runner(argv, cwd=cwd, env=env, timeout=ceiling)
        if completed.returncode != 0:
            raise GitCommandError(argv, completed.returncode, completed.stderr)
        return completed.stdout

    def clone(self, url: str, destination: Path, *, token: str = "") -> None:
        """Shallow-clone ``url`` and tag the starting commit as the base.

        ``--depth 1`` keeps large repositories cheap; pushing from a shallow
        clone works against GitHub. The base tag is what the apply step diffs
        against, so revision loops report the whole change, not their own delta.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._git(
            ["clone", "--depth", "1", url, str(destination)],
            cwd=destination.parent,
            token=token,
            url=url,
            timeout=self.clone_timeout,
        )
        self._git(["tag", BASE_TAG], cwd=destination)

    def apply_and_commit(
        self,
        root: Path,
        edits: Sequence[FileEdit],
        branch: str,
        message: str,
        *,
        token: str = "",
    ) -> CommitResult:
        """Write ``edits`` and commit them on ``branch`` (created or reused).

        ``checkout -B`` re-anchors the branch at HEAD: on a revision loop the
        workspace already sits on this branch, so the new commit appends to it
        and the pushed PR updates in place.

        Raises:
            ValueError: An edit names a file that does not exist (delete) or a
                path escaping the workspace (defence in depth - the FileEdit
                model already rejects those).
            GitCommandError: Any git failure, including a commit with nothing
                to commit - identical edits mean the revision changed nothing.
        """
        self._git(["checkout", "-B", branch], cwd=root, token=token)
        resolved_root = root.resolve()
        for edit in edits:
            target = (root / edit.path).resolve()
            if not target.is_relative_to(resolved_root):
                raise ValueError(f"edit escapes the workspace: {edit.path}")
            if edit.op == "delete":
                if not target.exists():
                    raise ValueError(f"delete names a missing file: {edit.path}")
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # newline="\n" so file content is exactly what the role wrote,
                # regardless of the host's line-ending convention.
                target.write_text(edit.content, encoding="utf-8", newline="\n")
        self._git(
            [
                "-c",
                f"user.name={self.committer_name}",
                "-c",
                f"user.email={self.committer_email}",
                "add",
                "-A",
            ],
            cwd=root,
            token=token,
        )
        self._git(
            [
                "-c",
                f"user.name={self.committer_name}",
                "-c",
                f"user.email={self.committer_email}",
                "commit",
                "-m",
                message,
            ],
            cwd=root,
            token=token,
        )
        sha = self._git(["rev-parse", "HEAD"], cwd=root, token=token).strip()
        changed = self._git(["diff", "--name-only", f"{BASE_TAG}..HEAD"], cwd=root, token=token)
        files = tuple(line.strip() for line in changed.splitlines() if line.strip())
        return CommitResult(sha=sha, files_changed=files)

    def push(self, root: Path, branch: str, *, token: str = "") -> None:
        """Push ``branch`` to origin; idempotent for an unchanged branch."""
        url = self._git(["remote", "get-url", "origin"], cwd=root, token=token).strip()
        self._git(
            ["push", "origin", branch],
            cwd=root,
            token=token,
            url=url,
            timeout=self.push_timeout,
        )


__all__ = ["GitCommandError", "SubprocessGit"]

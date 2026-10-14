"""Tests for the subprocess git client.

Two layers, mirroring how the graph will use it: a fake runner pins the exact
command shapes and - more importantly - where credentials are allowed to
appear (env only, never argv, never on disk); real-git tests then prove the
commands actually work against a local origin, including the revision-loop
behaviour the apply node depends on.

The real-git layer skips without a git binary, the same honest-skip the
docker and postgres tests use.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from sdlc_agent.domain.engineering import FileEdit
from sdlc_agent.integrations import GitCommandError, SubprocessGit

TOKEN = "ghp_sentinel_token_0123456789"
HTTPS_URL = "https://github.com/example/reports.git"


@dataclass
class RecordedCall:
    argv: list[str]
    env: dict[str, str]
    cwd: Path
    timeout: float


class RecordingRunner:
    """Stands in for subprocess.run: records every call, replays canned results."""

    def __init__(self, *results: subprocess.CompletedProcess[str]) -> None:
        self.calls: list[RecordedCall] = []
        self._results = list(results)

    def __call__(
        self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(RecordedCall(list(argv), dict(env), cwd, timeout))
        if self._results:
            return self._results.pop(0)
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")


class TestCommandShapes:
    def test_clone_is_shallow_and_tags_the_base(self, tmp_path: Path) -> None:
        runner = RecordingRunner()
        git = SubprocessGit(runner=runner)

        git.clone(HTTPS_URL, tmp_path / "work", token=TOKEN)

        assert runner.calls[0].argv == [
            "git",
            "-c",
            "core.autocrlf=false",
            "clone",
            "--depth",
            "1",
            HTTPS_URL,
            str(tmp_path / "work"),
        ]
        assert runner.calls[1].argv == [
            "git",
            "-c",
            "core.autocrlf=false",
            "tag",
            "sdlc-agent/base",
        ]

    def test_auth_rides_in_env_not_argv(self, tmp_path: Path) -> None:
        """The process list is world-readable; the token must never be in it."""
        runner = RecordingRunner()
        git = SubprocessGit(runner=runner)

        git.clone(HTTPS_URL, tmp_path / "work", token=TOKEN)

        env = runner.calls[0].env
        expected_secret = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
        assert env["GIT_CONFIG_COUNT"] == "1"
        assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
        assert env["GIT_CONFIG_VALUE_0"] == f"Authorization: Basic {expected_secret}"
        assert TOKEN not in " ".join(runner.calls[0].argv)

    def test_no_token_means_no_auth_environment(self, tmp_path: Path) -> None:
        runner = RecordingRunner()
        git = SubprocessGit(runner=runner)

        git.clone(HTTPS_URL, tmp_path / "work")

        assert not any(key.startswith("GIT_CONFIG_") for key in runner.calls[0].env)

    def test_local_path_origins_get_no_auth_environment(self, tmp_path: Path) -> None:
        """Local origins are not http(s), so no header can be scoped to them -
        and none is needed. The header never leaks onto an unrelated host."""
        runner = RecordingRunner()
        git = SubprocessGit(runner=runner)

        git.clone(str(tmp_path / "origin"), tmp_path / "work", token=TOKEN)

        assert not any(key.startswith("GIT_CONFIG_") for key in runner.calls[0].env)

    def test_failure_raises_with_command_and_stderr_never_the_token(self, tmp_path: Path) -> None:
        runner = RecordingRunner(
            subprocess.CompletedProcess(["git"], 128, stdout="", stderr="fatal: boom")
        )
        git = SubprocessGit(runner=runner)

        with pytest.raises(GitCommandError, match="fatal: boom") as exc_info:
            git.clone(HTTPS_URL, tmp_path / "work", token=TOKEN)

        assert TOKEN not in str(exc_info.value)


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def make_origin(tmp_path: Path) -> Path:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-b", "main")
    (origin / "README.md").write_text("hello", encoding="utf-8")
    _git(origin, "add", "-A")
    _git(origin, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-m", "init")
    return origin


def _local_transport_works() -> bool:
    """Probe whether this host can run git's local transport from native Python.

    Every local-path clone/push routes through a transport helper that git
    launches via sh.exe - and on hosts where a resident MSYS session already
    owns the shared section with a restrictive DACL (measured on the
    agent-driven Windows host: NtCreateDirectoryObject 0xC0000022), any
    sh.exe spawned outside that session dies before doing anything. The
    fake-runner layer still pins the command shapes; this layer skips
    honestly, exactly as it does without a git binary. CI and normal
    terminals have no such conflict and run everything.
    """
    with tempfile.TemporaryDirectory() as tmp:
        origin = Path(tmp) / "origin"
        origin.mkdir()
        probe = subprocess.run(["git", "init", "-q", str(origin)], capture_output=True, text=True)
        if probe.returncode != 0:
            return False
        probe = subprocess.run(
            ["git", "clone", str(origin), str(Path(tmp) / "work")],
            capture_output=True,
            text=True,
        )
        return probe.returncode == 0


_LOCAL_TRANSPORT = _local_transport_works()

#: git missing, or the host cannot run msys helpers from native Python.
_REAL_GIT_USABLE = shutil.which("git") is not None and _LOCAL_TRANSPORT


@pytest.mark.skipif(not _REAL_GIT_USABLE, reason="local git transport unavailable on this host")
class TestRealGit:
    def test_clone_apply_push_round_trip(self, tmp_path: Path) -> None:
        origin = make_origin(tmp_path)
        git = SubprocessGit()
        work = tmp_path / "work"

        git.clone(str(origin), work)

        assert (work / "README.md").read_text(encoding="utf-8") == "hello"

        result = git.apply_and_commit(
            work,
            [
                FileEdit.model_validate({"path": "src/new.py", "content": "x = 1\n"}),
                FileEdit.model_validate({"path": "docs/note.md", "content": "note"}),
            ],
            "sdlc-agent/run1",
            "apply the design",
        )

        assert len(result.sha) == 40
        assert set(result.files_changed) == {"src/new.py", "docs/note.md"}
        assert (work / "src/new.py").read_text(encoding="utf-8") == "x = 1\n"

        git.push(work, "sdlc-agent/run1")

        assert _git(origin, "rev-parse", "--verify", "sdlc-agent/run1")

    def test_reapply_appends_and_reports_the_full_change(self, tmp_path: Path) -> None:
        """The revision loop re-enters apply on the same branch: the new
        commit must append, and files_changed must stay anchored to the clone
        base so the PR describes the whole change, not the last delta."""
        origin = make_origin(tmp_path)
        git = SubprocessGit()
        work = tmp_path / "work"
        git.clone(str(origin), work)

        first = git.apply_and_commit(
            work,
            [FileEdit.model_validate({"path": "src/new.py", "content": "x = 1\n"})],
            "sdlc-agent/run1",
            "apply the design",
        )
        second = git.apply_and_commit(
            work,
            [
                FileEdit.model_validate({"path": "src/new.py", "content": "x = 2\n"}),
                FileEdit.model_validate({"path": "tests/t.py", "content": "def test_x(): ..."}),
            ],
            "sdlc-agent/run1",
            "apply the revision",
        )

        assert second.sha != first.sha
        # One branch, two commits - a pushed PR would update in place.
        assert _git(work, "rev-list", "--count", "sdlc-agent/base..sdlc-agent/run1").strip() == "2"
        assert set(second.files_changed) == {"src/new.py", "tests/t.py"} | set(first.files_changed)

        git.push(work, "sdlc-agent/run1")

    def test_delete_removes_the_file(self, tmp_path: Path) -> None:
        origin = make_origin(tmp_path)
        git = SubprocessGit()
        work = tmp_path / "work"
        git.clone(str(origin), work)

        result = git.apply_and_commit(
            work,
            [FileEdit.model_validate({"path": "README.md", "op": "delete"})],
            "sdlc-agent/run1",
            "remove the readme",
        )

        assert not (work / "README.md").exists()
        assert result.files_changed == ("README.md",)

    def test_delete_of_a_missing_file_is_rejected(self, tmp_path: Path) -> None:
        origin = make_origin(tmp_path)
        git = SubprocessGit()
        work = tmp_path / "work"
        git.clone(str(origin), work)

        with pytest.raises(ValueError, match="missing file"):
            git.apply_and_commit(
                work,
                [FileEdit.model_validate({"path": "nope.py", "op": "delete"})],
                "sdlc-agent/run1",
                "delete",
            )

    def test_a_commit_with_nothing_to_commit_fails(self, tmp_path: Path) -> None:
        """No edits means the revision changed nothing; git refuses, and so
        must the caller - an empty commit is a fake change."""
        origin = make_origin(tmp_path)
        git = SubprocessGit()
        work = tmp_path / "work"
        git.clone(str(origin), work)

        with pytest.raises(GitCommandError, match=r"nothing to commit|no changes"):
            git.apply_and_commit(work, [], "sdlc-agent/run1", "empty")

    def test_the_token_never_lands_on_disk(self, tmp_path: Path) -> None:
        """Auth rides in per-process env; scan the workspace's git metadata to
        prove no copy of it persisted anywhere after a full flow."""
        origin = make_origin(tmp_path)
        git = SubprocessGit()
        work = tmp_path / "work"

        git.clone(str(origin), work, token=TOKEN)
        git.apply_and_commit(
            work,
            [FileEdit.model_validate({"path": "src/new.py", "content": "x = 1\n"})],
            "sdlc-agent/run1",
            "apply",
            token=TOKEN,
        )
        git.push(work, "sdlc-agent/run1", token=TOKEN)

        for path in (work / ".git").rglob("*"):
            if path.is_file():
                assert TOKEN not in path.read_text(encoding="utf-8", errors="replace"), path

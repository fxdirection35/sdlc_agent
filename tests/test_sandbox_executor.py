"""Tests for the sandbox executor.

Two kinds, deliberately separated:

- Real-execution tests drive an actual pytest subprocess through a
  LocalWorkspace over a generated project in tmp_path. They are the point of
  this module: the executor's honesty rules (no evidence -> not_run) are only
  real if a real pytest produced the evidence. tmp_path sits inside this
  repository (pinned basetemp), which also makes these runs the guard against
  host pytest config leaking into the workspace - before the command pinned
  its config, a run crashed on this repo's own ``filterwarnings = error``
  before collecting anything.
- Parser tests feed crafted junit bytes to a pure function, for shapes this
  pipeline's own runs may never produce (skips, message-only failures).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import pytest
from openhands.sdk.workspace.base import BaseWorkspace
from openhands.sdk.workspace.models import CommandResult

from sdlc_agent.domain.product import Requirement
from sdlc_agent.domain.qa import TestPlan, TestRun, TestStatus
from sdlc_agent.evals import AgentRun, load_all_tasks, run_task
from sdlc_agent.pipeline.roles import RoleContext
from sdlc_agent.sandbox.executor import (
    SandboxExecutor,
    _aggregate,
    _parse_junit,
)

#: The SDK's shell utility leaks one file handle per executed command; the
#: unraisable warning surfaces during garbage collection, which can land in
#: any test of this module (measured: a parser test failed for a subprocess a
#: earlier test ran). Ignored here and only here - elsewhere the project's
#: filterwarnings = ["error"] keeps catching our own leaks.
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")

# -- helpers -------------------------------------------------------------------


def make_requirement() -> Requirement:
    return Requirement.model_validate(
        {
            "title": "Add CSV export",
            "description": "Users cannot get report data out.",
            "requester": "alice",
            "source_repo": "https://github.com/example/reports.git",
        }
    )


def make_plan() -> TestPlan:
    return TestPlan.model_validate(
        {
            "prd_id": "00000000-0000-0000-0000-000000000001",
            "code_change_id": "00000000-0000-0000-0000-000000000002",
            "items": (
                {
                    "criterion": "CSV has one header row",
                    "description": "Assert the header",
                    "test_path": "tests/test_export.py",
                },
            ),
        }
    )


def make_context(repo_root: str) -> RoleContext:
    return RoleContext(requirement=make_requirement(), repo_root=repo_root)


@pytest.fixture
def project(tmp_path: Any) -> Any:
    """Factory: a workspace directory with the given test files in it."""

    def factory(files: dict[str, str]) -> Any:
        workspace = tmp_path / "repo"
        workspace.mkdir()
        for name, content in files.items():
            path = workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return workspace

    return factory


def run_executor(workspace_dir: Any, **kwargs: Any) -> TestRun:
    executor = SandboxExecutor(**kwargs)
    return executor.execute(make_plan(), make_context(str(workspace_dir)))


# -- real execution ------------------------------------------------------------

PASS = "def test_ok():\n    assert 1 == 1\n"
FAIL = "def test_bad():\n    assert 1 == 2\n"


class TestRealExecution:
    def test_passing_suite_is_passed(self, project: Any) -> None:
        """Also the host-config guard: tmp_path is inside this repo, and an
        unpinned config would crash this run before it collected anything."""
        workspace = project({"test_ok.py": PASS})

        run = run_executor(workspace)

        assert run.status is TestStatus.PASSED
        assert run.exit_code == 0
        assert len(run.results) == 1
        result = run.results[0]
        assert result.status is TestStatus.PASSED
        assert result.test_path == "test_ok.py"
        assert result.test_name == "test_ok"
        assert result.failure_output == ""
        assert result.duration_seconds >= 0.0

    def test_failing_suite_reports_failure_output(self, project: Any) -> None:
        workspace = project({"test_ok.py": PASS, "test_bad.py": FAIL})

        run = run_executor(workspace)

        assert run.status is TestStatus.FAILED
        assert run.exit_code == 1
        assert run.passed_count == 1
        assert run.failed_count == 1
        failing = next(r for r in run.results if r.status is TestStatus.FAILED)
        assert "assert" in failing.failure_output

    def test_subdirectory_tests_map_to_paths(self, project: Any) -> None:
        workspace = project({"tests/test_export.py": PASS})

        run = run_executor(workspace)

        assert run.status is TestStatus.PASSED
        assert run.results[0].test_path == "tests/test_export.py"

    def test_collection_error_is_not_run(self, project: Any) -> None:
        """A broken import interrupts collection (exit 2): an environment
        problem, reported as NOT_RUN with the raw exit code preserved - not a
        failure that would route to the developer retry loop."""
        workspace = project(
            {"test_boom.py": "import does_not_exist\n\n\ndef test_x():\n    pass\n"}
        )

        run = run_executor(workspace)

        assert run.status is TestStatus.NOT_RUN
        assert run.exit_code == 2
        assert run.results == ()

    def test_no_tests_collected_is_not_run(self, project: Any) -> None:
        """Exit 5 with zero evidence is an environment problem, not a pass or
        a code failure - it must route to a human, not the retry loop."""
        workspace = project({"README.md": "no tests here"})

        run = run_executor(workspace)

        assert run.status is TestStatus.NOT_RUN
        assert run.exit_code == 5
        assert run.results == ()

    def test_missing_workspace_is_not_run(self) -> None:
        run = SandboxExecutor().execute(make_plan(), make_context(""))

        assert run.status is TestStatus.NOT_RUN
        assert run.exit_code == -1
        assert run.results == ()
        assert "not attempted" in run.command

    def test_timeout_is_not_run(self, project: Any) -> None:
        workspace = project(
            {"test_slow.py": "import time\n\n\ndef test_slow():\n    time.sleep(30)\n"}
        )

        run = run_executor(workspace, timeout_seconds=3.0)

        assert run.status is TestStatus.NOT_RUN
        assert run.results == ()
        assert "pytest" in run.command

    def test_the_run_leaves_the_change_under_test_untouched(self, project: Any) -> None:
        """The verify stage adding files to the workspace would corrupt the
        diff the merge gate reviews; junit and ini live outside it."""
        workspace = project({"test_ok.py": PASS})
        before = sorted(p.name for p in workspace.iterdir())

        run_executor(workspace)

        after = sorted(p.name for p in workspace.iterdir() if p.name != "__pycache__")
        assert after == before
        assert run_executor(workspace).status is TestStatus.PASSED

    def test_command_is_reproducible(self, project: Any) -> None:
        workspace = project({"test_ok.py": PASS})

        run = run_executor(workspace)

        assert "--junitxml" in run.command
        assert "--rootdir" in run.command
        assert "pytest" in run.command


class TestGoldenTaskAgainstExecutor:
    def test_the_not_run_golden_task_passes_against_the_executor(self, project: Any) -> None:
        """qa_reports_not_run_honestly is excluded from the role evals (no
        role produces a TestRun); this is its home. The scenario: the sandbox
        lacks a dependency, collection dies, and the only acceptable output is
        an honest TestRun the gate routes to a human."""
        task = next(
            task
            for task in load_all_tasks().values()
            if task.task_id == "qa_reports_not_run_honestly"
        )
        workspace = project({"test_export.py": "import psycopg2\n\n\ndef test_x():\n    pass\n"})

        run = run_executor(workspace)

        assert run.status is TestStatus.NOT_RUN
        agent_run = AgentRun(artifact=run, raw_payload=run.to_json())
        result = run_task(task, agent_run)
        assert result.passed, "\n".join(
            f"- {c.check_type}: {c.reason} ({c.detail})" for c in result.failures
        )


# -- stale evidence -------------------------------------------------------------


class NoRunWorkspace(BaseWorkspace):
    """A workspace whose commands never execute; records what was asked.

    Subclasses the real type (not a duck) so the injected factory keeps the
    executor's declared contract under mypy. ``object.__setattr__`` keeps the
    recorder working whether or not the base model is frozen.
    """

    commands: tuple[str, ...] = ()

    def execute_command(
        self, command: str, cwd: Any = None, timeout: float = 30.0
    ) -> CommandResult:
        object.__setattr__(self, "commands", (*self.commands, command))
        return CommandResult(
            command=command, exit_code=0, stdout="", stderr="", timeout_occurred=False
        )

    def file_upload(self, source_path: Any, destination_path: Any) -> Any:
        raise NotImplementedError

    def file_download(self, source_path: Any, destination_path: Any) -> Any:
        raise NotImplementedError

    def git_changes(self, path: Any) -> Any:
        raise NotImplementedError

    def git_diff(self, path: Any) -> Any:
        raise NotImplementedError


class TestStaleEvidence:
    def test_a_stale_junit_is_not_read_as_this_runs_evidence(self, project: Any) -> None:
        """If the report file survives from an earlier run and this run never
        wrote one, the old report must not become today's green light."""
        workspace = project({"test_ok.py": PASS})
        stale = workspace.parent / f"{workspace.name}.sandbox"
        stale.mkdir(exist_ok=True)
        (stale / "junit.xml").write_text(
            '<?xml version="1.0"?><testsuites><testsuite><testcase '
            'classname="test_old" name="test_old"/></testsuite></testsuites>',
            encoding="utf-8",
        )
        executor = SandboxExecutor(make_workspace=lambda root: NoRunWorkspace(working_dir=root))

        run = executor.execute(make_plan(), make_context(str(workspace)))

        assert run.status is TestStatus.NOT_RUN
        assert run.results == ()
        assert run.exit_code == 0  # the "process" succeeded; the evidence is absent


# -- pure parser ---------------------------------------------------------------


def _junit(body: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite name="pytest">'
        f"{body}</testsuite></testsuites>"
    ).encode()


class TestParseJunit:
    def test_passed_case_has_no_output(self) -> None:
        results = _parse_junit(
            _junit('<testcase classname="tests.test_a" name="test_one" time="0.01"/>'), 2000
        )
        assert len(results) == 1
        assert results[0].status is TestStatus.PASSED
        assert results[0].failure_output == ""
        assert results[0].duration_seconds == pytest.approx(0.01)

    def test_failure_text_wins_over_message(self) -> None:
        xml = _junit(
            '<testcase classname="tests.test_a" name="test_one">'
            '<failure message="short">long output body</failure></testcase>'
        )
        results = _parse_junit(xml, 2000)
        assert results[0].status is TestStatus.FAILED
        assert results[0].failure_output == "long output body"

    def test_message_only_failure_is_kept(self) -> None:
        xml = _junit(
            '<testcase classname="t" name="x"><failure message="only message"/></testcase>'
        )
        results = _parse_junit(xml, 2000)
        assert results[0].status is TestStatus.FAILED
        assert results[0].failure_output == "only message"

    def test_skipped_is_not_run(self) -> None:
        xml = _junit('<testcase classname="t" name="x"><skipped/></testcase>')
        assert _parse_junit(xml, 2000)[0].status is TestStatus.NOT_RUN

    def test_failure_output_is_truncated(self) -> None:
        xml = _junit(f'<testcase classname="t" name="x"><failure>{"y" * 500}</failure></testcase>')
        assert len(_parse_junit(xml, 100)[0].failure_output) == 100

    def test_empty_classname_falls_back_to_name_for_the_path(self) -> None:
        """Collection failures put the module in name and leave classname
        empty; the path must still be reconstructable."""
        xml = _junit('<testcase classname="" name="test_boom"><error>x</error></testcase>')
        result = _parse_junit(xml, 2000)[0]
        assert result.test_path == "test_boom.py"
        assert result.status is TestStatus.ERROR

    def test_garbage_bytes_raise_parse_error(self) -> None:
        with pytest.raises(ET.ParseError):
            _parse_junit(b"not xml at all", 2000)


class TestAggregate:
    def test_empty_is_not_run(self) -> None:
        assert _aggregate(()) is TestStatus.NOT_RUN

    def test_all_passed_is_passed(self) -> None:
        results = _parse_junit(
            _junit('<testcase classname="a" name="a"/><testcase classname="b" name="b"/>'),
            2000,
        )
        assert _aggregate(results) is TestStatus.PASSED

    def test_error_beats_failed_for_the_aggregate(self) -> None:
        """An errored suite is an environment/reporting problem; presenting it
        as plain test failure would send the developer to fix the wrong thing."""
        results = _parse_junit(
            _junit(
                '<testcase classname="a" name="a"><failure>f</failure></testcase>'
                '<testcase classname="b" name="b"><error>e</error></testcase>'
            ),
            2000,
        )
        assert _aggregate(results) is TestStatus.ERROR

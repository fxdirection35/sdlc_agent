"""Sandbox test execution: the real ``TestExecutor`` behind the verify node.

StubExecutor decided the outcome so the graph could be tested; this one runs
the suite. The OpenHands SDK supplies the workspace abstraction, so the same
executor drives a :class:`LocalWorkspace` today and a container-backed
workspace in deployment without change. LocalWorkspace is *not* a security
boundary - it executes on the host - which is fine for development and CI,
and is exactly why it is injectable rather than hardcoded (HARNESS_SPEC.md
section 7).

Three measured properties shape the command:

- **Host config leaks.** Pytest's config discovery climbs to the nearest
  ancestor config file, so a workspace that has no pytest config of its own
  inherits this repository's ``testpaths`` and ``filterwarnings`` - measured:
  a workspace run nested inside this repo crashed on our own
  ``filterwarnings = ["error"]`` before collecting a single test. The command
  therefore pins ``-c`` to an empty ini in executor-owned scratch space and
  ``--rootdir`` to the workspace. Accepted trade-off: the target repo's own
  pytest config is not honored yet; revisit when a target repo needs it.
- **The verify stage must not touch the change under test.** The junit report
  and the scratch ini live in a sibling scratch directory outside the
  workspace, never inside it.
- **Only completed-suite evidence decides per-test outcomes**, never the exit
  code alone: exit 1 with a missing report is ``NOT_RUN`` (no evidence), not a
  failure list the executor would have to invent. And only exits 0 and 1
  count as a completed suite at all - exit 2 (collection interrupted, e.g. a
  missing dependency) is an environment problem reported as ``NOT_RUN``, not
  a test failure that would burn the developer retry budget on code that is
  probably fine.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from openhands.sdk.workspace.base import BaseWorkspace
    from openhands.sdk.workspace.local import LocalWorkspace
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "sdlc_agent.sandbox needs the optional sandbox extra; "
        "install with: pip install '.[sandbox]'"
    ) from exc

from sdlc_agent.domain.qa import TestCaseResult, TestPlan, TestRun, TestStatus
from sdlc_agent.pipeline.roles import RoleContext

#: Recorded as the command when no workspace existed, so the artifact stays
#: honest about nothing having been attempted (``command`` is never blank).
_NO_WORKSPACE_COMMAND = "(not attempted: no sandbox workspace for this run)"

#: Exit code for "no process completed": no workspace, spawn failure, or
#: timeout. Matches the SDK's own convention of -1 for timed-out commands.
_NO_PROCESS = -1

#: pytest exit codes that mean the suite genuinely executed: 0 = all passed,
#: 1 = tests ran and some failed. Everything else (2 interrupted, 3 internal
#: error, 4 usage, 5 nothing collected) means no suite-level evidence.
_EVIDENCE_EXIT_CODES = frozenset({0, 1})

_INI_NAME = "sandbox.ini"
_INI_CONTENT = "# pinned by the sdlc_agent sandbox executor; keeps host pytest config out\n"
_JUNIT_NAME = "junit.xml"

#: junit child element -> test status. A testcase with no such child passed.
_OUTCOME_CHILDREN: dict[str, TestStatus] = {
    "failure": TestStatus.FAILED,
    "error": TestStatus.ERROR,
    "skipped": TestStatus.NOT_RUN,
}


@dataclass(frozen=True, slots=True)
class SandboxExecutor:
    """Runs the plan's suite in an OpenHands workspace, evidence-first.

    Attributes:
        python: Interpreter used to run pytest. Defaults to the interpreter
            this process runs on, which inside the service is the sandbox
            image's interpreter.
        timeout_seconds: Wall-clock ceiling for the whole suite. A suite that
            exceeds it never produced evidence, so it is ``NOT_RUN`` - a
            timeout is an environment problem, and burning the developer
            retry budget on it is the failure the QA golden tasks describe.
        max_failure_chars: Truncation for captured failure output before it
            enters an artifact (HARNESS_SPEC.md section 4.1: no unbounded
            text into agent-readable context).
        make_workspace: Builds the workspace for one run's repo root.
            Injectable so deployment can swap in a container-backed
            implementation and tests can fake execution entirely.
    """

    python: str = field(default_factory=lambda: sys.executable)
    timeout_seconds: float = 600.0
    max_failure_chars: int = 2000
    make_workspace: Callable[[str], BaseWorkspace] = lambda root: LocalWorkspace(working_dir=root)

    def execute(self, plan: TestPlan, context: RoleContext) -> TestRun:
        """Run the suite in the context's workspace and report the evidence.

        Every path that cannot produce per-test evidence returns ``NOT_RUN``
        with the command and raw exit code preserved: the quality gate then
        routes to a human instead of guessing, which is the whole design of
        ``QualityGate.from_evidence``.
        """
        if not context.repo_root:
            return TestRun(
                test_plan_id=plan.artifact_id,
                status=TestStatus.NOT_RUN,
                results=(),
                command=_NO_WORKSPACE_COMMAND,
                exit_code=_NO_PROCESS,
            )

        workspace_dir = Path(context.repo_root)
        scratch = workspace_dir.parent / f"{workspace_dir.name}.sandbox"
        scratch.mkdir(parents=True, exist_ok=True)
        ini = scratch / _INI_NAME
        ini.write_text(_INI_CONTENT, encoding="utf-8")
        junit = scratch / _JUNIT_NAME
        # A report from an earlier run of this workspace must never be
        # readable as this run's evidence.
        junit.unlink(missing_ok=True)

        command = (
            f'"{self.python}" -m pytest -q . '
            f'--rootdir=. -c="{ini.resolve()}" '
            f'--junitxml="{junit.resolve()}" -p no:cacheprovider'
        )
        workspace = self.make_workspace(context.repo_root)
        try:
            completed = workspace.execute_command(command, timeout=self.timeout_seconds)
        except Exception:
            # The workspace failed to run the command at all (shell missing,
            # spawn error). No process, no evidence.
            return self._not_run(plan, command)

        if completed.timeout_occurred:
            return self._not_run(plan, command)

        if completed.exit_code not in _EVIDENCE_EXIT_CODES:
            # Only 0 (ran clean) and 1 (ran, tests failed) mean the suite
            # actually executed. 2 is an interrupted collection - a missing
            # dependency, a syntax error - which is an environment problem:
            # reporting it as a test failure would spend the developer retry
            # budget fixing code that is probably fine (the QA golden task
            # qa_reports_not_run_honestly exists for exactly this). 3/4 are
            # pytest-internal or usage errors. All route to a human.
            return self._not_run(plan, command, exit_code=completed.exit_code)

        results = _read_results(junit, self.max_failure_chars)
        return TestRun(
            test_plan_id=plan.artifact_id,
            status=_aggregate(results),
            results=results,
            command=command,
            exit_code=completed.exit_code,
        )

    def _not_run(self, plan: TestPlan, command: str, exit_code: int = _NO_PROCESS) -> TestRun:
        return TestRun(
            test_plan_id=plan.artifact_id,
            status=TestStatus.NOT_RUN,
            results=(),
            command=command,
            exit_code=exit_code,
        )


def _read_results(junit: Path, max_failure_chars: int) -> tuple[TestCaseResult, ...]:
    """Per-test evidence from the junit report; ``()`` when there is none.

    A missing file (suite never wrote a report - usage error, crash) and an
    unparseable one are both "no evidence", which the caller reports as
    ``NOT_RUN``; distinguishing them would change nothing downstream.
    """
    try:
        data = junit.read_bytes()
    except OSError:
        return ()
    try:
        return _parse_junit(data, max_failure_chars)
    except ET.ParseError:
        return ()


def _parse_junit(data: bytes, max_failure_chars: int) -> tuple[TestCaseResult, ...]:
    """Turn one pytest junit report into per-test results.

    Kept as a pure function of the bytes so it can be tested against crafted
    reports - including shapes this pipeline's own runs may never produce.
    """
    root = ET.fromstring(data)
    results = []
    for case in root.iter("testcase"):
        status = TestStatus.PASSED
        output = ""
        for child in case:
            mapped = _OUTCOME_CHILDREN.get(child.tag)
            if mapped is not None:
                status = mapped
                output = (child.text or child.get("message") or "").strip()
                break
        classname = case.get("classname") or ""
        name = case.get("name") or classname or "unknown"
        results.append(
            TestCaseResult(
                test_path=_test_path(classname, name),
                test_name=name,
                status=status,
                duration_seconds=_duration(case),
                failure_output=output[:max_failure_chars],
            )
        )
    return tuple(results)


def _test_path(classname: str, name: str) -> str:
    """Workspace-relative file for one testcase.

    pytest's junit gives a dotted module path in ``classname`` (collection
    failures instead put it in ``name`` and leave ``classname`` empty), so the
    mapping is dots-to-slashes plus ``.py``. Tests nested in classes keep the
    class in the dotted path, so their reconstructed path names a nonexistent
    file - traceable, which is what the field promises, if not exact.
    """
    dotted = classname or name
    return f"{dotted.replace('.', '/')}.py"


def _duration(case: Any) -> float:
    try:
        return max(float(case.get("time") or 0), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _aggregate(results: tuple[TestCaseResult, ...]) -> TestStatus:
    """Derive the run status from per-test evidence, never the reverse.

    An empty result set is ``NOT_RUN`` regardless of exit code: zero executed
    tests is exactly the false-green the TestRun model validator refuses, and
    "collected nothing" is an environment problem, not a code failure.
    """
    if not results:
        return TestStatus.NOT_RUN
    if all(result.status is TestStatus.PASSED for result in results):
        return TestStatus.PASSED
    if any(result.status is TestStatus.ERROR for result in results):
        return TestStatus.ERROR
    return TestStatus.FAILED


__all__ = ["SandboxExecutor"]

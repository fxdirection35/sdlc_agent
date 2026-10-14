"""Running golden tasks and reporting the result.

The runner takes a completed agent run - the artifact it produced plus what it
spent and which tools it touched - and evaluates it against the task's checks.
Everything is deterministic: the same run always scores the same, which is what
lets the eval set gate a commit instead of merely informing it.

Results are recorded per check rather than per task, so a failure names the
expectation that broke. "Task failed" tells a prompt tuner nothing; "PRD lost
the acceptance criterion about the header row" tells them where to look.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from sdlc_agent.domain import ARTIFACT_TYPES, Artifact
from sdlc_agent.evals.tasks import (
    AnyCheck,
    EvalDimension,
    FieldNonEmptyCheck,
    GoldenTask,
    KindCorrectCheck,
    MaxOutputTokensCheck,
    SchemaValidCheck,
    TextAbsentCheck,
    TextMatchesCheck,
    ToolsWithinCheck,
)


@dataclass(frozen=True, slots=True)
class AgentRun:
    """What one agent invocation produced.

    Attributes:
        artifact: The artifact returned, or ``None`` if the agent failed to
            produce one at all. A missing artifact is itself an evaluable
            outcome rather than an exception, because "returned nothing" is the
            failure most worth reporting.
        raw_payload: The raw JSON the agent emitted, before validation. Kept
            separately because a schema-invalid response is still informative:
            the report can say what was wrong with it.
        output_tokens: Tokens the agent emitted.
        tools_called: Tool names invoked, in call order.
        duration_seconds: Wall time, reported but not yet gated on.
    """

    artifact: Artifact | None
    raw_payload: str = ""
    output_tokens: int = 0
    tools_called: tuple[str, ...] = ()
    duration_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one check against one run."""

    check_type: str
    dimension: EvalDimension
    passed: bool
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Outcome of one golden task."""

    task_id: str
    role: str
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate outcome of an eval run, and the thing a gate inspects."""

    results: tuple[TaskResult, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def is_green(self) -> bool:
        """True only when every task passed and there was at least one task.

        An empty eval set is not green. Reporting zero failures when nothing
        ran is exactly the false signal the QA domain models refuse to produce,
        and the same reasoning applies here: it would let a commit through the
        gate on the strength of tests that never executed.
        """
        return self.total > 0 and self.failed == 0

    def by_dimension(self) -> dict[EvalDimension, tuple[int, int]]:
        """Pass and total counts per dimension, to expose thin coverage.

        A dimension with zero total is unexercised. That is legitimate early on,
        but it should be visible rather than implied by its absence.
        """
        counts: dict[EvalDimension, tuple[int, int]] = {
            dimension: (0, 0) for dimension in EvalDimension
        }
        for result in self.results:
            for check in result.checks:
                passed, total = counts[check.dimension]
                counts[check.dimension] = (passed + int(check.passed), total + 1)
        return counts

    def render(self) -> str:
        """Human-readable summary, for CI output and a failed gate's message."""
        lines = [f"eval: {self.passed}/{self.total} tasks passed"]
        for result in self.results:
            mark = "PASS" if result.passed else "FAIL"
            lines.append(f"  [{mark}] {result.role}: {result.task_id}")
            for failure in result.failures:
                lines.append(f"      - {failure.check_type}: {failure.reason}")
                if failure.detail:
                    lines.append(f"        {failure.detail}")
        return "\n".join(lines)


class EvalRunError(Exception):
    """Raised when a task cannot be evaluated at all.

    Distinct from a failed check: a broken eval definition is a bug in the eval
    set, not a regression in the agent, and conflating the two sends whoever
    investigates to the wrong place.
    """


def _field_value(artifact: Artifact, field_name: str) -> Any:
    """Read one field off an artifact for a content check.

    Raises:
        EvalRunError: If the field does not exist on the model. A check naming
            a field the artifact does not have is a mistake in the eval set, and
            silently passing it would hide that mistake forever.
    """
    # Read the field set off the class, not the instance: pydantic deprecated
    # instance access in 2.11 and warns, which fails the suite outright under
    # this project's filterwarnings = ["error"].
    fields = type(artifact).model_fields
    if field_name not in fields:
        raise EvalRunError(
            f"field {field_name!r} does not exist on {type(artifact).__name__}; "
            f"available: {', '.join(sorted(fields))}"
        )
    return getattr(artifact, field_name)


def _as_text(value: Any) -> str:
    """Render a field value as text for regex checks.

    Collections are joined so a pattern can match across entries without the
    check having to know which fields are collections. Nested models contribute
    their *values* only, never their field names.

    The names matter more than they look. Rendering a nested model with
    ``str(model_dump())`` puts the keys in the text too, so a check for the word
    "verification" would match the key ``verification`` even when no criterion
    actually states one - a check that can never fail, which is worse than no
    check because it reports coverage that does not exist.
    """
    if isinstance(value, tuple | list):
        return "\n".join(_as_text(item) for item in value)
    if hasattr(value, "model_dump"):
        return "\n".join(_as_text(item) for item in value.model_dump().values())
    return str(value)


def _evaluate_check(check: AnyCheck, run: AgentRun) -> CheckResult:
    """Apply one check to one run.

    Checks that need an artifact treat a missing artifact as a failure rather
    than an error: an agent that produced nothing genuinely failed the
    expectation, which is different from the eval set being broken.
    """
    if isinstance(check, MaxOutputTokensCheck):
        passed = run.output_tokens <= check.limit
        return CheckResult(
            check_type=check.type,
            dimension=check.dimension,
            passed=passed,
            reason=check.reason,
            detail=("" if passed else f"emitted {run.output_tokens} tokens, limit {check.limit}"),
        )

    if isinstance(check, ToolsWithinCheck):
        disallowed = sorted({tool for tool in run.tools_called if tool not in check.allowed_tools})
        return CheckResult(
            check_type=check.type,
            dimension=check.dimension,
            passed=not disallowed,
            reason=check.reason,
            detail=(
                ""
                if not disallowed
                else f"called {', '.join(disallowed)}; allowed: {', '.join(check.allowed_tools)}"
            ),
        )

    # Everything below inspects the artifact.
    if run.artifact is None:
        return CheckResult(
            check_type=check.type,
            dimension=check.dimension,
            passed=False,
            reason=check.reason,
            detail="agent produced no artifact",
        )

    if isinstance(check, SchemaValidCheck | KindCorrectCheck):
        expected_type = ARTIFACT_TYPES.get(check.expected_kind)
        if expected_type is None:
            raise EvalRunError(f"no artifact type registered for kind {check.expected_kind!r}")
        passed = isinstance(run.artifact, expected_type)
        detail = "" if passed else f"got {type(run.artifact).__name__}"
        if not passed and run.raw_payload:
            # Re-validate the raw payload so the report carries the real reason
            # instead of just "wrong type", which is what makes a schema
            # failure diagnosable from the report alone.
            try:
                expected_type.model_validate_json(run.raw_payload)
            except ValidationError as exc:
                errors = exc.errors()
                if errors:
                    first = errors[0]
                    location = ".".join(str(part) for part in first["loc"])
                    detail = f"{detail}: {location} {first['msg']}".strip(": ")
        return CheckResult(
            check_type=check.type,
            dimension=check.dimension,
            passed=passed,
            reason=check.reason,
            detail=detail,
        )

    if isinstance(check, FieldNonEmptyCheck):
        value = _field_value(run.artifact, check.field_name)
        text = _as_text(value).strip()
        return CheckResult(
            check_type=check.type,
            dimension=check.dimension,
            passed=bool(text),
            reason=check.reason,
            detail="" if text else f"{check.field_name} is empty",
        )

    if isinstance(check, TextMatchesCheck | TextAbsentCheck):
        text = _as_text(_field_value(run.artifact, check.field_name))
        found = bool(re.search(check.pattern, text, re.IGNORECASE))
        wants_match = isinstance(check, TextMatchesCheck)
        passed = found if wants_match else not found
        detail = ""
        if not passed:
            detail = (
                f"{check.field_name} does not match {check.pattern!r}"
                if wants_match
                else f"{check.field_name} matched forbidden pattern {check.pattern!r}"
            )
        return CheckResult(
            check_type=check.type,
            dimension=check.dimension,
            passed=passed,
            reason=check.reason,
            detail=detail,
        )

    raise EvalRunError(f"no runner support for check type {type(check).__name__}")


def run_task(task: GoldenTask, run: AgentRun) -> TaskResult:
    """Evaluate one golden task against one agent run."""
    return TaskResult(
        task_id=task.task_id,
        role=task.role.value,
        checks=tuple(_evaluate_check(check, run) for check in task.checks),
    )


def run_eval_set(
    tasks_and_runs: dict[str, tuple[GoldenTask, AgentRun]],
) -> EvalReport:
    """Evaluate a whole set, keyed by task id.

    Args:
        tasks_and_runs: Mapping of task id to the task and the run that answered
            it. The key must match the task's own ``task_id``; a mismatch is an
            error, because a mislabelled entry would silently score one task
            against another's expectations.

    Raises:
        EvalRunError: On a key mismatch, or if any check cannot be evaluated.
    """
    for key, (task, _) in tasks_and_runs.items():
        if key != task.task_id:
            raise EvalRunError(
                f"entry keyed {key!r} holds task {task.task_id!r}; keys must match task_id"
            )

    results = [run_task(task, run) for task, run in tasks_and_runs.values()]
    return EvalReport(results=tuple(results))

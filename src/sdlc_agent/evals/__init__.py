"""Golden-task evaluation set for the pipeline's role agents.

HARNESS_SPEC.md section 5.2: every role ships with golden tasks, and changing a
prompt or a flow requires running them to catch a regression. The task
definitions live as YAML data next to the code rather than in it, so whoever
tunes a prompt can edit an expectation without touching the runner.

Tasks load lazily and validate on load, so a malformed expectation fails with
the offending file and task named, rather than as a confusing failure partway
through an eval run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sdlc_agent.evals.runner import (
    AgentRun,
    CheckResult,
    EvalReport,
    EvalRunError,
    TaskResult,
    run_eval_set,
    run_task,
)
from sdlc_agent.evals.tasks import (
    CHECK_TYPES,
    AnyCheck,
    Check,
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

#: Golden task definitions live here, as one YAML file per role.
EVAL_DIR = Path(__file__).parent / "golden"


def load_tasks(path: Path) -> tuple[GoldenTask, ...]:
    """Load every golden task from one YAML file.

    A file holds a top-level list of task mappings. Empty files and files whose
    top level is not a list are errors: both mean the eval set silently contains
    fewer tasks than it appears to, and an eval set that runs nothing reports
    green.

    Raises:
        FileNotFoundError: If the path does not exist.
        EvalRunError: If the file is empty, malformed, or holds a task that
            fails validation. The message names the file and, where it can be
            determined, the offending task.
    """
    if not path.exists():
        raise FileNotFoundError(f"golden task file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EvalRunError(f"{path}: not valid YAML: {exc}") from exc

    if raw is None:
        raise EvalRunError(f"{path}: file is empty, so it defines no golden tasks")
    if not isinstance(raw, list):
        raise EvalRunError(f"{path}: expected a top-level list of tasks, got {type(raw).__name__}")

    tasks: list[GoldenTask] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise EvalRunError(f"{path}: task {index} is a {type(entry).__name__}, not a mapping")
        label = entry.get("task_id", f"index {index}")
        try:
            tasks.append(GoldenTask.model_validate(entry))
        except Exception as exc:
            # Broad by design: any pydantic failure is re-raised with the file
            # and task named, which is the information a plain ValidationError
            # lacks. Swallowing nothing - every path raises.
            known = ", ".join(CHECK_TYPES)
            hint = f" (known check types: {known})" if "type" in str(exc) else ""
            raise EvalRunError(f"{path}: task {label} is invalid{hint}: {exc}") from exc

    if not tasks:
        raise EvalRunError(f"{path}: defines an empty task list")

    return tuple(tasks)


def load_all_tasks(directory: Path | None = None) -> dict[str, GoldenTask]:
    """Load every golden task under a directory, keyed by task id.

    Args:
        directory: Where to look. Defaults to :data:`EVAL_DIR`.

    Returns:
        Mapping of task id to task.

    Raises:
        EvalRunError: If any file fails to load, or if two tasks anywhere in the
            directory share an id. A duplicate id would make a keyed report
            silently drop one task, so it is refused rather than tolerated.
    """
    root = directory if directory is not None else EVAL_DIR
    if not root.exists():
        raise EvalRunError(f"golden task directory not found: {root}")

    paths = sorted(set(root.rglob("*.yaml")) | set(root.rglob("*.yml")))

    tasks: dict[str, GoldenTask] = {}
    for path in paths:
        for task in load_tasks(path):
            if task.task_id in tasks:
                raise EvalRunError(
                    f"duplicate task_id {task.task_id!r}: already loaded from another file, "
                    f"conflict in {path.name}"
                )
            tasks[task.task_id] = task

    return tasks


def tasks_for_role(role: str, directory: Path | None = None) -> dict[str, GoldenTask]:
    """Load only the golden tasks belonging to one role.

    Used to run a single role's eval set after tuning its prompt, without
    spending on the others.
    """
    return {
        task_id: task
        for task_id, task in load_all_tasks(directory).items()
        if task.role.value == role
    }


def summarize_coverage(tasks: dict[str, GoldenTask]) -> dict[str, Any]:
    """Report which dimensions the eval set exercises, per role.

    HARNESS_SPEC.md section 5.2 wants thin coverage visible rather than implied.
    A role with no efficiency or process check is not necessarily wrong, but
    saying so out loud is what makes the gap a decision instead of an accident.
    """
    coverage: dict[str, dict[str, list[str]]] = {}
    for task in tasks.values():
        role = task.role.value
        coverage.setdefault(role, {})
        for dimension in task.dimensions_covered():
            coverage[role].setdefault(dimension.value, []).append(task.task_id)

    return {
        "tasks_per_role": {
            role: sum(1 for task in tasks.values() if task.role.value == role) for role in coverage
        },
        "dimensions": coverage,
        "unexercised": {
            role: sorted(
                dimension.value for dimension in EvalDimension if dimension.value not in dimensions
            )
            for role, dimensions in coverage.items()
        },
    }


__all__ = [
    "CHECK_TYPES",
    "EVAL_DIR",
    "AgentRun",
    "AnyCheck",
    "Check",
    "CheckResult",
    "EvalDimension",
    "EvalReport",
    "EvalRunError",
    "FieldNonEmptyCheck",
    "GoldenTask",
    "KindCorrectCheck",
    "MaxOutputTokensCheck",
    "SchemaValidCheck",
    "TaskResult",
    "TextAbsentCheck",
    "TextMatchesCheck",
    "ToolsWithinCheck",
    "load_all_tasks",
    "load_tasks",
    "run_eval_set",
    "run_task",
    "summarize_coverage",
    "tasks_for_role",
]

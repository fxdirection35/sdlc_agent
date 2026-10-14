"""The golden set run against the real litellm-backed roles: record once, replay forever.

pytest-recording (vcrpy) intercepts at the HTTP layer, so:

- **Cassette present** (the committed default): every call is replayed, no
  key is consulted, no money is spent. A dummy key is injected for replay
  only, because ``LitellmRole`` resolves the environment variable before
  the call - the HTTP layer never sees it.
- **Cassette missing and ``ANTHROPIC_API_KEY`` set**: the run *records*
  (vcr's default mode "once"). This makes real API calls - one per task,
  at the models and effort levels ``config.example.yaml`` names - and is
  the deliberate first-record / re-record flow.
- **Neither**: the task skips. A skipped eval is honest; an eval that never
  ran reporting green is exactly what ``EvalReport.is_green`` refuses.

Re-record after changing a prompt, a golden task or the model config (spends
real money; existing cassettes force replay, so they must be deleted or the
record mode overridden):

    export ANTHROPIC_API_KEY=sk-ant-...
    .venv/Scripts/python -m pytest tests/test_llm_golden.py --record-mode=all
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sdlc_agent.config import PlatformConfig, load_config
from sdlc_agent.evals import GoldenTask, load_all_tasks, run_task
from sdlc_agent.evals.role_runner import exclusion_reason, run_golden_task
from sdlc_agent.pipeline.llm_roles import LitellmRole

#: One cassette per task, so re-recording one task cannot invalidate the rest.
#: The directory is pytest-recording's default: ``<module>/cassettes/<name>``.
CASSETTE_DIR = Path(__file__).parent / "cassettes" / "test_llm_golden"

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.example.yaml"

#: The addressable set: one param per task the roles can actually produce.
TASKS: tuple[GoldenTask, ...] = tuple(
    task for task in load_all_tasks().values() if not exclusion_reason(task)
)

#: Injected only when replaying with no real key in the environment. Never
#: sent anywhere: vcr answers before the transport runs.
_REPLAY_KEY = "sk-ant-replay-only"


def _cassette_for(task_id: str) -> Path:
    """The cassette pytest-recording will look for, matching its default
    naming: the test node name, parametrized id, no sanitized characters."""
    return CASSETTE_DIR / f"test_golden_task_passes[{task_id}].yaml"


def _params() -> Iterator[Any]:
    for task in TASKS:
        cassette = _cassette_for(task.task_id)
        can_run = cassette.exists() or bool(os.environ.get("ANTHROPIC_API_KEY"))
        yield pytest.param(
            task,
            id=task.task_id,
            marks=(
                ()
                if can_run
                else (
                    pytest.mark.skip(
                        reason=(
                            f"no cassette {cassette.name} and no ANTHROPIC_API_KEY; "
                            "see module docstring for the record flow"
                        )
                    ),
                )
            ),
        )


@pytest.fixture
def config() -> PlatformConfig:
    """The shipped example config: the eval measures what users get.

    Loading needs no credentials (keys resolve lazily), so this works
    identically when recording and when replaying.
    """
    return load_config(CONFIG_PATH)


@pytest.mark.vcr
@pytest.mark.parametrize("task", list(_params()))
def test_golden_task_passes(
    task: GoldenTask, config: PlatformConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cassette = _cassette_for(task.task_id)
    if cassette.exists() and not os.environ.get("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", _REPLAY_KEY)

    role = LitellmRole(task.role, config)
    result = run_task(task, run_golden_task(role, task))

    failure_lines = [
        f"- {check.check_type}: {check.reason}" + (f" ({check.detail})" if check.detail else "")
        for check in result.failures
    ]
    assert result.passed, f"{task.task_id} ({task.role.value}):\n" + "\n".join(failure_lines)

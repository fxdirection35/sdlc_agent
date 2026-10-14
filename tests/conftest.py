"""Shared pytest configuration and fixtures.

The fixtures live here so pytest injects them without any import from the test
modules. They are factories rather than values because most tests need to vary
one field and keep the rest fixed, and a factory keeps that intent visible.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sdlc_agent.domain.base import ArtifactKind, PipelineRole
from sdlc_agent.domain.product import AcceptanceCriterion, ProductRequirements
from sdlc_agent.evals.runner import AgentRun
from sdlc_agent.evals.tasks import GoldenTask

#: Repository root, i.e. the parent of ``tests/``.
REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest's temp tree inside the repository, and out of git.

    The system temp directory is not writable in every environment this runs in:
    a sandboxed runner can raise PermissionError for it. When that happens
    Python's ``tempfile`` falls back to the current directory, so pytest creates
    ``pytest-of-<user>/`` in the repository root and pollutes the working tree on
    every run. Pinning basetemp to a gitignored path makes the behaviour the same
    whether or not system temp is available.

    An explicit ``--basetemp`` still wins.
    """
    if config.option.basetemp is None:
        config.option.basetemp = REPO_ROOT / ".pytest-tmp"


def _criterion(statement: str, verification: str) -> AcceptanceCriterion:
    return AcceptanceCriterion.model_validate(
        {"statement": statement, "verification": verification}
    )


@pytest.fixture
def make_prd() -> Callable[..., ProductRequirements]:
    """Build a valid PRD, overriding only the fields a test cares about."""

    def factory(
        problem_statement: str = "Report users cannot get data out of the app.",
        goals: tuple[str, ...] = ("Export any filtered report as CSV",),
        acceptance_criteria: tuple[AcceptanceCriterion, ...] | None = None,
        non_goals: tuple[str, ...] = (),
        open_questions: tuple[str, ...] = (),
        **extra: Any,
    ) -> ProductRequirements:
        criteria = acceptance_criteria
        if criteria is None:
            criteria = (
                _criterion("CSV has one header row", "pytest tests/test_export.py::test_header"),
                _criterion("Commas are escaped", "pytest tests/test_export.py::test_escaping"),
            )
        return ProductRequirements.model_validate(
            {
                "requirement_id": uuid.uuid4(),
                "problem_statement": problem_statement,
                "goals": goals,
                "acceptance_criteria": criteria,
                "non_goals": non_goals,
                "open_questions": open_questions,
                **extra,
            }
        )

    return factory


@pytest.fixture
def sample_pm_task() -> GoldenTask:
    """A product_manager task mixing outcome, efficiency and process checks."""
    return GoldenTask.model_validate(
        {
            "task_id": "pm_fixture",
            "role": PipelineRole.PRODUCT_MANAGER.value,
            "prompt": "Users cannot export report data.",
            "checks": [
                {
                    "type": "schema_valid",
                    "expected_kind": ArtifactKind.PRD.value,
                    "reason": "must be a valid PRD",
                },
                {
                    "type": "field_non_empty",
                    "field_name": "goals",
                    "reason": "goals drive the acceptance criteria",
                },
                {
                    "type": "text_matches",
                    "field_name": "acceptance_criteria",
                    "pattern": "pytest",
                    "reason": "each criterion must name an executable verification",
                },
                {
                    "type": "max_output_tokens",
                    "limit": 8000,
                    "reason": "a longer PRD is padding",
                },
            ],
        }
    )


@pytest.fixture
def passing_prd_run(make_prd: Callable[..., ProductRequirements]) -> Callable[..., AgentRun]:
    """A run that satisfies the sample task."""

    def factory(**overrides: Any) -> AgentRun:
        return AgentRun(
            artifact=make_prd(),
            raw_payload=make_prd().to_json(),
            output_tokens=1200,
            tools_called=("read_file",),
            **overrides,
        )

    return factory


@pytest.fixture
def failing_prd_run(make_prd: Callable[..., ProductRequirements]) -> Callable[..., AgentRun]:
    """A run whose criteria name no executable verification, breaking text_matches."""

    def factory(**overrides: Any) -> AgentRun:
        prd = make_prd(
            acceptance_criteria=(_criterion("Export works", "reviewer confirms it looks right"),),
        )
        defaults: dict[str, Any] = {
            "artifact": prd,
            "raw_payload": prd.to_json(),
            "output_tokens": 1200,
        }
        defaults.update(overrides)
        return AgentRun(**defaults)

    return factory


@pytest.fixture
def prd_payload_missing_criteria() -> Callable[[], str]:
    """A PRD-shaped payload that fails validation, for schema-failure reporting.

    Built as raw JSON rather than through the model, since the point is a
    payload the model rejects. ``model_construct`` would skip validation but
    leave the field absent from the instance, which serializes unpredictably.
    """

    def factory() -> str:
        return json.dumps(
            {
                "kind": ArtifactKind.PRD.value,
                "requirement_id": str(uuid.uuid4()),
                "problem_statement": "p",
                "goals": ["g"],
                # acceptance_criteria deliberately absent: this is the defect.
            }
        )

    return factory

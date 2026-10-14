"""Tests for the product-role artifacts: Requirement and ProductRequirements."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from sdlc_agent.domain.product import AcceptanceCriterion, ProductRequirements, Requirement


def make_requirement(**overrides: object) -> Requirement:
    payload: dict[str, object] = {
        "title": "Add CSV export to the reports page",
        "description": "Users need to export filtered reports as CSV.",
        "requester": "alice",
        "source_repo": "https://github.com/fxdirection35/sdlc_agent.git",
    }
    payload.update(overrides)
    return Requirement.model_validate(payload)


class TestRequirement:
    def test_valid_requirement(self) -> None:
        requirement = make_requirement()

        assert requirement.title == "Add CSV export to the reports page"
        assert requirement.requester == "alice"
        assert requirement.kind.value == "requirement"

    def test_strips_surrounding_whitespace(self) -> None:
        requirement = make_requirement(title="  Padded title  ", requester="  alice ")

        assert requirement.title == "Padded title"
        assert requirement.requester == "alice"

    @pytest.mark.parametrize("field", ["title", "description", "requester", "source_repo"])
    def test_rejects_blank_fields(self, field: str) -> None:
        """Whitespace-only input must fail at construction, not downstream."""
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_requirement(**{field: "   "})

    @pytest.mark.parametrize("field", ["title", "description", "requester", "source_repo"])
    def test_rejects_missing_fields(self, field: str) -> None:
        payload = {
            "title": "t",
            "description": "d",
            "requester": "r",
            "source_repo": "s",
        }
        del payload[field]

        with pytest.raises(ValidationError):
            Requirement.model_validate(payload)

    def test_rejects_overlong_title(self) -> None:
        with pytest.raises(ValidationError):
            make_requirement(title="x" * 201)

    def test_rejects_extra_fields(self) -> None:
        """extra="forbid" catches a hallucinated field from an LLM payload."""
        with pytest.raises(ValidationError, match="priority"):
            Requirement.model_validate(
                {
                    "title": "t",
                    "description": "d",
                    "requester": "r",
                    "source_repo": "s",
                    "priority": "high",
                }
            )

    def test_json_round_trip(self) -> None:
        original = make_requirement()

        restored = Requirement.from_json(original.to_json())

        assert restored.artifact_id == original.artifact_id
        assert restored.title == original.title
        assert restored.source_repo == original.source_repo


def make_criterion(**overrides: object) -> AcceptanceCriterion:
    payload: dict[str, object] = {
        "statement": "Export produces a CSV with one header row and one row per result",
        "verification": "pytest tests/test_export.py::test_csv_shape",
    }
    payload.update(overrides)
    return AcceptanceCriterion.model_validate(payload)


def make_prd(**overrides: object) -> ProductRequirements:
    payload: dict[str, object] = {
        "requirement_id": uuid.uuid4(),
        "problem_statement": "Report consumers cannot get data out of the app.",
        "goals": ("Users can export any filtered report as CSV",),
        "non_goals": ("Excel format support",),
        "acceptance_criteria": (make_criterion(),),
    }
    payload.update(overrides)
    return ProductRequirements.model_validate(payload)


class TestAcceptanceCriterion:
    def test_valid_criterion(self) -> None:
        criterion = make_criterion()

        assert criterion.verification.startswith("pytest")

    @pytest.mark.parametrize("field", ["statement", "verification"])
    def test_rejects_blank_fields(self, field: str) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_criterion(**{field: "  "})

    def test_is_frozen(self) -> None:
        criterion = make_criterion()

        with pytest.raises(ValidationError):
            criterion.statement = "changed"  # type: ignore[misc]


class TestProductRequirements:
    def test_valid_prd(self) -> None:
        prd = make_prd()

        assert prd.kind.value == "prd"
        assert len(prd.goals) == 1
        assert len(prd.acceptance_criteria) == 1
        assert isinstance(prd.requirement_id, uuid.UUID)

    def test_optional_collections_default_empty(self) -> None:
        prd = ProductRequirements.model_validate(
            {
                "requirement_id": uuid.uuid4(),
                "problem_statement": "p",
                "goals": ("g",),
                "acceptance_criteria": (make_criterion(),),
            }
        )

        assert prd.non_goals == ()
        assert prd.open_questions == ()

    def test_requires_at_least_one_goal(self) -> None:
        """A PRD with no goal gives the architect nothing to design against."""
        with pytest.raises(ValidationError):
            make_prd(goals=())

    def test_requires_at_least_one_acceptance_criterion(self) -> None:
        """The QA agent derives its whole plan from these; none means no gate."""
        with pytest.raises(ValidationError):
            make_prd(acceptance_criteria=())

    def test_rejects_blank_goal(self) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_prd(goals=("real goal", "   "))

    def test_rejects_blank_problem_statement(self) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            make_prd(problem_statement="\n\t ")

    def test_json_round_trip_preserves_nested_criteria(self) -> None:
        original = make_prd(
            acceptance_criteria=(
                make_criterion(),
                make_criterion(statement="Second one"),
            ),
        )

        restored = ProductRequirements.from_json(original.to_json())

        assert len(restored.acceptance_criteria) == 2
        assert restored.acceptance_criteria[1].statement == "Second one"
        assert restored.requirement_id == original.requirement_id

    def test_is_frozen(self) -> None:
        prd = make_prd()

        with pytest.raises(ValidationError):
            prd.problem_statement = "rewritten"  # type: ignore[misc]

"""Golden-task evaluation: expectations, checks and the tasks that carry them.

HARNESS_SPEC.md section 5.2 requires every role agent to have golden tasks
before it ships, and requires deterministic checks to come first: an
LLM-as-judge is added only where a deterministic check cannot express the
requirement. The check types here are all deterministic on purpose. There is
deliberately no judge in this module - adding one is a decision that needs a
concrete requirement the checks below cannot cover, not a speculative
extension point.

Each check declares which of the four evaluation dimensions it serves, so a
report can show where coverage is thin. A task that only checks outcomes is
legitimate, but the gap should be visible rather than implied.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sdlc_agent.domain.base import ArtifactKind, PipelineRole, reject_blank


class EvalDimension(StrEnum):
    """The four dimensions an evaluation should cover.

    Outcome and process are what the deterministic checks serve today. Style
    and efficiency are declared so a report can show they are unexercised;
    filling them with expensive judged checks is a later, evidence-driven
    decision rather than something to pre-build.
    """

    OUTCOME = "outcome"
    PROCESS = "process"
    STYLE = "style"
    EFFICIENCY = "efficiency"


class Check(BaseModel):
    """Base for every deterministic expectation on one golden task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    dimension: EvalDimension
    #: Why this expectation exists. Shown in the report when the check fails, so
    #: a failure reads as "the PRD lost a criterion" rather than "assert false".
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        return reject_blank(value)


class SchemaValidCheck(Check):
    """The agent must return an artifact that validates against its model."""

    type: Literal["schema_valid"] = "schema_valid"
    dimension: Literal[EvalDimension.OUTCOME] = EvalDimension.OUTCOME
    expected_kind: ArtifactKind


class KindCorrectCheck(Check):
    """The artifact must be of the kind this role is responsible for."""

    type: Literal["kind_correct"] = "kind_correct"
    dimension: Literal[EvalDimension.OUTCOME] = EvalDimension.OUTCOME
    expected_kind: ArtifactKind


class FieldNonEmptyCheck(Check):
    """A named field must be present and non-empty.

    Catches the most common agent failure mode: a structurally valid artifact
    whose meaningful field was left blank or filled with a placeholder.
    """

    type: Literal["field_non_empty"] = "field_non_empty"
    dimension: Literal[EvalDimension.OUTCOME] = EvalDimension.OUTCOME
    field_name: str = Field(min_length=1)

    @field_validator("field_name")
    @classmethod
    def _strip_field(cls, value: str) -> str:
        return reject_blank(value)


class TextMatchesCheck(Check):
    """A field must match a regex, for content that must be present."""

    type: Literal["text_matches"] = "text_matches"
    dimension: Literal[EvalDimension.OUTCOME] = EvalDimension.OUTCOME
    field_name: str = Field(min_length=1)
    pattern: str = Field(min_length=1)

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, value: str) -> str:
        """Reject a broken regex at load time, not on the first eval run."""
        pattern = reject_blank(value)
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return pattern

    @field_validator("field_name")
    @classmethod
    def _strip_field(cls, value: str) -> str:
        return reject_blank(value)


class TextAbsentCheck(Check):
    """A field must NOT match a regex, for content that must never appear.

    Used for leaked credentials, placeholder text such as ``TODO``, and
    invented API names. A negative check is the only way to catch these, since
    the artifact otherwise validates fine.
    """

    type: Literal["text_absent"] = "text_absent"
    dimension: Literal[EvalDimension.OUTCOME] = EvalDimension.OUTCOME
    field_name: str = Field(min_length=1)
    pattern: str = Field(min_length=1)

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, value: str) -> str:
        pattern = reject_blank(value)
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return pattern

    @field_validator("field_name")
    @classmethod
    def _strip_field(cls, value: str) -> str:
        return reject_blank(value)


class MaxOutputTokensCheck(Check):
    """The run must not exceed a token budget.

    Guards against an agent that reaches the right answer by emitting ten times
    the necessary text - correct and unaffordable at once.
    """

    type: Literal["max_output_tokens"] = "max_output_tokens"
    dimension: Literal[EvalDimension.EFFICIENCY] = EvalDimension.EFFICIENCY
    limit: int = Field(gt=0)


class ToolsWithinCheck(Check):
    """The agent may only call tools in the allowed set.

    This is the process dimension, and it enforces HARNESS_SPEC.md section 3.3:
    behaviour is constrained by the tool schema exposed to the role, so a QA
    agent that somehow edits source code fails the eval even if its test plan
    looked fine.
    """

    type: Literal["tools_within"] = "tools_within"
    dimension: Literal[EvalDimension.PROCESS] = EvalDimension.PROCESS
    allowed_tools: tuple[str, ...] = Field(min_length=1)

    @field_validator("allowed_tools")
    @classmethod
    def _strip_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(reject_blank(item) for item in value)


AnyCheck = Annotated[
    SchemaValidCheck
    | KindCorrectCheck
    | FieldNonEmptyCheck
    | TextMatchesCheck
    | TextAbsentCheck
    | MaxOutputTokensCheck
    | ToolsWithinCheck,
    Field(discriminator="type"),
]

#: Every check type, so a loader can report an unknown ``type`` accurately.
CHECK_TYPES: tuple[str, ...] = (
    "schema_valid",
    "kind_correct",
    "field_non_empty",
    "text_matches",
    "text_absent",
    "max_output_tokens",
    "tools_within",
)


class GoldenTask(BaseModel):
    """One evaluation case for one role.

    Attributes:
        task_id: Stable identifier. Reports key on this, so renaming a task
            silently breaks trend lines; treat it as an API.
        role: The agent under evaluation.
        prompt: The input the agent receives. Written as the requester would
            write it, including its ambiguities - a prompt cleaned up to be
            unambiguous measures nothing about the role's real job.
        checks: What must hold. At least one, or the task asserts nothing.
        notes: Context for whoever tunes the task later: why this input, what
            failure it was added to catch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    role: PipelineRole
    prompt: str = Field(min_length=1)
    checks: tuple[AnyCheck, ...] = Field(min_length=1)
    notes: str = ""

    @field_validator("task_id", "prompt")
    @classmethod
    def _strip(cls, value: str) -> str:
        return reject_blank(value)

    def dimensions_covered(self) -> frozenset[EvalDimension]:
        return frozenset(check.dimension for check in self.checks)

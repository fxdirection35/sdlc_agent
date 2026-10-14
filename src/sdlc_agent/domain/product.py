"""Product-role artifacts: the requirement that starts a run and the PRD it becomes.

The PM agent turns a raw requirement into a PRD, which then hits the first
human gate. Scope boundaries and acceptance criteria are mandatory here
because HARNESS_SPEC.md section 6.1 requires in-scope / out-of-scope and the
verification criteria to be written down before work starts, not
reconstructed afterwards.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sdlc_agent.domain.base import Artifact, ArtifactKind, reject_blank


class Requirement(Artifact):
    """The raw input that kicks off a pipeline run.

    Attributes:
        title: One-line summary of what is being asked for.
        description: The requirement as the requester stated it. Kept verbatim
            so later roles can check the PRD against the original intent.
        requester: Who asked for this; used for the audit log and for routing
            the human gates back to the right person.
        source_repo: Git remote the work will land in, cloned into the sandbox.
    """

    kind: ArtifactKind = ArtifactKind.REQUIREMENT

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    requester: str = Field(min_length=1)
    source_repo: str = Field(min_length=1)

    @field_validator("title", "description", "requester", "source_repo")
    @classmethod
    def _strip(cls, value: str) -> str:
        return reject_blank(value)


class AcceptanceCriterion(BaseModel):
    """One testable statement of "done" for the PRD.

    The QA agent derives its test plan from these, so each criterion must be
    verifiable by a command an agent can run itself (HARNESS_SPEC.md
    section 5.1). A criterion nobody can check is worse than none because it
    creates the illusion of a gate.

    Attributes:
        statement: The criterion in one sentence, phrased so that pass/fail is
            unambiguous.
        verification: How to check it. Prefer an executable command over prose.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    statement: str = Field(min_length=1)
    verification: str = Field(min_length=1)

    @field_validator("statement", "verification")
    @classmethod
    def _strip(cls, value: str) -> str:
        return reject_blank(value)


class ProductRequirements(Artifact):
    """The PRD: the PM agent's output and the first human gate's input.

    Attributes:
        requirement_id: Back-pointer to the requirement this answers, so a
            reviewer can diff intent against interpretation.
        problem_statement: What hurts today, in the requester's terms.
        goals: What success looks like. Deliberately separate from
            ``non_goals`` so scope creep is visible during review.
        non_goals: Explicitly out of scope for this run.
        acceptance_criteria: The testable definition of done.
        open_questions: Unresolved points the reviewer must settle. Surfacing
            these is better than letting the PM agent guess silently.
    """

    kind: ArtifactKind = ArtifactKind.PRD

    requirement_id: uuid.UUID
    problem_statement: str = Field(min_length=1)
    goals: tuple[str, ...] = Field(min_length=1)
    non_goals: tuple[str, ...] = ()
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = Field(min_length=1)
    open_questions: tuple[str, ...] = ()

    @field_validator("problem_statement")
    @classmethod
    def _strip_problem(cls, value: str) -> str:
        return reject_blank(value)

    @field_validator("goals", "non_goals", "open_questions")
    @classmethod
    def _strip_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(reject_blank(item) for item in value)

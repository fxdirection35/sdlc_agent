"""Engineering-role artifacts: design, task breakdown and the resulting code change.

The architect agent turns an approved PRD into a design, which hits the second
human gate. The developer agent then produces a code change that lands in the
sandbox repository as a branch, ready for QA.
"""

from __future__ import annotations

import uuid

from pydantic import Field, field_validator

from sdlc_agent.domain.base import Artifact, ArtifactKind, reject_blank


class Design(Artifact):
    """The architect's output and the second human gate's input.

    Attributes:
        prd_id: Back-pointer to the PRD this design implements.
        summary: The approach in a few sentences, for reviewers who skim first.
        components: Modules or services touched, with the responsibility of
            each. Kept as free-form strings on purpose: the architect should
            not be forced into a schema the target project does not share.
        api_changes: Endpoints, function signatures or data shapes that change.
            Listed explicitly because these are the contracts QA will test.
        data_model_changes: Schema or storage changes, which are the riskiest
            part of most runs and deserve their own review focus.
        risks: What could go wrong and what mitigates it.
        alternatives_considered: Rejected approaches with the reason. Recording
            these stops a later reviewer from re-litigating a settled decision.
    """

    kind: ArtifactKind = ArtifactKind.DESIGN

    prd_id: uuid.UUID
    summary: str = Field(min_length=1)
    components: tuple[str, ...] = Field(min_length=1)
    api_changes: tuple[str, ...] = ()
    data_model_changes: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    alternatives_considered: tuple[str, ...] = ()

    @field_validator("summary")
    @classmethod
    def _strip_summary(cls, value: str) -> str:
        return reject_blank(value)

    @field_validator(
        "components",
        "api_changes",
        "data_model_changes",
        "risks",
        "alternatives_considered",
    )
    @classmethod
    def _strip_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(reject_blank(item) for item in value)


class Task(Artifact):
    """One unit of implementation work carved out of the design.

    Attributes:
        design_id: Back-pointer to the design this task implements.
        title: What to do, phrased as an instruction to the developer agent.
        acceptance_criteria: How the developer knows this task is done.
            Distinct from the PRD-level criteria: these are local to the task.
        depends_on: Tasks that must finish first. Used to order execution and
            to detect cycles before any LLM call is made.
    """

    kind: ArtifactKind = ArtifactKind.TASK_BREAKDOWN

    design_id: uuid.UUID
    title: str = Field(min_length=1, max_length=300)
    acceptance_criteria: tuple[str, ...] = ()
    depends_on: tuple[uuid.UUID, ...] = ()

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        return reject_blank(value)

    @field_validator("acceptance_criteria")
    @classmethod
    def _strip_criteria(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(reject_blank(item) for item in value)


class CodeChange(Artifact):
    """What the developer agent produced inside the sandbox.

    The change itself lives in git; this artifact records where to find it and
    what the agent believes it did. Keeping the pointer separate from the diff
    means the database index stays small while the authoritative content
    remains versioned in the repository (HARNESS_SPEC.md section 6).

    Attributes:
        task_id: The task this change implements.
        branch: Branch name pushed from the sandbox.
        commit_sha: Head commit of the branch at handoff time.
        files_changed: Paths touched, for review triage and blast-radius checks.
        summary: The agent's own account of what changed and why.
    """

    kind: ArtifactKind = ArtifactKind.CODE_CHANGE

    task_id: uuid.UUID
    branch: str = Field(min_length=1)
    commit_sha: str = Field(min_length=7, max_length=40)
    files_changed: tuple[str, ...] = ()
    summary: str = Field(min_length=1)

    @field_validator("branch", "summary")
    @classmethod
    def _strip(cls, value: str) -> str:
        return reject_blank(value)

    @field_validator("files_changed")
    @classmethod
    def _strip_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(reject_blank(item) for item in value)

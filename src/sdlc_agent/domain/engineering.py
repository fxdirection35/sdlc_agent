"""Engineering-role artifacts: design, task breakdown and the resulting code change.

The architect agent turns an approved PRD into a design, which hits the second
human gate. The developer agent then produces a code change that lands in the
sandbox repository as a branch, ready for QA.
"""

from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    kind: Literal[ArtifactKind.DESIGN] = ArtifactKind.DESIGN

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

    kind: Literal[ArtifactKind.TASK_BREAKDOWN] = ArtifactKind.TASK_BREAKDOWN

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


class FileEdit(BaseModel):
    """One file-level edit the developer role claims, before anything is written.

    The deterministic apply step turns these into a git commit; the model only
    validates the claim (HARNESS_SPEC.md: the LLM produces content, code
    controls the flow).

    Attributes:
        path: Workspace-relative POSIX path. Absolute paths, backslashes and
            ``..`` segments are rejected: an edit that can escape the workspace
            is not an edit, it is a write anywhere on the machine.
        content: The complete file content for a ``write`` - the whole file,
            not a diff, because single-shot structured output cannot be
            trusted to re-anchor hunks. Must be empty for a ``delete``.
        op: ``write`` creates or replaces the file; ``delete`` removes it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    content: str = ""
    op: Literal["write", "delete"] = "write"

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        value = reject_blank(value)
        if "\\" in value:
            raise ValueError("path must use forward slashes, not backslashes")
        if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
            raise ValueError("path must be workspace-relative, not absolute")
        if ".." in value.split("/"):
            raise ValueError("path must not traverse upward (..)")
        return value

    @model_validator(mode="after")
    def _delete_carries_no_content(self) -> FileEdit:
        """A delete edit with content is neither a delete nor a write."""
        if self.op == "delete" and self.content:
            raise ValueError("a delete edit must not carry content")
        return self


class CodeChange(Artifact):
    """What the developer agent produced inside the sandbox.

    The change itself lives in git; this artifact records where to find it and
    what the agent believes it did. Keeping the pointer separate from the diff
    means the database index stays small while the authoritative content
    remains versioned in the repository (HARNESS_SPEC.md section 6).

    Attributes:
        task_id: The task this change implements.
        branch: Branch name of the applied change. Empty until the deterministic
            apply step commits the edits and fills it in - the developer role
            cannot know a name it did not cause.
        commit_sha: Head commit of the branch. Empty until apply, for the same
            reason.
        files_changed: Paths touched, for review triage and blast-radius checks.
            The developer's claim pre-apply; the apply step cross-checks it
            against what git actually recorded.
        edits: The developer's claimed file edits. This is the payload the
            apply step writes into the workspace - the patch bridge between a
            structured-output call and a real commit.
        summary: The agent's own account of what changed and why.
        pr_url: Where the change can be reviewed once the deliver step has
            pushed the branch and opened a pull request. Empty until then.
    """

    kind: Literal[ArtifactKind.CODE_CHANGE] = ArtifactKind.CODE_CHANGE

    task_id: uuid.UUID
    branch: str = ""
    commit_sha: str = ""
    files_changed: tuple[str, ...] = ()
    edits: tuple[FileEdit, ...] = ()
    summary: str = Field(min_length=1)
    pr_url: str = ""

    @field_validator("summary")
    @classmethod
    def _strip(cls, value: str) -> str:
        return reject_blank(value)

    @field_validator("branch")
    @classmethod
    def _strip_branch(cls, value: str) -> str:
        # Empty is the honest pre-apply state; whitespace-only is a blank
        # pretending to be a value and stays rejected.
        if value == "":
            return value
        return reject_blank(value)

    @field_validator("commit_sha")
    @classmethod
    def _check_sha(cls, value: str) -> str:
        # Same 7-40 window as before, with an added empty state for the
        # not-yet-applied artifact.
        if value == "":
            return value
        if not 7 <= len(value) <= 40:
            raise ValueError("commit_sha must be empty (before apply) or 7-40 characters")
        return value

    @field_validator("files_changed")
    @classmethod
    def _strip_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(reject_blank(item) for item in value)

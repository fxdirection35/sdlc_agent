"""Plain helpers for the pipeline tests.

These are ordinary functions rather than fixtures, so tests and conftest can
both import them. Importing from ``conftest`` directly is not supported by
pytest's collection model, which is why they live here.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.types import Command

from sdlc_agent.domain import Artifact, ArtifactKind, PipelineRole
from sdlc_agent.domain.engineering import CodeChange, Design, FileEdit
from sdlc_agent.domain.product import (
    AcceptanceCriterion,
    ProductRequirements,
    Requirement,
)
from sdlc_agent.domain.qa import TestPlan, TestPlanItem, TestRun
from sdlc_agent.pipeline.codeflow import CommitResult
from sdlc_agent.pipeline.roles import RoleContext
from sdlc_agent.pipeline.state import PipelineState
from sdlc_agent.pipeline.verify import StubExecutor

#: The artifact each role is responsible for producing.
ROLE_KINDS: dict[PipelineRole, ArtifactKind] = {
    PipelineRole.PRODUCT_MANAGER: ArtifactKind.PRD,
    PipelineRole.ARCHITECT: ArtifactKind.DESIGN,
    PipelineRole.DEVELOPER: ArtifactKind.CODE_CHANGE,
    PipelineRole.QA: ArtifactKind.TEST_PLAN,
}

#: Generous bound for :func:`approve_until_terminal`; the real path needs four.
MAX_INVOKES = 12


def make_requirement() -> Requirement:
    return Requirement.model_validate(
        {
            "title": "Add CSV export to the reports page",
            "description": "Users screenshot report tables and retype them into Excel.",
            "requester": "alice",
            "source_repo": "https://github.com/example/reports.git",
        }
    )


def make_artifact(kind: ArtifactKind, context: RoleContext) -> Artifact:
    """Build a valid artifact of ``kind`` from what the role can see.

    Upstream ids come from the role's own context so back-pointers stay
    consistent with the artifacts actually in the run.
    """
    if kind is ArtifactKind.PRD:
        return ProductRequirements.model_validate(
            {
                "requirement_id": context.requirement.artifact_id,
                "problem_statement": "Report data cannot leave the app.",
                "goals": ("Users can export any filtered report as CSV",),
                "non_goals": ("Excel format support",),
                "acceptance_criteria": (
                    AcceptanceCriterion.model_validate(
                        {
                            "statement": "CSV has one header row",
                            "verification": "pytest tests/test_export.py",
                        }
                    ),
                    AcceptanceCriterion.model_validate(
                        {
                            "statement": "Commas are escaped",
                            "verification": "pytest tests/test_escaping.py",
                        }
                    ),
                ),
            }
        )
    if kind is ArtifactKind.DESIGN:
        prd = context.upstream.get(ArtifactKind.PRD)
        return Design.model_validate(
            {
                "prd_id": prd.artifact_id if prd else uuid.uuid4(),
                "summary": "Stream rows through a generator-backed CSV writer.",
                "components": ("src/reports/export.py", "src/reports/views.py"),
                "api_changes": ("GET /reports/{id}/export?format=csv",),
                "risks": ("200k rows may exceed the container memory ceiling",),
            }
        )
    if kind is ArtifactKind.CODE_CHANGE:
        return CodeChange.model_validate(
            {
                "task_id": uuid.uuid4(),
                "branch": "feat/csv-export",
                "commit_sha": "a1b2c3d4e5f6",
                "files_changed": ("src/reports/export.py", "tests/test_export.py"),
                "edits": (
                    {
                        "path": "src/reports/export.py",
                        "content": "def stream_csv(rows):\n    ...\n",
                    },
                    {
                        "path": "tests/test_export.py",
                        "content": "def test_export():\n    ...\n",
                    },
                ),
                "summary": "Added a streaming CSV writer behind the existing endpoint.",
            }
        )
    if kind is ArtifactKind.TEST_PLAN:
        change = context.upstream.get(ArtifactKind.CODE_CHANGE)
        return TestPlan.model_validate(
            {
                "prd_id": uuid.uuid4(),
                "code_change_id": change.artifact_id if change else uuid.uuid4(),
                "items": (
                    TestPlanItem.model_validate(
                        {
                            "criterion": "CSV has one header row",
                            "description": "Assert the header and one row per result",
                            "test_path": "tests/test_export.py",
                        }
                    ),
                ),
            }
        )
    raise AssertionError(f"no factory for {kind}")


def approve(comment: str = "looks good") -> Command[Any]:
    """Resume payload approving the pending gate."""
    return Command(resume={"verdict": "approved", "reviewer": "alice", "comment": comment})


def reject(comment: str = "not right yet") -> Command[Any]:
    """Resume payload rejecting the pending gate."""
    return Command(resume={"verdict": "rejected", "reviewer": "alice", "comment": comment})


def gate_name(result: dict[str, Any]) -> str:
    """The gate the run is waiting at, or '' if it is not waiting."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return ""
    return str(interrupts[0].value["gate"])


def audit_trail(result: PipelineState) -> list[str]:
    """Stage:event lines from the audit trail, for asserting the path taken."""
    return [f"{event.stage}:{event.event}" for event in result["audit"]]


def audit_details(result: PipelineState) -> list[str]:
    """The detail text of every audit entry.

    Kept separate from :func:`audit_trail` because the detail carries free text
    (rejection comments, cost figures) that a path assertion should not have to
    parse around.
    """
    return [event.detail for event in result["audit"]]


def approve_until_terminal(app: Any, thread: dict[str, Any], state: dict[str, Any]) -> Any:
    """Run the graph to completion, approving every gate it stops at.

    Returns the final state. Raises if the run does not settle, which would mean
    a loop the bounds failed to close.
    """
    result = app.invoke(state, thread)
    for _ in range(MAX_INVOKES):
        if not gate_name(result):
            return result
        result = app.invoke(approve(), thread)
    raise AssertionError(f"run did not settle within {MAX_INVOKES} invokes")


def artifact_kinds(result: PipelineState) -> set[str]:
    """Artifact kinds present in state, as plain strings.

    Keys arrive as ``str`` after a checkpoint round-trip and as ``ArtifactKind``
    within a single invoke, so normalize before comparing. Both work as lookup
    keys because ArtifactKind is a StrEnum and hashes equal to its value.
    """
    return {str(kind) for kind in result["artifacts"]}


@dataclass
class FakeRepoClient:
    """RepoClient that records calls and materializes a workspace on clone.

    Never touches a network or a real git binary: what the graph needs to know
    is which operations were requested with which arguments, not whether git
    works. The real subprocess client has its own integration tests.

    ``head`` tracks what ``head_sha`` reports - the clone base until an apply
    moves it. Tests simulating a diverged workspace (rebuilt volume, lost
    clone) set ``head`` directly after the apply.
    """

    fail_with: Exception | None = None
    clones: list[tuple[str, Path, str]] = field(default_factory=list)
    applies: list[tuple[Path, tuple[FileEdit, ...], str, str]] = field(default_factory=list)
    pushes: list[tuple[Path, str]] = field(default_factory=list)
    #: Sequence of per-apply commit shas; defaults keep the historical
    #: all-"f" sha so single-apply assertions stay byte-stable. A revision
    #: loop can hand two distinct shas to prove the deliver guard follows the
    #: *latest* verified commit.
    shas: tuple[str, ...] = ()
    head: str = "0" * 40

    def clone(self, url: str, destination: Path, *, token: str = "") -> None:
        self.clones.append((url, destination, token))
        if self.fail_with is not None:
            raise self.fail_with
        destination.mkdir(parents=True, exist_ok=True)
        (destination / ".git").mkdir()
        (destination / "src").mkdir()
        (destination / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")

    def apply_and_commit(
        self,
        root: Path,
        edits: Sequence[FileEdit],
        branch: str,
        message: str,
        *,
        token: str = "",
    ) -> CommitResult:
        self.applies.append((root, tuple(edits), branch, message))
        for edit in edits:
            target = root / edit.path
            if edit.op == "delete":
                target.unlink(missing_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edit.content, encoding="utf-8")
        index = len(self.applies) - 1
        self.head = self.shas[index] if index < len(self.shas) else "f" * 40
        return CommitResult(
            sha=self.head,
            files_changed=tuple(sorted(edit.path for edit in edits)),
        )

    def push(self, root: Path, branch: str, *, token: str = "") -> None:
        self.pushes.append((root, branch))

    def head_sha(self, root: Path, *, token: str = "") -> str:
        return self.head


@dataclass
class RecordingExecutor:
    """TestExecutor that records the context it was handed, then stubs the run.

    Pins what the verify node passes through - most importantly the workspace,
    which the sandbox executor refuses to run without.
    """

    seen: list[str] = field(default_factory=list)

    def execute(self, plan: TestPlan, context: RoleContext) -> TestRun:
        self.seen.append(context.repo_root)
        return StubExecutor().execute(plan, context)

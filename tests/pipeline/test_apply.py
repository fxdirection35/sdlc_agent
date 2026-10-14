"""Tests for the apply node: the patch bridge where claims become commits.

Node-level tests pin the artifact rebuild (id preservation, platform-filled
fields); graph-level tests pin the wiring - developer hands off to apply, QA
plans against the result, revision loops re-apply on the same branch - and
the tightenings: a real repository with no claimed edits escalates rather
than green-running an empty change.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

from sdlc_agent.domain import ArtifactKind
from sdlc_agent.domain.engineering import CodeChange
from sdlc_agent.domain.qa import TestPlan, TestRun, TestStatus
from sdlc_agent.pipeline import (
    CodeIntegrations,
    RunStatus,
    StageName,
    StubExecutor,
    build_graph,
    initial_state,
    make_apply_node,
)
from sdlc_agent.pipeline.codeflow import CommitResult
from sdlc_agent.pipeline.roles import RoleContext, ScriptedRole
from sdlc_agent.pipeline.state import PipelineState

from .helpers import (
    ROLE_KINDS,
    FakeRepoClient,
    approve,
    approve_until_terminal,
    audit_details,
    audit_trail,
    gate_name,
    make_artifact,
    make_requirement,
)

TOKEN = "ghp_sentinel_apply_0123456789"


def make_edits_change() -> CodeChange:
    return CodeChange.model_validate(
        {
            "task_id": uuid.uuid4(),
            "files_changed": (),
            "edits": ({"path": "src/new.py", "content": "x = 1\n"},),
            "summary": "claim one file",
        }
    )


def run_apply(state_overrides: dict[str, Any], code: CodeIntegrations) -> Any:
    """Invoke the apply node directly on a state with a workspace."""
    requirement = make_requirement()
    state = cast(
        PipelineState,
        {**initial_state(requirement), "repo_root": "C:/ws/run-1", **state_overrides},
    )
    node = make_apply_node(code, "apply")
    result: dict[str, Any] = node(state, {"configurable": {"thread_id": "run-1"}})
    return result


class TestApplyNode:
    def test_preserves_the_artifact_id_and_fills_the_branch_and_sha(self, tmp_path: Any) -> None:
        """The rebuild must keep the id the developer's artifact was created
        with: TestPlan.code_change_id points at it."""
        repo = FakeRepoClient()
        change = make_edits_change()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path)

        result = run_apply({"artifacts": {ArtifactKind.CODE_CHANGE: change}}, code)

        updated = result["artifacts"][ArtifactKind.CODE_CHANGE]
        assert isinstance(updated, CodeChange)
        assert updated.artifact_id == change.artifact_id
        assert updated.branch == "sdlc-agent/run-1"
        assert updated.commit_sha == "f" * 40
        assert updated.files_changed == ("src/new.py",)
        # The edits survive too: later revision loops diff against them.
        assert updated.edits == change.edits

    def test_the_token_reaches_the_client_and_nothing_else(self, tmp_path: Any) -> None:
        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, token=TOKEN)

        result = run_apply({"artifacts": {ArtifactKind.CODE_CHANGE: make_edits_change()}}, code)

        assert repo.applies[0][0] == Path("C:/ws/run-1")
        assert not any(TOKEN in detail for detail in audit_details(result))
        assert not any(TOKEN in str(value) for value in result.values())

    def test_no_edits_on_a_real_repository_escalates(self, tmp_path: Any) -> None:
        """The pre-bridge code change faked a branch name and slid through;
        a run that is supposed to change code must not green-run on nothing."""
        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path)
        empty = CodeChange.model_validate({"task_id": uuid.uuid4(), "summary": "no edits"})

        result = run_apply({"artifacts": {ArtifactKind.CODE_CHANGE: empty}}, code)

        assert result["status"] is RunStatus.ESCALATED
        assert "apply:apply_failed" in audit_trail(result)
        assert repo.applies == []

    def test_a_failed_apply_escalates(self, tmp_path: Any) -> None:
        class FailingApply(FakeRepoClient):
            def apply_and_commit(self, *args: Any, **kwargs: Any) -> CommitResult:
                raise RuntimeError("git failed (commit): nothing to commit")

        repo = FailingApply()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path)

        result = run_apply({"artifacts": {ArtifactKind.CODE_CHANGE: make_edits_change()}}, code)

        assert result["status"] is RunStatus.ESCALATED
        assert "nothing to commit" in audit_details(result)[0]

    def test_claimed_but_unchanged_paths_are_audited(self, tmp_path: Any) -> None:
        """Some claims make no actual difference; the commit still lands. The
        PR diff is the reviewer's truth - the mismatch is information."""

        class DroppingRepo(FakeRepoClient):
            def apply_and_commit(self, *args: Any, **kwargs: Any) -> CommitResult:
                root, edits, branch, message = args
                result = super().apply_and_commit(root, edits[:1], branch, message)
                return result

        repo = DroppingRepo()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path)
        two = CodeChange.model_validate(
            {
                "task_id": uuid.uuid4(),
                "edits": (
                    {"path": "src/a.py", "content": "a = 1\n"},
                    {"path": "src/b.py", "content": "b = 1\n"},
                ),
                "summary": "two claims",
            }
        )

        result = run_apply({"artifacts": {ArtifactKind.CODE_CHANGE: two}}, code)

        updated = result["artifacts"][ArtifactKind.CODE_CHANGE]
        assert isinstance(updated, CodeChange)
        assert "apply:applied" in audit_trail(result)
        assert "src/b.py" in audit_details(result)[0]


class TestApplyWiring:
    """The compiled graph: developer -> apply -> qa, and the loop back."""

    def test_without_a_workspace_apply_is_a_pass_through(
        self, make_graph: Any, config: Any, thread: Any, tmp_path: Any
    ) -> None:
        """An eval-shaped run inside a code-enabled graph: the node sits in
        the topology but changes nothing except an audit line."""
        requirement = make_requirement().model_copy(
            update={"source_repo": "eval://golden/dev_implement_writer"}
        )
        code = CodeIntegrations(repo=FakeRepoClient(), workspaces_dir=tmp_path)
        app, _ = make_graph(config, code=code)

        result = approve_until_terminal(app, thread, initial_state(requirement))

        stages = [entry.split(":")[0] for entry in audit_trail(result)]
        assert stages == [
            "checkout",
            "pm",
            "prd_review",
            "architect",
            "design_review",
            "developer",
            "apply",
            "qa",
            "verify",
            "merge_review",
            "finalize",
        ]
        assert "apply:apply_skipped" in audit_trail(result)
        assert result["status"].value == "completed"

    def test_the_change_is_committed_before_qa_plans(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any, tmp_path: Any
    ) -> None:
        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, token=TOKEN)
        app, _ = make_graph(config, code=code)

        result = approve_until_terminal(app, thread, initial_state(requirement))

        thread_id = thread["configurable"]["thread_id"]
        assert len(repo.applies) == 1
        root, edits, branch, message = repo.applies[0]
        assert root == tmp_path / thread_id
        assert branch == f"sdlc-agent/{thread_id}"
        assert {edit.path for edit in edits} == {
            "src/reports/export.py",
            "tests/test_export.py",
        }
        assert requirement.title in message

        change = result["artifacts"][ArtifactKind.CODE_CHANGE]
        assert isinstance(change, CodeChange)
        assert change.branch == f"sdlc-agent/{thread_id}"
        assert change.commit_sha == "f" * 40
        # QA planned against the applied change, not the raw claim.
        plan = result["artifacts"][ArtifactKind.TEST_PLAN]
        assert plan.code_change_id == change.artifact_id

    def test_a_revision_loop_reapplies_on_the_same_branch(
        self, config: Any, requirement: Any, thread: Any, tmp_path: Any
    ) -> None:
        """QA rejects, the developer revises, apply runs again: one branch,
        appended commits - the PR updates in place."""

        @dataclass
        class PassOnSecondTry:
            calls: int = 0

            def execute(self, plan: TestPlan, context: RoleContext) -> TestRun:
                self.calls += 1
                status = TestStatus.FAILED if self.calls == 1 else TestStatus.PASSED
                return StubExecutor(status=status).execute(plan, context)

        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path)
        roles = {
            role: ScriptedRole(role, partial(make_artifact, kind))
            for role, kind in ROLE_KINDS.items()
        }
        app = build_graph(roles, config, PassOnSecondTry(), code=code)

        result = app.invoke(initial_state(requirement), thread)
        result = app.invoke(approve(), thread)  # prd
        result = app.invoke(approve(), thread)  # design -> developer -> qa -> verify(fail)
        # -> developer -> apply -> qa -> verify(pass) -> merge gate
        assert gate_name(result) == StageName.MERGE_REVIEW.value
        result = app.invoke(approve(), thread)

        assert result["status"].value == "completed"
        thread_id = thread["configurable"]["thread_id"]
        assert len(repo.applies) == 2
        assert repo.applies[0][2] == repo.applies[1][2] == f"sdlc-agent/{thread_id}"

"""Tests for the deliver node: verified code reaches a real PR.

Node-level tests pin the guard (diverged workspace never pushes), the failure
policy (push/PR outages audit and continue - the merge gate is the safety
net) and the artifact rebuild. Graph-level tests pin the wiring and the plan's
key invariant: a merge-review rejection loop pushes each verified revision
but opens the PR exactly once.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from sdlc_agent.domain import ArtifactKind
from sdlc_agent.domain.engineering import CodeChange
from sdlc_agent.domain.product import Requirement
from sdlc_agent.domain.qa import QualityGate
from sdlc_agent.integrations import repo_slug_from_remote
from sdlc_agent.pipeline import (
    CodeIntegrations,
    RunStatus,
    StageName,
    initial_state,
    make_deliver_node,
)
from sdlc_agent.pipeline.state import PipelineState

from .helpers import (
    FakeRepoClient,
    approve,
    approve_until_terminal,
    audit_details,
    audit_trail,
    gate_name,
    make_requirement,
    reject,
)

TOKEN = "ghp_sentinel_deliver_0123456789"
PR_URL = "https://github.com/example/reports/pull/7"


@dataclass
class FakePulls:
    """PullRequestOpener double that separates POSTs from idempotent lookups.

    ``opens`` records only the calls that would have created a PR; a lookup
    finding the existing branch entry returns without appending - the same
    distinction GithubPullRequests.ensure_open makes against the real API.
    """

    url: str = PR_URL
    opens: list[tuple[str, str]] = field(default_factory=list)
    lookups: int = 0
    fail_with: Exception | None = None

    def ensure_open(self, *, repo: str, branch: str, title: str, body: str) -> str:
        self.lookups += 1
        if self.fail_with is not None:
            raise self.fail_with
        if (repo, branch) not in self.opens:
            self.opens.append((repo, branch))
        return self.url

    def slug_for(self, remote: str) -> str:
        # The real parser, not a canned answer: the deliver node's push-only
        # degradation is driven by exactly these refusals.
        return repo_slug_from_remote(remote)


def make_change(commit_sha: str = "f" * 40) -> CodeChange:
    return CodeChange.model_validate(
        {
            "task_id": uuid.uuid4(),
            "commit_sha": commit_sha,
            "branch": "sdlc-agent/run-1",
            "files_changed": ("src/new.py",),
            "edits": ({"path": "src/new.py", "content": "x = 1\n"},),
            "summary": "one file",
        }
    )


def make_gate() -> QualityGate:
    return QualityGate.model_validate(
        {"test_run_id": uuid.uuid4(), "verdict": "approved", "rationale": "suite passed"}
    )


def run_deliver(
    state_overrides: dict[str, Any],
    code: CodeIntegrations,
    *,
    requirement: Requirement | None = None,
) -> Any:
    """Invoke the deliver node directly on a state with a workspace."""
    state = cast(
        PipelineState,
        {
            **initial_state(requirement or make_requirement()),
            "repo_root": "C:/ws/run-1",
            **state_overrides,
        },
    )
    node = make_deliver_node(code, "deliver")
    result: dict[str, Any] = node(state, {"configurable": {"thread_id": "run-1"}})
    return result


class TestDeliverNode:
    def test_pushes_and_fills_pr_url_on_both_artifacts(self, tmp_path: Any) -> None:
        """The gate displays CODE_CHANGE, so the URL must live there;
        QualityGate.pr_url is filled too, honoring its docstring."""
        repo = FakeRepoClient()
        repo.head = "f" * 40  # the apply that produced this change moved HEAD
        pulls = FakePulls()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, token=TOKEN, pulls=pulls)
        change = make_change()

        result = run_deliver(
            {
                "artifacts": {
                    ArtifactKind.CODE_CHANGE: change,
                    ArtifactKind.QUALITY_GATE: make_gate(),
                }
            },
            code,
        )

        assert repo.pushes == [(Path("C:/ws/run-1"), "sdlc-agent/run-1")]
        assert pulls.opens == [("example/reports", "sdlc-agent/run-1")]
        updated = result["artifacts"][ArtifactKind.CODE_CHANGE]
        assert isinstance(updated, CodeChange)
        assert updated.pr_url == PR_URL
        # The rebuild must keep the id: TestPlan.code_change_id points at it.
        assert updated.artifact_id == change.artifact_id
        gate = result["artifacts"][ArtifactKind.QUALITY_GATE]
        assert isinstance(gate, QualityGate)
        assert gate.pr_url == PR_URL
        assert "deliver:delivered" in audit_trail(result)
        # The token reached the client and nothing the run persists.
        assert not any(TOKEN in detail for detail in audit_details(result))
        assert not any(TOKEN in str(value) for value in result.values())

    def test_a_run_without_a_workspace_skips(self, tmp_path: Any) -> None:
        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, pulls=FakePulls())

        result = run_deliver({"repo_root": ""}, code)

        assert "deliver:deliver_skipped" in audit_trail(result)
        assert repo.pushes == []

    def test_a_diverged_workspace_escalates_without_pushing(self, tmp_path: Any) -> None:
        """The volume was lost or the workspace rebuilt: whatever HEAD is now
        was never verified, and pushing it would publish unreviewed code."""
        repo = FakeRepoClient()
        repo.head = "9" * 40
        pulls = FakePulls()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, pulls=pulls)

        result = run_deliver({"artifacts": {ArtifactKind.CODE_CHANGE: make_change()}}, code)

        assert result["status"] is RunStatus.ESCALATED
        assert "deliver:deliver_diverged" in audit_trail(result)
        assert repo.pushes == []
        assert pulls.opens == []

    def test_an_unreadable_head_escalates(self, tmp_path: Any) -> None:
        class LostWorkspace(FakeRepoClient):
            def head_sha(self, root: Path, *, token: str = "") -> str:
                raise RuntimeError("git failed (rev-parse): fatal: not a git repository")

        code = CodeIntegrations(repo=LostWorkspace(), workspaces_dir=tmp_path, pulls=FakePulls())

        result = run_deliver({"artifacts": {ArtifactKind.CODE_CHANGE: make_change()}}, code)

        assert result["status"] is RunStatus.ESCALATED
        assert "deliver:deliver_diverged" in audit_trail(result)

    def test_no_code_change_escalates(self, tmp_path: Any) -> None:
        code = CodeIntegrations(repo=FakeRepoClient(), workspaces_dir=tmp_path)

        result = run_deliver({}, code)

        assert result["status"] is RunStatus.ESCALATED
        assert "deliver:deliver_failed" in audit_trail(result)

    def test_a_failed_push_audits_and_continues(self, tmp_path: Any) -> None:
        """A transient outage must not kill a suite-passed change: audit it,
        reach the gate, let delivery be re-driven. And never open a PR for a
        branch that is not on the remote."""

        class FailingPush(FakeRepoClient):
            def push(self, root: Path, branch: str, *, token: str = "") -> None:
                raise RuntimeError("git failed (push): connection reset")

        repo = FailingPush()
        repo.head = "f" * 40
        pulls = FakePulls()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, pulls=pulls)

        result = run_deliver({"artifacts": {ArtifactKind.CODE_CHANGE: make_change()}}, code)

        assert "status" not in result  # not escalated: the gate is the safety net
        assert "deliver:deliver_failed" in audit_trail(result)
        assert pulls.opens == []

    def test_a_failed_pr_call_audits_and_continues(self, tmp_path: Any) -> None:
        repo = FakeRepoClient()
        repo.head = "f" * 40
        pulls = FakePulls(fail_with=RuntimeError("GitHub POST /repos/example/reports/pulls -> 502"))
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, pulls=pulls)

        result = run_deliver({"artifacts": {ArtifactKind.CODE_CHANGE: make_change()}}, code)

        assert "status" not in result
        trail = audit_trail(result)
        assert "deliver:delivered" in trail  # the push landed
        assert "deliver:deliver_failed" in trail  # the PR call did not
        assert "artifacts" not in result  # nothing was rebuilt without a URL

    def test_a_non_github_remote_degrades_to_push_only(self, tmp_path: Any) -> None:
        """The opener refuses to serve a remote on another host: the branch is
        still delivered; the PR is simply not this pipeline's to open."""
        repo = FakeRepoClient()
        repo.head = "f" * 40
        pulls = FakePulls()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, pulls=pulls)
        requirement = make_requirement().model_copy(
            update={"source_repo": "https://gitlab.com/example/reports.git"}
        )

        result = run_deliver(
            {"artifacts": {ArtifactKind.CODE_CHANGE: make_change()}}, code, requirement=requirement
        )

        assert repo.pushes  # delivered
        assert pulls.opens == []
        assert "deliver:deliver_failed" in audit_trail(result)

    def test_without_an_opener_delivery_is_push_only(self, tmp_path: Any) -> None:
        repo = FakeRepoClient()
        repo.head = "f" * 40
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path)  # pulls=None

        result = run_deliver({"artifacts": {ArtifactKind.CODE_CHANGE: make_change()}}, code)

        assert repo.pushes == [(Path("C:/ws/run-1"), "sdlc-agent/run-1")]
        assert "deliver:delivered" in audit_trail(result)
        assert "artifacts" not in result  # no PR, so no pr_url to fill


class TestDeliverWiring:
    def test_a_verified_change_reaches_the_merge_gate_with_a_pr(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any, tmp_path: Any
    ) -> None:
        repo = FakeRepoClient()
        pulls = FakePulls()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, token=TOKEN, pulls=pulls)
        app, _ = make_graph(config, code=code)

        result = approve_until_terminal(app, thread, initial_state(requirement))

        thread_id = thread["configurable"]["thread_id"]
        assert result["status"].value == "completed"
        branch = f"sdlc-agent/{thread_id}"
        assert repo.pushes == [(tmp_path / thread_id, branch)]
        assert pulls.opens == [("example/reports", branch)]
        change = result["artifacts"][ArtifactKind.CODE_CHANGE]
        assert isinstance(change, CodeChange)
        assert change.pr_url == PR_URL
        gate = result["artifacts"][ArtifactKind.QUALITY_GATE]
        assert isinstance(gate, QualityGate)
        assert gate.pr_url == PR_URL
        assert "deliver:delivered" in audit_trail(result)
        assert not any(TOKEN in detail for detail in audit_details(result))
        assert not any(TOKEN in str(value) for value in result.values())

    def test_a_merge_rejection_loop_pushes_twice_but_opens_one_pr(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any, tmp_path: Any
    ) -> None:
        """The plan's key invariant: each verified revision is pushed (a PR
        pointing at a stale sha would show the human an old diff), yet the PR
        is opened exactly once - re-delivery finds it."""
        repo = FakeRepoClient(shas=("a" * 40, "b" * 40))
        pulls = FakePulls()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, pulls=pulls)
        app, _ = make_graph(config, code=code)

        result = app.invoke(initial_state(requirement), thread)
        assert gate_name(result) == StageName.PRD_REVIEW.value
        result = app.invoke(approve(), thread)
        assert gate_name(result) == StageName.DESIGN_REVIEW.value
        result = app.invoke(approve(), thread)  # -> developer -> qa -> verify
        # -> deliver #1 -> merge gate
        assert gate_name(result) == StageName.MERGE_REVIEW.value
        delivered = result["artifacts"][ArtifactKind.CODE_CHANGE]
        assert isinstance(delivered, CodeChange)
        assert delivered.pr_url == PR_URL
        assert delivered.commit_sha == "a" * 40

        result = app.invoke(reject("the diff is thin"), thread)  # -> developer -> apply
        # -> qa -> verify -> deliver #2 -> merge gate
        assert gate_name(result) == StageName.MERGE_REVIEW.value
        result = app.invoke(approve(), thread)

        assert result["status"].value == "completed"
        assert len(repo.pushes) == 2  # both verified revisions reached the remote
        assert len(pulls.opens) == 1  # ...but the PR was created exactly once
        assert pulls.lookups == 2
        change = result["artifacts"][ArtifactKind.CODE_CHANGE]
        assert isinstance(change, CodeChange)
        assert change.commit_sha == "b" * 40
        assert change.pr_url == PR_URL  # stable across the loop

    def test_divergence_at_delivery_escalates_from_the_graph(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any, tmp_path: Any
    ) -> None:
        class LostWorkspace(FakeRepoClient):
            def head_sha(self, root: Path, *, token: str = "") -> str:
                raise RuntimeError("git failed (rev-parse): fatal: not a git repository")

        repo = LostWorkspace()
        pulls = FakePulls()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, pulls=pulls)
        app, _ = make_graph(config, code=code)

        result = approve_until_terminal(app, thread, initial_state(requirement))

        assert result["status"] is RunStatus.ESCALATED
        assert "deliver:deliver_diverged" in audit_trail(result)
        assert repo.pushes == []
        assert pulls.opens == []

"""Graph-level tests for the checkout node's wiring.

The node's own logic is covered in test_codeflow.py by direct invocation.
What only the compiled graph can prove: the topology switch (code vs no
code), the workspace surviving into role contexts and the verify executor,
and the token staying out of everything the run persists.
"""

from __future__ import annotations

from typing import Any

from sdlc_agent.domain import PipelineRole
from sdlc_agent.pipeline import CodeIntegrations, RunStatus, StageName, initial_state

from .helpers import (
    FakeRepoClient,
    RecordingExecutor,
    approve_until_terminal,
    audit_details,
    audit_trail,
    gate_name,
    make_requirement,
)

TOKEN = "ghp_sentinel_graph_0123456789"


class TestTopologySwitch:
    def test_without_integrations_the_topology_is_unchanged(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any
    ) -> None:
        """code=None keeps the pre-code-flow topology: the very first stage is
        the PM and no checkout event ever appears."""
        app, _ = make_graph(config)

        result = app.invoke(initial_state(requirement), thread)

        assert gate_name(result) == StageName.PRD_REVIEW.value
        stages = [entry.split(":")[0] for entry in audit_trail(result)]
        assert stages[0] == StageName.PM.value
        assert StageName.CHECKOUT.value not in stages

    def test_a_synthetic_source_skips_checkout_and_still_runs(
        self, make_graph: Any, config: Any, thread: Any, tmp_path: Any
    ) -> None:
        """The eval harness feeds eval:// sources through this graph; skipping
        must be audited and the run must continue exactly as it did before."""
        requirement = make_requirement().model_copy(
            update={"source_repo": "eval://golden/dev_implement_writer"}
        )
        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path)
        app, _ = make_graph(config, code=code)

        result = app.invoke(initial_state(requirement), thread)

        assert gate_name(result) == StageName.PRD_REVIEW.value
        assert result["repo_root"] == ""
        assert f"{StageName.CHECKOUT.value}:checkout_skipped" in audit_trail(result)
        assert repo.clones == []


class TestWorkspaceFlow:
    def test_clones_into_a_workspace_named_by_the_thread(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any, tmp_path: Any
    ) -> None:
        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path, token=TOKEN)
        app, _ = make_graph(config, code=code)

        result = app.invoke(initial_state(requirement), thread)

        thread_id = thread["configurable"]["thread_id"]
        assert result["repo_root"] == str(tmp_path / thread_id)
        assert repo.clones[0][0] == requirement.source_repo
        # The token reaches the client and nothing else: no audit detail, no
        # state value, carries any part of it.
        assert repo.clones[0][2] == TOKEN
        assert not any(TOKEN in detail for detail in audit_details(result))
        assert not any(TOKEN in str(value) for value in result.values())

    def test_a_failed_clone_escalates_before_the_pm_runs(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any, tmp_path: Any
    ) -> None:
        """A run pointed at a real remote exists to change code. A failed clone
        must escalate loudly, not produce artifacts about a repo it never saw."""
        from sdlc_agent.integrations import GitCommandError

        repo = FakeRepoClient(
            fail_with=GitCommandError(["git", "clone"], 128, "fatal: repository not found")
        )
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path)
        app, roles = make_graph(config, code=code)

        result = app.invoke(initial_state(requirement), thread)

        assert result["status"] is RunStatus.ESCALATED
        assert gate_name(result) == ""
        assert f"{StageName.CHECKOUT.value}:checkout_failed" in audit_trail(result)
        assert roles[PipelineRole.PRODUCT_MANAGER].call_count == 0

    def test_the_tree_reaches_the_architect_and_developer_only(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any, tmp_path: Any
    ) -> None:
        """The tree is for roles that must name real paths; the PM works from
        the requirement and QA from the criteria and the change."""
        repo = FakeRepoClient()
        code = CodeIntegrations(repo=repo, workspaces_dir=tmp_path)
        app, roles = make_graph(config, code=code)

        result = approve_until_terminal(app, thread, initial_state(requirement))

        architect = roles[PipelineRole.ARCHITECT].calls[0]
        developer = roles[PipelineRole.DEVELOPER].calls[0]
        pm = roles[PipelineRole.PRODUCT_MANAGER].calls[0]
        qa = roles[PipelineRole.QA].calls[0]

        assert "app.py" in architect.repo_tree
        assert "src/" in developer.repo_tree
        assert pm.repo_tree == ""
        assert qa.repo_tree == ""
        assert developer.repo_root == result["repo_root"]

    def test_verify_runs_the_suite_against_the_workspace(
        self, make_graph: Any, config: Any, requirement: Any, thread: Any, tmp_path: Any
    ) -> None:
        """The executor's context carries the workspace, so the sandbox
        executor stops short-circuiting to NOT_RUN."""
        code = CodeIntegrations(repo=FakeRepoClient(), workspaces_dir=tmp_path)
        executor = RecordingExecutor()
        app, _ = make_graph(config, executor=executor, code=code)

        result = approve_until_terminal(app, thread, initial_state(requirement))

        assert executor.seen == [result["repo_root"]]
        assert result["repo_root"] != ""

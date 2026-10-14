"""Tests for the FastAPI service.

Scripted roles and a shared in-memory checkpointer, driven through a real
TestClient: these tests pin the HTTP contract - status codes, one advance per
request, error mapping - not the graph's behaviour, which tests/pipeline owns.

The in-memory factory wraps ONE saver in nullcontext per request. That is
deliberately the degenerate case of the per-request factory contract (a
Postgres factory would open a fresh connection each time); what matters here
is that the app goes through the factory at all, never around it.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any

from fastapi.testclient import TestClient

from sdlc_agent.config import PlatformConfig
from sdlc_agent.domain.base import PipelineRole
from sdlc_agent.pipeline import (
    CodeIntegrations,
    Role,
    ScriptedRole,
    StubExecutor,
    make_memory_checkpointer,
)
from sdlc_agent.service.app import create_app
from tests.pipeline.helpers import ROLE_KINDS, FakeRepoClient, make_artifact


def make_config(budget_usd: float = 5.0) -> PlatformConfig:
    return PlatformConfig.model_validate(
        {
            "providers": {"anthropic": {"api_key_env": "ANTHROPIC_API_KEY"}},
            "roles": {
                role.value: {"provider": "anthropic", "model": "claude-opus-5-5"}
                for role in PipelineRole
            },
            "budget": {"per_task_usd": budget_usd},
            "pipeline": {"max_qa_retries": 2},
        }
    )


def make_roles(**role_kwargs: Any) -> dict[PipelineRole, Role]:
    """The four scripted roles; per-role overrides as in the pipeline tests.

    Typed as the Role interface rather than ScriptedRole: create_app takes the
    interface, and a dict return type is invariant, so the factory is where
    the upcast belongs.
    """
    costs = role_kwargs.get("spent_usd") or {}
    return {
        role: ScriptedRole(
            role,
            partial(make_artifact, kind),
            spent_usd=costs.get(role, 0.0),
            tools_called=("read_file",),
            output_tokens=500,
        )
        for role, kind in ROLE_KINDS.items()
    }


def make_client(*, static_dir: Any = None, code: Any = None, **role_kwargs: Any) -> TestClient:
    saver = make_memory_checkpointer()
    app = create_app(
        roles=make_roles(**role_kwargs),
        config=make_config(),
        executor=StubExecutor(),
        checkpointer_factory=lambda: nullcontext(saver),
        static_dir=static_dir,
        code=code,
    )
    return TestClient(app)


CREATE_BODY: dict[str, str] = {
    "title": "Add CSV export",
    "description": "Users cannot get report data out.",
    "requester": "alice",
    "source_repo": "https://github.com/example/reports.git",
}


def create_run(client: TestClient, **overrides: str) -> Any:
    return client.post("/api/runs", json={**CREATE_BODY, **overrides})


def approve(client: TestClient, run_id: str) -> Any:
    """Answer the pending gate with a plain approval."""
    return client.post(f"/api/runs/{run_id}/review", json={"verdict": "approved"})


def run_to(client: TestClient, gate: str) -> str:
    """Create a run and approve gates until ``gate`` is pending."""
    response = create_run(client)
    run_id: str = response.json()["run_id"]
    while response.json()["pending"]["gate"] != gate:
        response = approve(client, run_id)
        assert response.status_code == 200, response.text
    return run_id


def gate_of(client: TestClient, run_id: str) -> str:
    """The gate a run currently waits at, via the plain run view."""
    pending = client.get(f"/api/runs/{run_id}").json()["pending"]
    return str(pending["gate"])


# -- create and read -------------------------------------------------------------


class TestCreateAndGet:
    def test_create_stops_at_the_first_gate(self) -> None:
        client = make_client()

        response = create_run(client)

        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "in_progress"
        assert body["pending"] is not None
        assert body["pending"]["kind"] == "gate"
        assert body["pending"]["gate"] == "prd_review"
        assert "approved" in body["pending"]["allowed_verdicts"]
        assert body["pending"]["budget_usd"] == 5.0
        assert body["spent_usd"] == 0.0
        assert body["requirement"]["title"] == "Add CSV export"
        assert body["audit"]  # the PM stage already recorded itself

    def test_get_returns_the_same_view(self) -> None:
        client = make_client()
        run_id = create_run(client).json()["run_id"]

        response = client.get(f"/api/runs/{run_id}")

        assert response.status_code == 200
        assert response.json()["run_id"] == run_id
        assert response.json()["pending"]["gate"] == "prd_review"

    def test_the_id_is_stable_across_requests(self) -> None:
        """The run id is the thread id; a fresh GET must not mint state."""
        client = make_client()
        created = create_run(client).json()

        fetched = client.get(f"/api/runs/{created['run_id']}").json()

        assert fetched["artifacts"] == created["artifacts"]
        assert fetched["audit"] == created["audit"]

    def test_two_runs_do_not_interfere(self) -> None:
        client = make_client()
        first = create_run(client).json()
        second = create_run(client, title="Second, unrelated ask").json()

        assert first["run_id"] != second["run_id"]
        assert gate_of(client, first["run_id"]) == "prd_review"
        approve(client, first["run_id"])
        assert gate_of(client, second["run_id"]) == "prd_review"

    def test_blank_fields_are_rejected_before_the_graph(self) -> None:
        client = make_client()

        response = create_run(client, title="   ")

        assert response.status_code == 422

    def test_unknown_fields_are_refused(self) -> None:
        client = make_client()

        response = create_run(client, priority="urgent")

        assert response.status_code == 422


# -- the review view -------------------------------------------------------------


class TestReviewView:
    def test_review_view_carries_the_artifact_under_discussion(self) -> None:
        client = make_client()
        run_id = create_run(client).json()["run_id"]

        response = client.get(f"/api/runs/{run_id}/review")

        assert response.status_code == 200
        body = response.json()
        assert body["artifact_kind"] == "prd"
        assert body["artifact"]["goals"]
        assert body["pending"]["gate"] == "prd_review"

    def test_review_view_follows_the_run_forward(self) -> None:
        client = make_client()
        run_id = run_to(client, "design_review")

        body = client.get(f"/api/runs/{run_id}/review").json()

        assert body["artifact_kind"] == "design"

    def test_review_view_refuses_when_no_gate_is_pending(self) -> None:
        client = make_client()
        run_id = run_to(client, "merge_review")
        approve(client, run_id)

        response = client.get(f"/api/runs/{run_id}/review")

        assert response.status_code == 409


# -- one advance per request -----------------------------------------------------


class TestReviewAdvance:
    def test_each_approval_advances_exactly_one_gate(self) -> None:
        client = make_client()
        run_id = create_run(client).json()["run_id"]

        assert approve(client, run_id).json()["pending"]["gate"] == "design_review"
        assert approve(client, run_id).json()["pending"]["gate"] == "merge_review"

    def test_a_full_run_completes(self) -> None:
        client = make_client()
        run_id = create_run(client).json()["run_id"]

        response = approve(client, run_id)
        response = approve(client, run_id)
        response = approve(client, run_id)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["pending"] is None
        assert set(body["artifacts"]) >= {"prd", "design", "code_change"}

    def test_a_rejection_sends_the_run_back(self) -> None:
        client = make_client()
        run_id = create_run(client).json()["run_id"]

        response = client.post(
            f"/api/runs/{run_id}/review",
            json={"verdict": "rejected", "reviewer": "bob", "comment": "goals too thin"},
        )

        assert response.status_code == 200
        assert response.json()["pending"]["gate"] == "prd_review"  # revise and return
        details = response.json()["audit"]
        assert any("rejected" in event["event"] or "revis" in event["event"] for event in details)

    def test_reviewer_and_comment_are_optional(self) -> None:
        """A bare verdict is a complete answer; the graph defaults the rest."""
        client = make_client()
        run_id = create_run(client).json()["run_id"]

        assert approve(client, run_id).json()["pending"]["gate"] == "design_review"

    def test_verdicts_are_normalized(self) -> None:
        client = make_client()
        run_id = create_run(client).json()["run_id"]

        response = client.post(f"/api/runs/{run_id}/review", json={"verdict": " APPROVED "})

        assert response.status_code == 200
        assert response.json()["pending"]["gate"] == "design_review"

    def test_an_unknown_verdict_is_400_and_the_run_survives(self) -> None:
        """The bad payload must not consume the checkpoint: a corrected retry
        still works, which is only true if the run never moved."""
        client = make_client()
        run_id = create_run(client).json()["run_id"]

        bad = client.post(f"/api/runs/{run_id}/review", json={"verdict": "maybe"})
        good = approve(client, run_id)

        assert bad.status_code == 400
        assert "verdict" in bad.json()["detail"]
        assert good.status_code == 200
        assert good.json()["pending"]["gate"] == "design_review"

    def test_reviewing_a_finished_run_conflicts(self) -> None:
        client = make_client()
        run_id = run_to(client, "merge_review")
        approve(client, run_id)

        response = approve(client, run_id)

        assert response.status_code == 409

    def test_reviewing_an_unknown_run_is_404(self) -> None:
        client = make_client()

        response = approve(client, "0" * 32)

        assert response.status_code == 404


# -- the budget gate -------------------------------------------------------------


class TestBudgetGate:
    def make_broke_client(self) -> TestClient:
        """PM and architect each cost 3 of a 5 ceiling: the developer node
        finds 6 spent against 5 allowed and stops at the budget gate."""
        return make_client(
            spent_usd={PipelineRole.PRODUCT_MANAGER: 3.0, PipelineRole.ARCHITECT: 3.0}
        )

    def test_the_budget_interrupt_is_visible_as_a_pending_decision(self) -> None:
        client = self.make_broke_client()
        run_id = create_run(client).json()["run_id"]
        approve(client, run_id)
        response = approve(client, run_id)

        assert response.status_code == 200
        pending = response.json()["pending"]
        assert pending["kind"] == "budget"
        assert pending["gate"] == "budget_review"
        assert pending["blocked_stage"] == "developer"
        assert pending["budget_usd"] is None

    def test_a_grant_resumes_the_blocked_stage(self) -> None:
        client = self.make_broke_client()
        run_id = create_run(client).json()["run_id"]
        approve(client, run_id)
        approve(client, run_id)

        response = client.post(f"/api/runs/{run_id}/budget", json={"grant_usd": 10.0})

        assert response.status_code == 200
        assert response.json()["status"] == "in_progress"
        assert response.json()["pending"]["gate"] == "merge_review"
        assert response.json()["budget_usd"] == 10.0

    def test_no_grant_ends_the_run_honestly(self) -> None:
        client = self.make_broke_client()
        run_id = create_run(client).json()["run_id"]
        approve(client, run_id)
        approve(client, run_id)

        response = client.post(f"/api/runs/{run_id}/budget", json={"grant_usd": 0.0})

        assert response.status_code == 200
        assert response.json()["status"] == "budget_exceeded"
        assert response.json()["pending"] is None

    def test_budget_answers_only_a_budget_pause(self) -> None:
        client = self.make_broke_client()
        run_id = create_run(client).json()["run_id"]

        response = client.post(f"/api/runs/{run_id}/budget", json={"grant_usd": 5.0})

        assert response.status_code == 409

    def test_review_answers_only_a_gate_pause(self) -> None:
        client = self.make_broke_client()
        run_id = create_run(client).json()["run_id"]
        approve(client, run_id)
        approve(client, run_id)

        review = approve(client, run_id)
        view = client.get(f"/api/runs/{run_id}/review")

        assert review.status_code == 409
        assert view.status_code == 409


# -- the run list -----------------------------------------------------------------


class TestListRuns:
    def test_lists_every_run_with_its_pending_gate(self) -> None:
        client = make_client()
        first = create_run(client).json()
        second = create_run(client, title="A second ask").json()

        response = client.get("/api/runs")

        assert response.status_code == 200
        by_id = {item["run_id"]: item for item in response.json()}
        assert set(by_id) == {first["run_id"], second["run_id"]}
        row = by_id[first["run_id"]]
        assert row["title"] == "Add CSV export"
        assert row["requester"] == "alice"
        assert row["status"] == "in_progress"
        assert row["pending_kind"] == "gate"
        assert row["pending_gate"] == "prd_review"
        assert row["budget_usd"] == 5.0
        assert row["updated_at"]  # the sort key is present

    def test_the_list_follows_a_run_forward(self) -> None:
        client = make_client()
        run_id = create_run(client).json()["run_id"]

        approve(client, run_id)
        row = {item["run_id"]: item for item in client.get("/api/runs").json()}[run_id]

        assert row["pending_gate"] == "design_review"

    def test_the_newest_run_is_first(self) -> None:
        client = make_client()
        create_run(client, title="older")
        create_run(client, title="newer")

        response = client.get("/api/runs")

        titles = [item["title"] for item in response.json()]
        assert titles.index("newer") < titles.index("older")

    def test_a_service_with_no_runs_lists_nothing(self) -> None:
        client = make_client()

        assert client.get("/api/runs").json() == []


# -- the SPA mount ---------------------------------------------------------------


class TestStaticDir:
    def test_the_built_spa_is_served_at_the_root(self, tmp_path: Any) -> None:
        (tmp_path / "index.html").write_text(
            "<html><body>spa-marker</body></html>", encoding="utf-8"
        )
        client = make_client(static_dir=tmp_path)

        page = client.get("/")
        health = client.get("/api/health")
        missing = client.get("/no-such-asset.js")

        assert page.status_code == 200
        assert "spa-marker" in page.text
        # API routes registered before the mount keep winning.
        assert health.status_code == 200
        assert missing.status_code == 404

    def test_without_a_static_dir_the_root_is_not_a_page(self) -> None:
        client = make_client()

        assert client.get("/").status_code == 404


# -- 404 and health ---------------------------------------------------------------


class TestUnknownRun:
    def test_every_endpoint_404s_on_an_unknown_id(self) -> None:
        client = make_client()
        missing = "0" * 32

        assert client.get(f"/api/runs/{missing}").status_code == 404
        assert client.get(f"/api/runs/{missing}/review").status_code == 404
        assert (
            client.post(f"/api/runs/{missing}/review", json={"verdict": "approved"}).status_code
            == 404
        )
        assert client.post(f"/api/runs/{missing}/budget", json={"grant_usd": 1}).status_code == 404

    def test_health_needs_nothing(self) -> None:
        client = make_client()

        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# -- code flow --------------------------------------------------------------------


class TestCodeFlowThroughHttp:
    """The bundle passed to create_app decides the topology the service runs:
    with one, the very first request grounds the run in a workspace; without
    it, the pre-code-flow shape is served unchanged."""

    def test_a_run_with_the_bundle_checks_out_before_the_pm(self, tmp_path: Any) -> None:
        repo = FakeRepoClient()
        client = make_client(code=CodeIntegrations(repo=repo, workspaces_dir=tmp_path))

        response = create_run(client)

        assert response.status_code == 201
        trail = [f"{entry['stage']}:{entry['event']}" for entry in response.json()["audit"]]
        assert trail[0] == "checkout:checkout_cloned"
        assert len(repo.clones) == 1

    def test_without_the_bundle_no_checkout_stage_ever_runs(self) -> None:
        client = make_client()

        response = create_run(client)

        trail = [entry["stage"] for entry in response.json()["audit"]]
        assert "checkout" not in trail

"""The FastAPI service: HTTP around the graph, and nothing else.

One request advances the run exactly once - create runs to the first
interrupt, each review or budget decision runs to the next one - because the
human gates are the only places the flow legitimately stops. That one-invoke
rule is what keeps the API honest: a request that advanced twice would answer
a gate nobody had seen yet.

The checkpointer is manufactured per request through a context-manager
factory. A Postgres checkpointer owns a psycopg connection, and a connection
is not safe to share across the threadpool threads FastAPI runs sync
endpoints on - so no saver is ever shared. The compiled graph is rebuilt per
request too, which sounds wasteful and is not: compilation is pure topology
(no I/O, no model), and it means the saver and the graph that uses it have
exactly the same lifetime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

import sdlc_agent
from sdlc_agent.config import PlatformConfig
from sdlc_agent.domain.base import PipelineRole
from sdlc_agent.domain.product import Requirement
from sdlc_agent.pipeline import (
    GATE_ARTIFACT,
    GraphConstructionError,
    Role,
    StageName,
    TestExecutor,
    build_graph,
    initial_state,
)
from sdlc_agent.pipeline.state import RunStatus
from sdlc_agent.service.models import (
    BUDGET_PAUSE,
    GATE_PAUSE,
    AuditEventOut,
    BudgetDecisionRequest,
    PendingDecision,
    RequirementOut,
    ReviewRequest,
    ReviewView,
    RunCreateRequest,
    RunResponse,
    RunSummary,
)

#: What a factory produces: a context manager owning one checkpointer. The
#: saver's channel type is left ``Any`` on purpose: implementations differ
#: (InMemorySaver, PostgresSaver) and no code here reads or writes a channel.
CheckpointerFactory = Callable[[], AbstractContextManager[BaseCheckpointSaver[Any]]]

#: Environment variables the deployment assembly reads (service.main). They
#: live here - the import-safe module - rather than there, because main builds
#: its app at import time and there is no config-file entry for them: the
#: first two are credentials (HARNESS_SPEC.md 4.1), the third is a path that
#: exists only in a built deployment.
DATABASE_URL_ENV = "SDLC_DATABASE_URL"
CONFIG_PATH_ENV = "SDLC_CONFIG_PATH"
STATIC_DIR_ENV = "SDLC_STATIC_DIR"

#: Bound on the run-list scan. Generous for an internal tool; exists so a
#: checkpoint store that grows without cleanup cannot turn the list endpoint
#: into an unbounded scan.
_LIST_LIMIT = 500


def _ceiling(values: Mapping[str, Any], config: PlatformConfig) -> float:
    """Effective budget for the run; mirrors the graph's own rule."""
    override = values.get("budget_override_usd", 0.0)
    return override if override > 0 else config.budget.per_task_usd


def _pending_from(snapshot: Any) -> PendingDecision | None:
    """The decision the run waits on, or ``None`` when it is not waiting.

    Distinguished by shape rather than by gate name: a gate payload carries
    ``allowed_verdicts``, the budget payload carries ``blocked_stage``. Both
    are Mapping values inside the snapshot's task interrupts; a payload that
    is neither (a future pause kind) is skipped rather than guessed at, so an
    unknown pause surfaces as "no pending decision" and the client asks the
    run view - loud, not silently misrouted to the wrong endpoint.
    """
    for task in snapshot.tasks:
        for interrupt in task.interrupts:
            value = interrupt.value
            if not isinstance(value, Mapping):
                continue
            if "allowed_verdicts" in value:
                return PendingDecision(
                    kind=GATE_PAUSE,
                    gate=str(value["gate"]),
                    prompt=str(value["prompt"]),
                    spent_usd=float(value.get("spent_usd", 0.0)),
                    budget_usd=float(value["budget_usd"]),
                    allowed_verdicts=tuple(str(v) for v in value["allowed_verdicts"]),
                )
            if "blocked_stage" in value:
                return PendingDecision(
                    kind=BUDGET_PAUSE,
                    gate=str(value["gate"]),
                    prompt=str(value["prompt"]),
                    spent_usd=float(value.get("spent_usd", 0.0)),
                    budget_usd=None,
                    blocked_stage=str(value["blocked_stage"]),
                )
    return None


def _run_response(
    run_id: str, values: Mapping[str, Any], config: PlatformConfig, snapshot: Any
) -> RunResponse:
    """Project one snapshot into the wire shape.

    ``values`` keys arrive as enums inside a live invoke and as strings after
    a checkpoint round trip; ``str()`` normalizes both because every enum in
    state is a StrEnum.
    """
    requirement = values.get("requirement")
    if requirement is None:
        # A run exists only because start() invoked under its id, and the
        # requirement is the immutable seed of that state. Values without one
        # are corrupt checkpoints, and presenting them as a run would be worse
        # than refusing.
        raise HTTPException(status_code=500, detail="checkpoint has no requirement")
    artifacts = values.get("artifacts", {})
    audit = values.get("audit", ())
    status = values.get("status", RunStatus.IN_PROGRESS)
    return RunResponse(
        run_id=run_id,
        status=str(status),
        pending=_pending_from(snapshot),
        spent_usd=float(values.get("spent_usd", 0.0)),
        budget_usd=_ceiling(values, config),
        requirement=RequirementOut(
            title=requirement.title,
            description=requirement.description,
            requester=requirement.requester,
            source_repo=requirement.source_repo,
        ),
        artifacts={str(kind): str(artifact.artifact_id) for kind, artifact in artifacts.items()},
        audit=tuple(
            AuditEventOut(
                stage=event.stage,
                event=event.event,
                detail=event.detail,
                at=event.at.isoformat(),
            )
            for event in audit
        ),
    )


class _Runs:
    """The dependencies every route needs, in one place.

    A class rather than module globals so tests build isolated apps with
    their own roles, config and checkpointer factory - no shared mutable
    state between test cases.
    """

    def __init__(
        self,
        roles: dict[PipelineRole, Role],
        config: PlatformConfig,
        executor: TestExecutor,
        checkpointer_factory: CheckpointerFactory,
    ) -> None:
        self._roles = roles
        self._config = config
        self._executor = executor
        self._checkpointer_factory = checkpointer_factory

    def _graph_with(self, saver: BaseCheckpointSaver[Any]) -> Any:
        return build_graph(self._roles, self._config, self._executor, checkpointer=saver)

    def start(self, request: RunCreateRequest) -> RunResponse:
        """Mint a run id and run the pipeline to its first stop.

        The id is the LangGraph thread id: one namespace for checkpoints,
        resume payloads and the URL the client polls, with no secondary
        mapping to keep consistent.
        """

        run_id = uuid4().hex
        requirement = Requirement.model_validate(request.model_dump())
        with self._checkpointer_factory() as saver:
            graph = self._graph_with(saver)
            thread = {"configurable": {"thread_id": run_id}}
            graph.invoke(initial_state(requirement), thread)
            snapshot = graph.get_state(thread)
        return _run_response(run_id, snapshot.values, self._config, snapshot)

    def list_runs(self) -> list[RunSummary]:
        """Every run in the checkpoint store, newest first.

        ``saver.list(None)`` enumerates checkpoint tuples across all threads
        (both in-memory and Postgres savers support it); the tuples are used
        only for their thread ids, because a full projection needs the same
        deserializing ``get_state`` path the single-run endpoint uses - going
        through it keeps list and detail from disagreeing about a run. A
        thread whose state is empty is skipped: it is either mid-creation (the
        first checkpoint not yet visible) or corrupt, and listing a ghost run
        that 404s on click would be a lie with extra steps.
        """
        summaries: list[RunSummary] = []
        with self._checkpointer_factory() as saver:
            graph = self._graph_with(saver)
            thread_ids = {
                str(item.config["configurable"]["thread_id"])
                for item in saver.list(None, limit=_LIST_LIMIT)
            }
            for run_id in thread_ids:
                snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
                if snapshot.created_at is None or not snapshot.values:
                    continue
                requirement = snapshot.values.get("requirement")
                if requirement is None:
                    continue
                pending = _pending_from(snapshot)
                summaries.append(
                    RunSummary(
                        run_id=run_id,
                        status=str(snapshot.values.get("status", RunStatus.IN_PROGRESS)),
                        pending_kind=pending.kind if pending else "",
                        pending_gate=pending.gate if pending else "",
                        spent_usd=float(snapshot.values.get("spent_usd", 0.0)),
                        budget_usd=_ceiling(snapshot.values, self._config),
                        title=requirement.title,
                        requester=requirement.requester,
                        updated_at=str(snapshot.created_at or ""),
                    )
                )
        # ISO timestamps sort lexicographically; created_at is either "" (which
        # sorts last) or an ISO string from the checkpoint store.
        summaries.sort(key=lambda item: item.updated_at, reverse=True)
        return summaries

    def load(self, run_id: str) -> tuple[Any, Any]:
        """Snapshot a run; raises 404 for an id this service never created.

        LangGraph's ``get_state`` returns an empty snapshot for an unknown
        thread rather than ``None`` (measured), so the check is on the
        snapshot being genuinely empty: a real run always has a checkpoint
        with values, because the only way an id exists is that ``start``
        invoked under it.
        """
        with self._checkpointer_factory() as saver:
            graph = self._graph_with(saver)
            snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
        if snapshot.created_at is None or not snapshot.values:
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
        return snapshot, snapshot.values

    def resume(self, run_id: str, payload: Mapping[str, Any]) -> RunResponse:
        """Advance a stopped run by exactly one decision."""
        snapshot, _ = self.load(run_id)
        pending = _pending_from(snapshot)
        if pending is None:
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} is not waiting for a decision (status "
                f"{snapshot.values.get('status', '?')})",
            )
        with self._checkpointer_factory() as saver:
            graph = self._graph_with(saver)
            thread = {"configurable": {"thread_id": run_id}}
            try:
                graph.invoke(Command(resume=dict(payload)), thread)
            except GraphConstructionError as exc:
                # The payload could not become a decision. The run is still
                # stopped where it was - the checkpoint survived the failed
                # resume - so the client may correct and retry.
                raise HTTPException(status_code=400, detail=str(exc)) from None
            after = graph.get_state(thread)
        return _run_response(run_id, after.values, self._config, after)


def create_app(
    roles: dict[PipelineRole, Role],
    config: PlatformConfig,
    executor: TestExecutor,
    checkpointer_factory: CheckpointerFactory,
    static_dir: Path | None = None,
) -> FastAPI:
    """Assemble the service.

    Everything injectable is a parameter: the roles (scripted in tests,
    litellm in deployment), the executor (sandbox extra in deployment), the
    checkpointer factory (in-memory in tests, Postgres in deployment), and
    the built SPA's directory (None in tests and API-only deployments).
    ``create_app_from_env`` in :mod:`sdlc_agent.service.main` is the deployed
    assembly; this function is the tested one.
    """
    runs = _Runs(roles, config, executor, checkpointer_factory)
    app = FastAPI(title="sdlc-agent", version=sdlc_agent.__version__)

    @app.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": sdlc_agent.__version__})

    @app.get("/api/runs")
    def list_runs() -> list[RunSummary]:
        return runs.list_runs()

    @app.post("/api/runs", status_code=201)
    def create_run(request: RunCreateRequest) -> RunResponse:
        return runs.start(request)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> RunResponse:
        snapshot, values = runs.load(run_id)
        return _run_response(run_id, values, config, snapshot)

    @app.get("/api/runs/{run_id}/review")
    def get_review(run_id: str) -> ReviewView:
        """The pending gate with the artifact under discussion attached."""
        snapshot, values = runs.load(run_id)
        pending = _pending_from(snapshot)
        if pending is None or pending.kind != GATE_PAUSE:
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} has no gate waiting for review",
            )
        kind = GATE_ARTIFACT[StageName(pending.gate)]
        artifact = values.get("artifacts", {}).get(kind)
        if artifact is None:
            # The gate cannot have fired without its artifact in state; if
            # this ever fires, state and interrupt disagree and a human should
            # see that, not a review view missing its subject.
            raise HTTPException(
                status_code=500,
                detail=f"gate {pending.gate} is pending but no {kind.value} artifact exists",
            )
        return ReviewView(
            pending=pending,
            artifact_kind=kind.value,
            artifact=artifact.model_dump(mode="json"),
        )

    @app.post("/api/runs/{run_id}/review")
    def post_review(run_id: str, request: ReviewRequest) -> RunResponse:
        """Answer the pending gate; the run advances to its next stop."""
        snapshot, _ = runs.load(run_id)
        pending = _pending_from(snapshot)
        if pending is None or pending.kind != GATE_PAUSE:
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} has no gate waiting for review",
            )
        # The verdict is validated HERE, before any invoke, because a bad
        # resume value poisons the thread permanently: LangGraph persists the
        # resume payload with the pending task and re-delivers it to the
        # re-executed node, so a verdict the graph rejects once is re-jected
        # on every later resume - the run can never leave that gate (measured;
        # see test_an_unknown_verdict_is_400_and_the_run_survives). Rejecting
        # at the boundary never touches the checkpoint, so the client may
        # correct and retry. The graph's own GraphConstructionError remains
        # mapped to 400 below as a backstop, not as the first line.
        unknown = (
            f"gate {pending.gate} got verdict {request.verdict!r}; expected one of: "
            f"{', '.join(pending.allowed_verdicts)}"
        )
        if request.verdict not in pending.allowed_verdicts:
            raise HTTPException(status_code=400, detail=unknown)
        # Optional fields are omitted rather than sent blank: the graph
        # defaults a missing reviewer to "unspecified", and an empty string
        # would overwrite that default with nothing.
        payload: dict[str, Any] = {"verdict": request.verdict}
        if request.reviewer:
            payload["reviewer"] = request.reviewer
        if request.comment:
            payload["comment"] = request.comment
        return runs.resume(run_id, payload)

    @app.post("/api/runs/{run_id}/budget")
    def post_budget(run_id: str, request: BudgetDecisionRequest) -> RunResponse:
        """Grant more budget (run continues) or confirm the stop (run ends)."""
        snapshot, _ = runs.load(run_id)
        pending = _pending_from(snapshot)
        if pending is None or pending.kind != BUDGET_PAUSE:
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} is not waiting for a budget decision",
            )
        return runs.resume(run_id, {"grant_usd": request.grant_usd})

    # The SPA mount goes LAST so every /api route above wins: Starlette matches
    # routes in registration order, and a mount on "/" would otherwise swallow
    # paths the API has opinions about. html=True serves index.html at "/".
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="spa")

    return app


__all__ = [
    "CONFIG_PATH_ENV",
    "DATABASE_URL_ENV",
    "STATIC_DIR_ENV",
    "CheckpointerFactory",
    "create_app",
]

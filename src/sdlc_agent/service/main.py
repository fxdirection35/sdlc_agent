"""Deployable assembly: environment in, running app out.

``create_app`` takes everything as parameters and is what the tests exercise;
this module reads the environment once and wires the real implementations
together. Keeping them apart is what lets the tested composition stay
parameterized - a factory that reads ``os.environ`` cannot be unit-tested
without mutating global state.

Run with::

    SDLC_DATABASE_URL=postgres://... uvicorn sdlc_agent.service.main:app

Credentials come from the environment only (HARNESS_SPEC.md section 4.1): the
config file names key-bearing variables, the database URL never enters a file,
and loading this module requires no secret at all.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path

from fastapi import FastAPI

from sdlc_agent.config import MissingCredentialError, PlatformConfig, load_config
from sdlc_agent.domain.base import PipelineRole
from sdlc_agent.pipeline import TestExecutor, make_memory_checkpointer
from sdlc_agent.pipeline.llm_roles import LitellmRole
from sdlc_agent.pipeline.postgres import postgres_checkpointer
from sdlc_agent.service.app import (
    CONFIG_PATH_ENV,
    DATABASE_URL_ENV,
    STATIC_DIR_ENV,
    CheckpointerFactory,
    create_app,
)

DEFAULT_CONFIG_PATH = Path("config.yaml")

logger = logging.getLogger(__name__)


def checkpointer_from_env() -> tuple[CheckpointerFactory, str]:
    """The per-request checkpointer factory and a human-readable description.

    With a database URL: a fresh Postgres connection per request, so human
    gates survive a restart. Without one: one shared in-memory saver, logged
    loudly - a mode in which a restart drops every pending gate on the floor,
    acceptable only for a single-user development session.
    """
    url = os.environ.get(DATABASE_URL_ENV, "")
    if url:
        return (lambda: postgres_checkpointer(url)), f"postgres ({DATABASE_URL_ENV})"
    saver = make_memory_checkpointer()
    return (lambda: nullcontext(saver)), "in-memory (gates lost on restart)"


def require_startup_credentials(config: PlatformConfig) -> None:
    """Refuse to serve without every provider's credentials.

    A service that boots keyless would pass /api/health and then fail every
    run: green-but-broken, the false positive this project refuses elsewhere.
    The check is at boot only - a process's environment is frozen after
    startup, so a credential that existed at boot exists for every request,
    and the graph's RoleError path stays reserved for genuine provider
    trouble (a missing key escaping the graph is pinned as a loud failure in
    tests/pipeline/test_llm_roles.py and stays that way).

    Raises with ALL missing variables named, so an operator fixes the
    deployment in one restart instead of one per role.
    """
    missing: list[str] = []
    for role in PipelineRole:
        try:
            config.resolve_api_key(role)
        except MissingCredentialError as exc:
            missing.append(str(exc))
    if missing:
        raise RuntimeError(
            "refusing to start: "
            + "; ".join(missing)
            + f" (set them before launching; see {CONFIG_PATH_ENV} config)"
        )


def executor_from_env() -> TestExecutor:
    """The sandbox executor; needs the ``sandbox`` extra (see CLAUDE.md).

    Imported inside the function so this module - and therefore the service -
    starts even where only the core install is present, and only a deployment
    that actually reaches the verify stage pays for the OpenHands SDK.
    """
    from sdlc_agent.sandbox import SandboxExecutor

    return SandboxExecutor()


def create_app_from_env() -> FastAPI:
    """Build the deployed app: config file, litellm roles, sandbox executor."""
    config_path = Path(os.environ.get(CONFIG_PATH_ENV, str(DEFAULT_CONFIG_PATH)))
    config = load_config(config_path)
    require_startup_credentials(config)
    factory, storage = checkpointer_from_env()
    if "in-memory" in storage:
        logger.warning(
            "%s is not set; checkpoint storage is %s - pending gates will not survive a restart",
            DATABASE_URL_ENV,
            storage,
        )
    else:
        logger.info("checkpoint storage is %s", storage)
    static_dir = static_dir_from_env()
    return create_app(
        roles={role: LitellmRole(role, config) for role in PipelineRole},
        config=config,
        executor=executor_from_env(),
        checkpointer_factory=factory,
        static_dir=static_dir,
    )


def static_dir_from_env() -> Path | None:
    """The built SPA directory, or None when unset or not yet built.

    A build that has not happened yet (no dist) is a warning, not a failure:
    the service still serves the API, and the SPA can be built later. A
    configured path that exists but is not a directory is a real mistake and
    fails startup instead of serving from a file path by accident.
    """
    raw = os.environ.get(STATIC_DIR_ENV, "")
    if not raw:
        return None
    path = Path(raw)
    if path.is_dir():
        return path
    logger.warning("%s=%s is not a built SPA directory; serving API only", STATIC_DIR_ENV, raw)
    return None


#: The uvicorn target: ``uvicorn sdlc_agent.service.main:app``. Built at import
#: time by design - that is how uvicorn finds it - which is also why every
#: failure here (missing config file, missing extra) is a startup failure
#: rather than a surprise mid-request.
app = create_app_from_env()

__all__ = [
    "CONFIG_PATH_ENV",
    "DATABASE_URL_ENV",
    "app",
    "checkpointer_from_env",
    "create_app_from_env",
    "require_startup_credentials",
    "static_dir_from_env",
]

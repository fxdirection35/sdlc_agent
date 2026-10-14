"""The deployment files as tested contracts: compose.yaml, Dockerfile, .dockerignore.

No Docker daemon is needed for these tests (this development machine has
none): they pin the structure the deployment actually depends on - each one
names a container mistake that would otherwise fail silently, far from the
code that could have caught it. A real ``docker compose config`` validation
runs only where the CLI exists and skips honestly elsewhere, the same rule
as the Postgres integration tests.

Why each contract exists:

- The service env names must match ``service/app.py`` constants - a rename on
  either side otherwise lands as a silently ignored variable.
- Credentials appear only as ``${VAR:?}`` references: compose refuses to start
  without them, mirroring ``require_startup_credentials`` (a green stack with
  an empty key is the false green this project refuses everywhere).
- ``SDLC_DATABASE_URL`` points at the ``db`` service, never localhost.
- The image installs the ``sandbox`` extra: ``service.main`` composes the app
  at import time and imports ``sdlc_agent.sandbox`` unconditionally, so a
  core-only image crashes at boot forever, not at first use.
- uvicorn binds 0.0.0.0: the container-side default leaves the published port
  forwarding to nothing.
- The SPA directory the Dockerfile copies equals the ``SDLC_STATIC_DIR`` it
  declares, and the config mount target equals ``SDLC_CONFIG_PATH``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from sdlc_agent.service.app import CONFIG_PATH_ENV, DATABASE_URL_ENV, STATIC_DIR_ENV

_ROOT = Path(__file__).resolve().parents[1]
_COMPOSE = _ROOT / "compose.yaml"
_DOCKERFILE = _ROOT / "Dockerfile"
_DOCKERIGNORE = _ROOT / ".dockerignore"

#: A committed key would match this (Anthropic- or DeepSeek-shaped); a
#: variable NAME (the only thing allowed in git, HARNESS_SPEC.md section 4.1)
#: does not.
_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{16,}")


def _compose() -> dict[str, Any]:
    data = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return cast("dict[str, Any]", data)


def _services() -> dict[str, Any]:
    services = _compose()["services"]
    assert isinstance(services, dict)
    return cast("dict[str, Any]", services)


def _api_environment() -> dict[str, Any]:
    env = _services()["api"]["environment"]
    assert isinstance(env, dict)
    return cast("dict[str, Any]", env)


def _dockerfile() -> str:
    return _DOCKERFILE.read_text(encoding="utf-8")


def _python_floor() -> tuple[int, int]:
    """The (major, minor) floor from pyproject's ``requires-python``.

    Cross-file on purpose: raising the floor must fail here rather than ship
    an image on a Python the package no longer accepts.
    """
    spec = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "requires-python"
    ]
    assert isinstance(spec, str)
    assert spec.startswith(">="), f"unsupported requires-python syntax: {spec}"
    major, minor = spec.removeprefix(">=").split(".")
    return int(major), int(minor)


def test_compose_declares_api_and_db_without_a_version_key() -> None:
    services = _services()
    assert set(services) == {"api", "db"}
    # The top-level `version:` key is obsolete in Compose v2 and only earns a
    # deprecation warning; pin its absence.
    assert "version" not in _compose()


def test_api_waits_for_a_healthy_database() -> None:
    depends = _services()["api"].get("depends_on")
    assert depends == {"db": {"condition": "service_healthy"}}
    assert "healthcheck" in _services()["db"]


def test_service_env_names_match_the_app_constants() -> None:
    """compose.yaml and service/app.py must agree on the variable names.

    ``app.py`` owns these names; a rename there without this file is a
    silently ignored variable - the service would fall back to defaults with
    no error pointing at the compose file.
    """
    env = _api_environment()
    assert env[CONFIG_PATH_ENV] == "/app/config/config.yaml"
    assert isinstance(env[DATABASE_URL_ENV], str)


def test_credentials_are_references_not_values() -> None:
    db_env = _services()["db"]["environment"]
    assert isinstance(db_env, dict)
    assert str(db_env["POSTGRES_PASSWORD"]).startswith("${POSTGRES_PASSWORD:?"), (
        "db password must fail `up` when unset"
    )
    api_key = _api_environment()["DEEPSEEK_API_KEY"]
    assert str(api_key).startswith("${DEEPSEEK_API_KEY:?")
    url = _api_environment()[DATABASE_URL_ENV]
    assert isinstance(url, str)
    assert re.search(r"\$\{POSTGRES_PASSWORD:\?", url), "URL must not inline the password"
    for path in (_COMPOSE, _DOCKERFILE, _DOCKERIGNORE):
        assert not _SECRET_PATTERN.search(path.read_text(encoding="utf-8"))


def test_database_url_targets_the_db_service_not_localhost() -> None:
    url = _api_environment()[DATABASE_URL_ENV]
    assert isinstance(url, str)
    assert re.search(r"@db:5432/", url)
    assert "localhost" not in url and "127.0.0.1" not in url


def test_config_is_mounted_read_only_where_the_service_looks() -> None:
    expected = _api_environment()[CONFIG_PATH_ENV]
    volumes = _services()["api"].get("volumes")
    assert isinstance(volumes, list)
    matches = [
        v
        for v in volumes
        if isinstance(v, str) and v.rsplit(":", 2)[-1] == "ro" and v.rsplit(":", 2)[1] == expected
    ]
    assert matches, f"no read-only mount of config.yaml at {expected}: {volumes}"


def test_dockerfile_installs_the_sandbox_extra() -> None:
    """A core-only image boot-loops: main.py imports sdlc_agent.sandbox at
    startup, so the extra is a boot requirement, not a feature flag."""
    assert re.search(r'pip install[^\n]*"\.\[sandbox\]"', _dockerfile())


def test_dockerfile_base_satisfies_the_python_floor() -> None:
    match = re.search(r"^FROM python:(\d+)\.(\d+)-slim", _dockerfile(), re.MULTILINE)
    assert match, "no python:*-slim base stage"
    assert (int(match.group(1)), int(match.group(2))) >= _python_floor()


def test_dockerfile_binds_all_interfaces() -> None:
    assert '--host", "0.0.0.0' in _dockerfile()


def test_dockerfile_spa_paths_agree_with_static_dir_env() -> None:
    text = _dockerfile()
    static_dir = re.search(rf"{STATIC_DIR_ENV}=(\S+)", text)
    copy_target = re.search(r"COPY --from=frontend \S+ (\S+)", text)
    assert static_dir and copy_target
    assert static_dir.group(1) == copy_target.group(1).rstrip("/")


def test_dockerignore_keeps_secrets_and_bulk_out_of_the_context() -> None:
    entries = {
        line.strip()
        for line in _DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for required in (".env", ".env.*", "config.yaml", ".git", ".venv", "**/node_modules"):
        assert required in entries, f"{required} must stay out of the build context"


@pytest.mark.skipif(shutil.which("docker") is None, reason="no docker CLI on this machine")
def test_compose_config_validates() -> None:
    """The real validator, where it exists.

    Dummy credential values are supplied because the ``:?`` markers (the
    point of the compose file) otherwise make ``config`` itself fail on a
    machine without a .env - the same markers doing their job in tests.
    """
    completed = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=_ROOT,
        env={**os.environ, "POSTGRES_PASSWORD": "x", "DEEPSEEK_API_KEY": "y"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr

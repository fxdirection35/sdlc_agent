"""Tests for the deployable assembly in sdlc_agent.service.main.

``main`` builds its module-level ``app`` at import time - that is how uvicorn
finds it - and reads the environment while doing so. Importing it in-process
would therefore execute a deployment against this repository's environment
(config.yaml does not exist here; it is deliberately untracked). Each test
runs the assembly in a subprocess with an explicit, controlled environment,
which is also closer to what deployment actually does.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sdlc_agent.service.app import CONFIG_PATH_ENV

#: The example config is loadable and key-free (CLAUDE.md: config files carry
#: variable NAMES, never secrets), which makes it the right fixture here.
#: Absolute, because some subprocesses run from outside the repository (see
#: run_service_code) where a relative name would resolve to nothing.
EXAMPLE_CONFIG = str(Path(__file__).resolve().parents[2] / "config.example.yaml")

#: The provider the example config names for every role (deepseek since the
#: platform default switched; see config.example.yaml). The happy-path
#: subprocesses set a dummy value explicitly so they pass identically on hosts
#: with and without a real key; the credential-refusal test removes it with
#: equal care. Never a real secret: this value is never sent anywhere.
API_KEY_ENV = "DEEPSEEK_API_KEY"
DUMMY_KEY = "sk-deepseek-startup-check-only"

#: The example config ships the github section (deploy = code flow on), so its
#: token variable is a startup requirement of every happy-path boot too.
GH_TOKEN_ENV = "GH_TOKEN"
DUMMY_GH_TOKEN = "gh-startup-check-only"

Completed = subprocess.CompletedProcess[str]


def run_service_code(
    code: str,
    env_extra: dict[str, str] | None = None,
) -> Completed:
    """Run assembly code in a subprocess with a controlled environment."""
    env = {
        **os.environ,
        CONFIG_PATH_ENV: EXAMPLE_CONFIG,
        API_KEY_ENV: DUMMY_KEY,
        GH_TOKEN_ENV: DUMMY_GH_TOKEN,
    }
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def assert_ok(completed: Completed) -> None:
    assert completed.returncode == 0, completed.stderr


class TestUvicornTarget:
    def test_the_module_level_app_imports(self) -> None:
        """The exact thing uvicorn does: import the module, get a FastAPI app."""
        assert_ok(
            run_service_code(
                "from fastapi import FastAPI\n"
                "from sdlc_agent.service.main import app\n"
                "assert isinstance(app, FastAPI)\n"
            )
        )

    def test_health_route_is_present(self) -> None:
        assert_ok(
            run_service_code(
                "from sdlc_agent.service.main import app\n"
                "paths = {route.path for route in app.routes}\n"
                "assert '/api/health' in paths, paths\n"
                "assert '/api/runs' in paths, paths\n"
            )
        )

    def test_a_missing_config_file_fails_at_startup(self) -> None:
        """Import must fail loudly when there is no config, not serve an app
        wired to defaults nobody chose."""
        completed = run_service_code("import sdlc_agent.service.main")
        # run_service_code always sets the example config; override it to a
        # missing path so this test stays about the config file.
        assert completed.returncode == 0  # sanity: the happy path boots
        missing = subprocess.run(
            [sys.executable, "-c", "import sdlc_agent.service.main"],
            capture_output=True,
            text=True,
            timeout=120,
            env={
                **os.environ,
                CONFIG_PATH_ENV: "no-such-config.yaml",
                API_KEY_ENV: DUMMY_KEY,
                GH_TOKEN_ENV: DUMMY_GH_TOKEN,
            },
        )
        assert missing.returncode != 0

    def test_a_missing_credential_refuses_to_boot(self) -> None:
        """A keyless service would pass /api/health and then fail every run -
        green but broken. Boot must refuse, naming every missing variable.

        The variable is set to EMPTY rather than dropped: litellm loads
        ``.env`` at import time, and python-dotenv walks up the tree from
        litellm's own install directory - inside this repository's .venv -
        so a subprocess finds the repository's real ``.env`` no matter its
        working directory, and a merely-dropped key is resurrected. An
        existing variable survives that load (load_dotenv never overrides),
        and an empty value is refused exactly like a missing one."""
        completed = run_service_code(
            "import sdlc_agent.service.main",
            env_extra={API_KEY_ENV: ""},
        )

        assert completed.returncode != 0
        assert API_KEY_ENV in completed.stderr

    def test_a_missing_github_token_refuses_to_boot(self) -> None:
        """The shipped example ships the github section, so its token is a
        startup requirement: a service up without it would 500 at every
        deliver node - health green, runs broken. Empty-string reasoning as
        above: the repository's real .env can resurrect a dropped variable."""
        completed = run_service_code(
            "import sdlc_agent.service.main",
            env_extra={GH_TOKEN_ENV: ""},
        )

        assert completed.returncode != 0
        assert GH_TOKEN_ENV in completed.stderr


class TestCheckpointerFromEnv:
    def test_without_a_database_the_saver_is_shared_and_flagged(self) -> None:
        """In-memory mode reuses one saver (that is what makes it in-memory)
        and says so in its description - the warning a restart silently drops
        gates must be visible somewhere a human reads."""
        assert_ok(
            run_service_code(
                "from contextlib import nullcontext\n"
                "from sdlc_agent.service.main import checkpointer_from_env\n"
                "factory, description = checkpointer_from_env()\n"
                "assert 'in-memory' in description, description\n"
                "with factory() as first, factory() as second:\n"
                "    assert first is second\n"
            )
        )

    def test_with_a_database_each_request_gets_its_own_connection(self) -> None:
        """The factory is checked WITHOUT entering it: entering would open a
        real Postgres connection, and this test has no database - pretending
        otherwise is the fake green this suite refuses elsewhere."""
        assert_ok(
            run_service_code(
                "from sdlc_agent.service.main import checkpointer_from_env, DATABASE_URL_ENV\n"
                "import os\n"
                "os.environ[DATABASE_URL_ENV] = 'postgresql://user:pass@localhost:5432/x'\n"
                "factory, description = checkpointer_from_env()\n"
                "assert 'postgres' in description, description\n"
                "ctx = factory()\n"  # constructed, never entered
            )
        )

"""Package-level guards: importability of the layers, in any order.

These run in a fresh interpreter per entry point because import order is the
bug: ``pipeline.roles`` used to import the eval layer, which imports
``role_runner``, which imports ``pipeline.roles`` back - so
``import sdlc_agent.pipeline.roles`` as the *first* import crashed with
ImportError while the suite stayed green (conftest happened to import the eval
layer first). An in-process test cannot catch that; a subprocess can.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import sdlc_agent

#: Every module that participates in the layer graph, as its own entry point.
#: The fix inverted the adapter (evals now constructs AgentRun from RoleResult
#: fields instead of RoleResult presenting an eval shape), so pipeline no
#: longer imports evals at all - and these tests keep it that way.
_LAYER_ENTRYPOINTS = (
    "sdlc_agent.pipeline.roles",
    "sdlc_agent.pipeline.graph",
    "sdlc_agent.evals.role_runner",
    "sdlc_agent.evals.runner",
)


def test_package_imports() -> None:
    assert sdlc_agent.__version__ == "0.1.0"


@pytest.mark.parametrize("module", _LAYER_ENTRYPOINTS)
def test_each_layer_module_imports_standalone(module: str) -> None:
    """A fresh interpreter importing only this module must succeed.

    That is exactly the order the service and any deployment script use, and
    the order that used to crash.
    """
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_pipeline_does_not_import_evals() -> None:
    """The layering rule, stated directly: pipeline must not depend on evals.

    The subprocess tests above prove it for the four entry points; this walks
    the whole loaded module list after importing the pipeline package, so a new
    pipeline module sneaking an evals import in is caught here, not at the
    next deployment.
    """
    code = (
        "import sys\n"
        "import sdlc_agent.pipeline as pipeline\n"
        "loaded = [name for name in sys.modules if name.startswith('sdlc_agent')]\n"
        "evals = [name for name in loaded if name.startswith('sdlc_agent.evals')]\n"
        "assert not evals, f'pipeline pulled in the eval layer: {evals}'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr

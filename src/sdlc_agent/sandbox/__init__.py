"""Sandbox execution layer: optional, and imported only when installed.

This package needs the ``sandbox`` extra (the OpenHands SDK), so the pipeline
core never imports it - the service wires it in when composing the graph.
The executor implements the same :class:`~sdlc_agent.pipeline.verify.TestExecutor`
protocol as the stub, so the graph does not change when it swaps in.
"""

from __future__ import annotations

from sdlc_agent.sandbox.executor import SandboxExecutor

__all__ = ["SandboxExecutor"]

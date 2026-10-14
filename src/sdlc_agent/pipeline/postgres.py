"""Postgres checkpointer: a gate only survives a restart if its checkpoint does.

The in-memory saver loses every pending interrupt when the process exits, which
would turn a human gate into a fresh run. This module wraps ``PostgresSaver``
with the same allowlisted serializer used in memory, so a checkpoint written
before a restart deserializes under exactly the rules that wrote it - and the
allowlist stays the single source of truth for both.

The connection string is a credential (HARNESS_SPEC.md section 4.1: no secrets
where the pipeline can read them), so callers read it from the environment.
This module takes it as a parameter and keeps no default.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver

from sdlc_agent.pipeline.serde import make_serializer


@contextmanager
def postgres_checkpointer(conn_string: str) -> Iterator[PostgresSaver]:
    """Yield a set-up ``PostgresSaver`` bound to one connection.

    A context manager because the saver owns the connection it was built on;
    exiting the block closes it. ``setup()`` runs on every open - its
    migrations are idempotent, and paying one cheap ``CREATE TABLE IF NOT
    EXISTS`` per startup beats asking every caller to track whether this
    database was initialized yet.

    The serializer is assigned after construction rather than threaded through
    a duplicated connection setup: ``from_conn_string`` owns the load-bearing
    connection options (``autocommit``, ``prepare_threshold=0``, dict rows) and
    those may change upstream. ``serde`` is a plain attribute on every saver,
    and ``JsonPlusSerializer`` already implements the typed dump/load protocol
    natively, so the assignment is equivalent to the constructor argument.
    """
    with PostgresSaver.from_conn_string(conn_string) as saver:
        saver.serde = make_serializer()
        saver.setup()
        yield saver


__all__ = ["postgres_checkpointer"]

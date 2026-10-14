"""The HTTP service: FastAPI around the pipeline graph.

The routes advance runs one step per request (create runs to the first
interrupt; each review or budget decision runs to the next), map graph errors
to HTTP codes, and own no business logic. Deployment wiring - environment,
config file, litellm roles, Postgres checkpointer - lives in
:mod:`sdlc_agent.service.main`; everything here is parameterized so tests
drive it with scripted roles and an in-memory checkpointer.
"""

from __future__ import annotations

from sdlc_agent.service.app import CheckpointerFactory, create_app
from sdlc_agent.service.models import (
    BudgetDecisionRequest,
    PendingDecision,
    ReviewRequest,
    ReviewView,
    RunCreateRequest,
    RunResponse,
)

__all__ = [
    "BudgetDecisionRequest",
    "CheckpointerFactory",
    "PendingDecision",
    "ReviewRequest",
    "ReviewView",
    "RunCreateRequest",
    "RunResponse",
    "create_app",
]

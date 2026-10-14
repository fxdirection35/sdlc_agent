"""Role interface: what a pipeline role does, independent of how it does it.

P1 runs the flow on deterministic stubs so state transitions and gates can be
verified without an API key or a bill. P2 swaps in litellm-backed roles; the
graph does not change, because nothing in it depends on which implementation
produced an artifact.

``RoleResult`` deliberately carries the same evidence the eval harness scores -
tokens, tools, raw payload - so a production run can be replayed through the
golden tasks without a separate instrumentation path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from sdlc_agent.domain import Artifact, ArtifactKind, PipelineRole
from sdlc_agent.domain.product import Requirement


class RoleError(RuntimeError):
    """A role could not produce its artifact.

    Distinct from a bad artifact: this is an infrastructure or provider failure,
    which should not consume a QA retry or a gate revision.
    """


@dataclass(frozen=True, slots=True)
class RoleResult:
    """What one role invocation produced, plus what it cost.

    Attributes:
        role: Which role ran, so the graph can record it without trusting the
            artifact's own kind alone.
        artifact: The produced artifact.
        spent_usd: Cost of this invocation. Accumulated into the run's budget.
        output_tokens: Tokens emitted, for the efficiency dimension of evals.
        tools_called: Tool names invoked, in order. Enforced against the role's
            permitted set by the ``tools_within`` eval check.
        raw_payload: Unvalidated model output, kept so a schema failure can be
            diagnosed from the report instead of guessed at.
    """

    role: PipelineRole
    artifact: Artifact
    spent_usd: float = 0.0
    output_tokens: int = 0
    tools_called: tuple[str, ...] = ()
    raw_payload: str = ""


@dataclass(frozen=True, slots=True)
class RoleContext:
    """What a role is allowed to see when it runs.

    Scoped deliberately rather than passing the whole state: a role that can
    read everything can also be confused by everything, and HARNESS_SPEC.md
    section 4.1 requires context to be cut to the task.

    Attributes:
        requirement: The originating request. Always in scope.
        upstream: Artifacts this role depends on, keyed by kind.
        feedback: The rejection comment from the gate that sent this role back,
            empty on a first attempt.
        attempt: Which attempt this is, from 1. Lets a role vary its approach on
            a retry rather than reproducing the rejected output.
        repo_root: Sandbox working directory, if the role touches code.
    """

    requirement: Requirement
    upstream: dict[ArtifactKind, Artifact] = field(default_factory=dict)
    feedback: str = ""
    attempt: int = 1
    repo_root: str = ""


class Role(Protocol):
    """One pipeline role.

    Implementations must be side-effect free with respect to the repository:
    they return an artifact and the graph decides what to persist. That is what
    makes the resume behaviour safe (HARNESS_SPEC.md section 2.4) - a node that
    wrote files before an interrupt would write them twice.
    """

    role: PipelineRole

    def run(self, context: RoleContext) -> RoleResult:
        """Produce this role's artifact."""
        ...


class ScriptedRole:
    """A deterministic role that returns prearranged artifacts.

    Used by the P1 tests to drive the graph down a chosen path - approve,
    reject, exhaust retries - without an LLM. Also the reference implementation
    of the ``Role`` protocol, which the litellm-backed roles must match.
    """

    def __init__(
        self,
        role: PipelineRole,
        factory: Callable[[RoleContext], Artifact],
        *,
        spent_usd: float = 0.0,
        output_tokens: int = 100,
        tools_called: tuple[str, ...] = (),
    ) -> None:
        self.role = role
        self._factory = factory
        self._spent_usd = spent_usd
        self._output_tokens = output_tokens
        self._tools_called = tools_called
        self.calls: list[RoleContext] = []

    def run(self, context: RoleContext) -> RoleResult:
        self.calls.append(context)
        artifact = self._factory(context)
        return RoleResult(
            role=self.role,
            artifact=artifact,
            spent_usd=self._spent_usd,
            output_tokens=self._output_tokens,
            tools_called=self._tools_called,
            raw_payload=artifact.to_json(),
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FailingRole:
    """A role that always raises, to exercise the error path."""

    def __init__(self, role: PipelineRole, message: str = "provider unavailable") -> None:
        self.role = role
        self._message = message
        self.call_count = 0

    def run(self, context: RoleContext) -> RoleResult:
        self.call_count += 1
        raise RoleError(f"{self.role.value}: {self._message}")


#: Artifacts each role is responsible for, and the ones it consumes. The graph
#: uses this to build each role's scoped context and to route handoffs.
ROLE_OUTPUTS: dict[PipelineRole, ArtifactKind] = {
    PipelineRole.PRODUCT_MANAGER: ArtifactKind.PRD,
    PipelineRole.ARCHITECT: ArtifactKind.DESIGN,
    PipelineRole.DEVELOPER: ArtifactKind.CODE_CHANGE,
    PipelineRole.QA: ArtifactKind.TEST_PLAN,
}

#: What each role reads in addition to the requirement. Kept minimal on purpose:
#: QA needs the PRD's criteria and the change under test, and nothing else.
ROLE_INPUTS: dict[PipelineRole, tuple[ArtifactKind, ...]] = {
    PipelineRole.PRODUCT_MANAGER: (),
    PipelineRole.ARCHITECT: (ArtifactKind.PRD,),
    PipelineRole.DEVELOPER: (ArtifactKind.PRD, ArtifactKind.DESIGN),
    PipelineRole.QA: (ArtifactKind.PRD, ArtifactKind.CODE_CHANGE),
}

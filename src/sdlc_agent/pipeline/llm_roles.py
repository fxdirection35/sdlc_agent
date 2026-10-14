"""litellm-backed role implementations.

The P1 scripted roles exercise the flow; these exercise the models. Both
implement the same :class:`~sdlc_agent.pipeline.roles.Role` protocol, so the
graph does not care which is driving it - which is the point of the interface.

Structured output is enforced by instructor against the artifact model rather
than described in prose: the prompt says *what* the role produces, the schema
says what shape that must take. Describing the schema in prose would duplicate
it and drift the moment either changes.

Secrets stay out of this module. The config names the environment variable; the
key is read at call time by ``resolve_api_key`` and passed to litellm directly.
"""

from __future__ import annotations

import json
from typing import Any

import instructor
from instructor import Instructor, from_litellm
from litellm import completion as litellm_completion
from pydantic import ValidationError

from sdlc_agent.config import PlatformConfig, StructuredOutputMode
from sdlc_agent.domain import ARTIFACT_TYPES, Artifact, PipelineRole
from sdlc_agent.pipeline.roles import (
    ROLE_OUTPUTS,
    RoleContext,
    RoleError,
    RoleResult,
)

#: One instructor client per structured-output mode, built on first use and
#: reused. instructor wraps litellm's completion and keeps no per-role state,
#: and client construction is pure offline setup, so caching by mode is safe
#: and avoids rebuilding per role invocation.
_CLIENTS: dict[instructor.Mode, Instructor] = {}

#: The mode used when a provider does not name one. JSON_SCHEMA, not
#: instructor's TOOLS default: TOOLS forces ``tool_choice='required'`` for
#: structured output, which litellm rejects client-side for Claude models
#: ("does not support forced tool use") - every role call would die in
#: parameter validation before any network traffic (measured against
#: instructor 1.17 / litellm 1.105; found by the service smoke test, hidden
#: from the suite until then because the golden cassettes had never been
#: recorded). JSON_SCHEMA puts the artifact schema in the request itself and
#: reaches the Anthropic API cleanly - but DeepSeek rejects that response
#: format API-wide and serves json_object instead, which is why the mode is
#: per provider (see StructuredOutputMode) rather than this constant.
_DEFAULT_MODE = instructor.Mode.JSON_SCHEMA

_MODES: dict[StructuredOutputMode, instructor.Mode] = {
    StructuredOutputMode.JSON_SCHEMA: instructor.Mode.JSON_SCHEMA,
    StructuredOutputMode.JSON: instructor.Mode.JSON,
    StructuredOutputMode.TOOLS: instructor.Mode.TOOLS,
    StructuredOutputMode.MD_JSON: instructor.Mode.MD_JSON,
}


def _client_for(mode: instructor.Mode) -> Instructor:
    client = _CLIENTS.get(mode)
    if client is None:
        client = from_litellm(litellm_completion, mode=mode)
        _CLIENTS[mode] = client
    return client


#: What each role is told about itself. Kept to one line of identity plus the
#: contract; everything else is the schema's job.
_ROLE_BRIEF: dict[PipelineRole, str] = {
    PipelineRole.PRODUCT_MANAGER: (
        "You are a product manager. Turn the requirement into a reviewable PRD "
        "with explicit goals, non-goals and acceptance criteria."
    ),
    PipelineRole.ARCHITECT: (
        "You are a software architect. Turn the approved PRD into a design "
        "naming the components, api changes and risks."
    ),
    PipelineRole.DEVELOPER: (
        "You are a software developer. Implement the task from the design and "
        "report the change honestly: what changed, where, and any deviations."
    ),
    PipelineRole.QA: (
        "You are a QA engineer. Write a test plan that proves every acceptance "
        "criterion. Record criteria you cannot test instead of dropping them."
    ),
}


def _system_prompt(role: PipelineRole) -> str:
    """The system prompt for one role.

    Stable text goes here because prompt caching is prefix-based (HARNESS_SPEC.md
    section 4.3): anything per-run goes in the user message instead.
    """
    return _ROLE_BRIEF[role]


def _artifact_json(artifact: Artifact) -> str:
    return artifact.to_json()


def _user_message(role: PipelineRole, context: RoleContext) -> str:
    """Assemble the per-run user message from the scoped context.

    Only the artifacts this role declares as inputs are included, mirroring
    ``_context_for`` in the graph. The feedback field carries a gate's rejection
    comment when this is a revision attempt.
    """
    sections = [f"Requirement:\n{context.requirement.to_json()}"]
    for kind, artifact in context.upstream.items():
        sections.append(f"{kind.value}:\n{_artifact_json(artifact)}")
    if context.feedback:
        sections.append(
            "Your previous attempt was rejected with this feedback. "
            f"Address it directly:\n{context.feedback}"
        )
    if context.attempt > 1 and not context.feedback:
        sections.append(f"This is attempt {context.attempt}. Revise your previous output.")
    sections.append(f"Produce the {ROLE_OUTPUTS[role].value} now.")
    return "\n\n".join(sections)


def _call_kwargs(role: PipelineRole, config: PlatformConfig) -> dict[str, Any]:
    """litellm kwargs derived from the role's model config.

    ``temperature`` is omitted unless explicitly set: current Claude models
    reject any non-default sampling value, so the default must be absence.
    ``effort`` maps to litellm's ``reasoning_effort``, which for Anthropic models
    becomes ``output_config.effort`` - the actual quality lever.
    """
    role_config = config.roles[role]
    kwargs: dict[str, Any] = {
        "model": config.litellm_model(role),
        "max_tokens": role_config.max_output_tokens,
    }
    if role_config.effort is not None:
        kwargs["reasoning_effort"] = role_config.effort.value
    if role_config.temperature is not None:
        kwargs["temperature"] = role_config.temperature
    return kwargs


class LitellmRole:
    """A role backed by a real model call through litellm."""

    def __init__(self, role: PipelineRole, config: PlatformConfig) -> None:
        self.role = role
        self._config = config

    def run(self, context: RoleContext) -> RoleResult:
        response_model = ARTIFACT_TYPES[ROLE_OUTPUTS[self.role]]
        api_key = self._config.resolve_api_key(self.role)
        # The wire shape is per provider: model string and structured-output
        # mode are resolved together from the same config.
        client = _client_for(_MODES[self._config.instructor_mode_for(self.role)])

        try:
            artifact, raw = client.chat.completions.create_with_completion(
                messages=[
                    {"role": "system", "content": _system_prompt(self.role)},
                    {"role": "user", "content": _user_message(self.role, context)},
                ],
                response_model=response_model,
                max_retries=2,
                api_key=api_key,
                **_call_kwargs(self.role, self._config),
            )
        except (ValidationError, ValueError) as exc:
            # instructor re-raises validation failures after its retries are
            # spent; surface them as a role error so the graph escalates instead
            # of treating an unparseable answer as bad work by the role.
            raise RoleError(
                f"{self.role.value}: model output failed validation after retries: {exc}"
            ) from exc
        except Exception as exc:
            # Broad on purpose: every failure mode here (auth, rate limit,
            # network) should escalate the run, and RoleError preserves the
            # original message for the audit trail.
            raise RoleError(f"{self.role.value}: model call failed: {exc}") from exc

        usage = getattr(raw, "usage", None)
        hidden = getattr(raw, "_hidden_params", None) or {}
        response_cost = hidden.get("response_cost", 0.0)

        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        spent_usd = float(response_cost or 0.0)

        return RoleResult(
            role=self.role,
            artifact=artifact,
            spent_usd=spent_usd,
            output_tokens=int(output_tokens),
            raw_payload=json.dumps(artifact.model_dump(mode="json"), indent=2, default=str),
        )

"""Tests for the litellm-backed roles, without any LLM calls.

The real client is a module-level singleton, so tests swap in a fake that
returns a canned artifact and records the kwargs it was called with. That is
what makes these tests assert the contract - the model string, the reasoning
effort, the absence of a temperature parameter - rather than a provider's
behaviour.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import instructor
import pytest

from sdlc_agent.config import MissingCredentialError, PlatformConfig
from sdlc_agent.domain import Artifact, ArtifactKind, PipelineRole
from sdlc_agent.domain.product import AcceptanceCriterion, ProductRequirements, Requirement
from sdlc_agent.pipeline import llm_roles
from sdlc_agent.pipeline.roles import RoleContext, RoleError

from .helpers import make_requirement


class FakeCompletions:
    """Stands in for instructor's chat.completions.create_with_completion."""

    def __init__(
        self,
        artifact: ProductRequirements | None = None,
        *,
        fail_with: Exception | None = None,
    ) -> None:
        self._artifact = artifact or make_prd_artifact()
        self._fail_with = fail_with
        self.calls: list[dict[str, Any]] = []

    def create_with_completion(self, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append(kwargs)
        if self._fail_with is not None:
            raise self._fail_with
        raw = SimpleNamespace(
            usage=SimpleNamespace(completion_tokens=333),
            _hidden_params={"response_cost": 0.42},
        )
        return self._artifact, raw


def make_prd_artifact() -> ProductRequirements:
    return ProductRequirements.model_validate(
        {
            "requirement_id": uuid.uuid4(),
            "problem_statement": "Report data cannot leave the app.",
            "goals": ("Users can export any filtered report as CSV",),
            "acceptance_criteria": (
                AcceptanceCriterion.model_validate(
                    {"statement": "CSV has one header row", "verification": "pytest tests/x.py"}
                ),
            ),
        }
    )


class FakeInstructor:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def install_fake(
    monkeypatch: pytest.MonkeyPatch, completions: FakeCompletions, mode: instructor.Mode
) -> None:
    """Replace the client cache for one test with fakes under ``mode``.

    The cache dict itself is swapped (monkeypatch restores the original
    afterwards), so no fake or real client ever leaks between tests. Modes a
    test does NOT install stay absent - selecting one of those builds a real
    instructor client, which is offline but visible in a test's failure as a
    network attempt, exactly the loudness wanted here.
    """
    monkeypatch.setattr(llm_roles, "_CLIENTS", {mode: FakeInstructor(completions)})


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test needs the key in the environment; the fake client never uses it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


@pytest.fixture
def config() -> PlatformConfig:
    return PlatformConfig.model_validate(
        {
            "providers": {"anthropic": {"api_key_env": "ANTHROPIC_API_KEY"}},
            "roles": {
                role.value: {
                    "provider": "anthropic",
                    "model": "claude-opus-5-5",
                    "effort": "high",
                }
                for role in PipelineRole
            },
        }
    )


@pytest.fixture
def fake(
    monkeypatch: pytest.MonkeyPatch,
) -> FakeCompletions:
    fake_completions = FakeCompletions()
    install_fake(monkeypatch, fake_completions, instructor.Mode.JSON_SCHEMA)
    return fake_completions


@pytest.fixture
def role(config: PlatformConfig) -> llm_roles.LitellmRole:
    return llm_roles.LitellmRole(PipelineRole.PRODUCT_MANAGER, config)


def make_context(
    requirement: Requirement | None = None,
    *,
    feedback: str = "",
    upstream: dict[ArtifactKind, Artifact] | None = None,
) -> RoleContext:
    return RoleContext(
        requirement=requirement or make_requirement(),
        feedback=feedback,
        upstream=upstream or {},
    )


class TestModeSelection:
    def test_the_default_mode_is_json_schema(self) -> None:
        """instructor's TOOLS default forces tool_choice='required', which
        litellm rejects client-side for Claude models - every role call would
        die in parameter validation before any network traffic (measured;
        surfaced by the service smoke test, invisible to these tests because
        the fake replaces the whole client). If this pin breaks, the change
        was deliberate - re-measure against a real provider."""
        assert llm_roles._DEFAULT_MODE is instructor.Mode.JSON_SCHEMA

    def test_the_mode_comes_from_the_role_s_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DeepSeek serves json_object, not json_schema (measured against the
        real API), so a deepseek role must be driven by the JSON-mode client.
        The JSON_SCHEMA slot holds a fake that fails loudly, so selecting the
        default mode here is a visible failure, not a silent network call."""
        config = PlatformConfig.model_validate(
            {
                "providers": {
                    "deepseek": {"api_key_env": "DEEPSEEK_API_KEY", "instructor_mode": "json"}
                },
                "roles": {
                    role.value: {"provider": "deepseek", "model": "deepseek-v4-flash"}
                    for role in PipelineRole
                },
            }
        )
        fake_completions = FakeCompletions()
        monkeypatch.setattr(
            llm_roles,
            "_CLIENTS",
            {
                instructor.Mode.JSON: FakeInstructor(fake_completions),
                instructor.Mode.JSON_SCHEMA: FakeInstructor(
                    FakeCompletions(fail_with=AssertionError("wrong mode for deepseek"))
                ),
            },
        )
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")

        llm_roles.LitellmRole(PipelineRole.QA, config).run(make_context())

        assert fake_completions.calls, "the JSON-mode client must have served the call"

    def test_clients_are_cached_per_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One client per mode, reused across role calls: construction is pure
        offline setup, and rebuilding per request would allocate for nothing."""
        monkeypatch.setattr(llm_roles, "_CLIENTS", {})

        first = llm_roles._client_for(instructor.Mode.JSON_SCHEMA)
        second = llm_roles._client_for(instructor.Mode.JSON_SCHEMA)
        other = llm_roles._client_for(instructor.Mode.JSON)

        assert first is second
        assert first is not other


class TestRoleBriefs:
    def test_every_role_has_a_brief(self) -> None:
        """A missing brief would silently produce an empty system prompt."""
        assert set(llm_roles._ROLE_BRIEF) == set(PipelineRole)

    def test_briefs_are_stable_text(self) -> None:
        """Prompts are cached prefix-first; per-run content must not leak in."""
        for brief in llm_roles._ROLE_BRIEF.values():
            assert "Requirement:" not in brief
            assert "{feedback}" not in brief


class TestSuccessfulCall:
    def test_returns_the_artifact_with_cost_and_tokens(
        self, role: llm_roles.LitellmRole, fake: FakeCompletions
    ) -> None:
        result = role.run(make_context())

        assert isinstance(result.artifact, ProductRequirements)
        assert result.spent_usd == pytest.approx(0.42)
        assert result.output_tokens == 333
        assert result.role is PipelineRole.PRODUCT_MANAGER

    def test_calls_with_the_configured_model(
        self, role: llm_roles.LitellmRole, fake: FakeCompletions
    ) -> None:
        role.run(make_context())

        assert fake.calls[0]["model"] == "anthropic/claude-opus-5-5"

    def test_maps_effort_to_reasoning_effort(
        self, role: llm_roles.LitellmRole, fake: FakeCompletions
    ) -> None:
        role.run(make_context())

        assert fake.calls[0]["reasoning_effort"] == "high"

    def test_omits_temperature_when_not_configured(
        self, role: llm_roles.LitellmRole, fake: FakeCompletions
    ) -> None:
        """Current Claude models 400 on any non-default sampling value."""
        role.run(make_context())

        assert "temperature" not in fake.calls[0]

    def test_reads_the_api_key_from_the_environment(
        self,
        role: llm_roles.LitellmRole,
        fake: FakeCompletions,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        role.run(make_context())

        assert fake.calls[0]["api_key"] == "sk-ant-test"

    def test_user_message_contains_the_requirement(
        self, role: llm_roles.LitellmRole, fake: FakeCompletions
    ) -> None:
        requirement = make_requirement()

        role.run(make_context(requirement))

        messages = fake.calls[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert requirement.title in messages[1]["content"]
        # Serialized as dashed UUID, not bare hex.
        assert str(requirement.artifact_id) in messages[1]["content"]

    def test_feedback_is_included_in_the_user_message(
        self, role: llm_roles.LitellmRole, fake: FakeCompletions
    ) -> None:
        context = make_context(feedback="the PRD is missing scope")

        role.run(context)

        content = fake.calls[0]["messages"][1]["content"]
        assert "missing scope" in content

    def test_no_feedback_means_no_feedback_section(
        self, role: llm_roles.LitellmRole, fake: FakeCompletions
    ) -> None:
        role.run(make_context())

        content = fake.calls[0]["messages"][1]["content"]
        assert "previous attempt was rejected" not in content


class TestErrorHandling:
    @pytest.mark.parametrize(
        "error",
        [
            ValueError("validation failed"),
            RuntimeError("connection reset"),
        ],
    )
    def test_any_failure_surfaces_as_role_error(
        self,
        role: llm_roles.LitellmRole,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
    ) -> None:
        fake = FakeCompletions(fail_with=error)
        install_fake(monkeypatch, fake, instructor.Mode.JSON_SCHEMA)

        with pytest.raises(RoleError, match="product_manager"):
            role.run(make_context())

    def test_missing_api_key_fails_before_any_call(
        self,
        role: llm_roles.LitellmRole,
        fake: FakeCompletions,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The key resolves before the call; a missing one never reaches litellm."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(MissingCredentialError, match="ANTHROPIC_API_KEY"):
            role.run(make_context())

        assert fake.calls == []


class TestPromptShape:
    def test_upstream_artifacts_are_included(
        self,
        config: PlatformConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The architect sees the PRD; the QA role sees the PRD and the change."""
        fake = FakeCompletions()
        install_fake(monkeypatch, fake, instructor.Mode.JSON_SCHEMA)
        architect = llm_roles.LitellmRole(PipelineRole.ARCHITECT, config)
        prd = make_prd_artifact()

        architect.run(make_context(upstream={ArtifactKind.PRD: prd}))

        content = fake.calls[0]["messages"][1]["content"]
        assert "prd:" in content
        assert "code_change:" not in content

    def test_an_empty_repo_tree_leaves_the_message_byte_identical(self) -> None:
        """The cassette contract: a role without a workspace must send the
        exact message recorded before the tree section existed. Any change to
        this string is a deliberate prompt change and needs a re-record."""
        requirement = make_requirement()
        prd = make_prd_artifact()
        context = RoleContext(
            requirement=requirement,
            upstream={ArtifactKind.PRD: prd},
            feedback="make the goals testable",
        )

        expected = (
            f"Requirement:\n{requirement.to_json()}\n\n"
            f"prd:\n{prd.to_json()}\n\n"
            "Your previous attempt was rejected with this feedback. "
            "Address it directly:\nmake the goals testable\n\n"
            "Produce the design now."
        )

        assert llm_roles._user_message(PipelineRole.ARCHITECT, context) == expected

    def test_the_tree_section_sits_between_artifacts_and_feedback(self) -> None:
        context = RoleContext(
            requirement=make_requirement(),
            feedback="tighten it",
            repo_tree="src/\n  app.py",
        )

        message = llm_roles._user_message(PipelineRole.DEVELOPER, context)

        assert "Repository file tree:\nsrc/\n  app.py" in message
        assert message.index("Repository file tree:") < message.index(
            "Your previous attempt was rejected"
        )

"""Tests for the configuration layer.

The security-critical property is that a config file holds the *name* of an
environment variable, never a credential: the file is committed, so a key in
it would be a key in git history. Several tests below exist to keep that true.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from sdlc_agent.config import (
    BudgetConfig,
    EffortLevel,
    MissingCredentialError,
    PlatformConfig,
    ProviderConfig,
    RoleModelConfig,
    load_config,
)
from sdlc_agent.domain.base import PipelineRole

MINIMAL_YAML = """
providers:
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
roles:
  product_manager: {provider: anthropic, model: claude-opus-5}
  architect:       {provider: anthropic, model: claude-opus-5}
  developer:       {provider: anthropic, model: claude-sonnet-5-5}
  qa:              {provider: anthropic, model: claude-haiku-5-5}
"""


def write_config(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def minimal_config() -> PlatformConfig:
    return PlatformConfig.model_validate(
        {
            "providers": {"anthropic": {"api_key_env": "ANTHROPIC_API_KEY"}},
            "roles": {
                role.value: {"provider": "anthropic", "model": "claude-sonnet-5-5"}
                for role in PipelineRole
            },
        }
    )


class TestLoadConfig:
    def test_loads_a_valid_file(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, MINIMAL_YAML)

        config = load_config(path)

        assert set(config.providers) == {"anthropic"}
        assert config.roles[PipelineRole.QA].model == "claude-haiku-5-5"

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, MINIMAL_YAML)

        assert isinstance(load_config(str(path)), PlatformConfig)

    def test_missing_file_names_the_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.yaml"

        with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
            load_config(missing)

    def test_empty_file_is_an_error_not_an_empty_config(self, tmp_path: Path) -> None:
        """An empty file silently producing a config with no providers would
        fail much later, at the first LLM call."""
        path = write_config(tmp_path, "# only a comment\n")

        with pytest.raises(ValueError, match="empty"):
            load_config(path)

    def test_non_mapping_top_level_is_rejected(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, "- one\n- two\n")

        with pytest.raises(ValueError, match="mapping at the top level"):
            load_config(path)

    def test_invalid_yaml_is_rejected(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, "providers: [unclosed\n")

        with pytest.raises(Exception):  # noqa: B017 - yaml.YAMLError, not a pydantic error
            load_config(path)

    def test_schema_violation_surfaces_as_validation_error(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, MINIMAL_YAML.replace("api_key_env", "api_key"))

        with pytest.raises(ValidationError):
            load_config(path)


class TestProviderConfig:
    def test_uppercases_the_env_var_name(self) -> None:
        """Windows env vars are case-insensitive but the canonical form is upper."""
        provider = ProviderConfig.model_validate({"api_key_env": "anthropic_api_key"})

        assert provider.api_key_env == "ANTHROPIC_API_KEY"

    @pytest.mark.parametrize(
        "name",
        ["", "   ", "1NOT_VALID", "HAS-DASH", "HAS SPACE", "HAS.DOT"],
    )
    def test_rejects_malformed_env_var_names(self, name: str) -> None:
        """A typo here would otherwise surface as a missing credential later."""
        with pytest.raises(ValidationError):
            ProviderConfig.model_validate({"api_key_env": name})

    def test_optional_fields_default_to_none(self) -> None:
        provider = ProviderConfig.model_validate({"api_key_env": "K"})

        assert provider.base_url is None
        assert provider.litellm_prefix is None

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match="api_key"):
            ProviderConfig.model_validate({"api_key_env": "K", "api_key": "sk-secret"})

    def test_stores_a_name_not_a_secret(self, tmp_path: Path) -> None:
        """The whole point of api_key_env: no credential value in the model."""
        path = write_config(tmp_path, MINIMAL_YAML)

        config = load_config(path)
        serialized = config.model_dump_json()

        assert config.providers["anthropic"].api_key_env == "ANTHROPIC_API_KEY"
        assert "sk-" not in serialized
        assert 'api_key"' not in serialized


class TestRoleModelConfig:
    def test_defaults_are_conservative(self) -> None:
        role = RoleModelConfig.model_validate({"provider": "p", "model": "m"})

        assert role.temperature is None
        assert role.effort is None
        assert role.max_output_tokens == 4096

    def test_temperature_defaults_to_none_not_a_number(self) -> None:
        """Current-generation Claude models reject a non-default sampling value
        with a 400, so the parameter must be omitted unless deliberately set."""
        assert "temperature" not in RoleModelConfig.model_validate(
            {"provider": "p", "model": "m"}
        ).model_dump(exclude_none=True)

    @pytest.mark.parametrize("temperature", [-0.1, 2.1])
    def test_rejects_out_of_range_temperature(self, temperature: float) -> None:
        with pytest.raises(ValidationError):
            RoleModelConfig.model_validate(
                {"provider": "p", "model": "m", "temperature": temperature}
            )

    def test_accepts_explicit_temperature_for_providers_that_allow_it(self) -> None:
        role = RoleModelConfig.model_validate({"provider": "p", "model": "m", "temperature": 0.7})

        assert role.temperature == 0.7

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_accepts_every_effort_level(self, effort: str) -> None:
        assert RoleModelConfig.model_validate(
            {"provider": "p", "model": "m", "effort": effort}
        ).effort is EffortLevel(effort)

    def test_rejects_unknown_effort_level(self) -> None:
        """A typo here would be a 400 on the first call of every run."""
        with pytest.raises(ValidationError):
            RoleModelConfig.model_validate({"provider": "p", "model": "m", "effort": "turbo"})

    @pytest.mark.parametrize("tokens", [0, -100])
    def test_rejects_non_positive_token_cap(self, tokens: int) -> None:
        """A cap of zero would let one call consume nothing and silently fail."""
        with pytest.raises(ValidationError):
            RoleModelConfig.model_validate(
                {"provider": "p", "model": "m", "max_output_tokens": tokens}
            )

    def test_rejects_blank_model(self) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            RoleModelConfig.model_validate({"provider": "p", "model": "  "})


class TestCrossReferenceValidation:
    def test_rejects_role_pointing_at_undefined_provider(
        self, minimal_config: PlatformConfig
    ) -> None:
        """Catches the typo before it becomes a runtime routing failure."""
        payload = minimal_config.model_dump(mode="json")
        payload["roles"]["qa"]["provider"] = "openai"

        with pytest.raises(ValidationError, match="undefined provider") as excinfo:
            PlatformConfig.model_validate(payload)

        message = str(excinfo.value)
        assert "qa" in message
        assert "openai" in message
        assert "anthropic" in message  # names what IS defined

    def test_rejects_missing_role(self, minimal_config: PlatformConfig) -> None:
        payload = minimal_config.model_dump(mode="json")
        del payload["roles"]["architect"]

        with pytest.raises(ValidationError, match="no model configured") as excinfo:
            PlatformConfig.model_validate(payload)

        assert "architect" in str(excinfo.value)

    def test_rejects_empty_providers(self) -> None:
        with pytest.raises(ValidationError):
            PlatformConfig.model_validate({"providers": {}, "roles": {}})

    def test_reports_every_missing_role_at_once(self, minimal_config: PlatformConfig) -> None:
        """One error naming all gaps, not a cascade of four."""
        payload = minimal_config.model_dump(mode="json")
        del payload["roles"]["architect"]
        del payload["roles"]["qa"]

        with pytest.raises(ValidationError) as excinfo:
            PlatformConfig.model_validate(payload)

        message = str(excinfo.value)
        assert "architect" in message
        assert "qa" in message


class TestLitellmModelResolution:
    def test_joins_provider_and_model(self, minimal_config: PlatformConfig) -> None:
        assert minimal_config.litellm_model(PipelineRole.QA) == "anthropic/claude-sonnet-5-5"

    def test_uses_known_default_prefix(self, tmp_path: Path) -> None:
        """deepseek is a known provider, so no override is needed."""
        path = write_config(
            tmp_path,
            MINIMAL_YAML.replace("anthropic:", "deepseek:").replace(
                "provider: anthropic", "provider: deepseek"
            ),
        )

        config = load_config(path)

        assert config.litellm_model(PipelineRole.DEVELOPER).startswith("deepseek/")

    def test_explicit_prefix_overrides_the_key(self) -> None:
        """Needed when a config key is a friendly name, not the litellm prefix."""
        config = PlatformConfig.model_validate(
            {
                "providers": {
                    "our-gateway": {
                        "api_key_env": "GW_KEY",
                        "litellm_prefix": "openai",
                    }
                },
                "roles": {
                    role.value: {"provider": "our-gateway", "model": "gpt-x"}
                    for role in PipelineRole
                },
            }
        )

        assert config.litellm_model(PipelineRole.ARCHITECT) == "openai/gpt-x"

    def test_unknown_provider_key_falls_back_to_the_key_itself(self) -> None:
        config = PlatformConfig.model_validate(
            {
                "providers": {"acme": {"api_key_env": "ACME_KEY"}},
                "roles": {
                    role.value: {"provider": "acme", "model": "acme-1"} for role in PipelineRole
                },
            }
        )

        assert config.litellm_model(PipelineRole.QA) == "acme/acme-1"

    def test_different_roles_can_use_different_providers(self) -> None:
        """The multi-provider requirement: strong model for judgment, cheap for grunt work."""
        config = PlatformConfig.model_validate(
            {
                "providers": {
                    "anthropic": {"api_key_env": "ANTHROPIC_API_KEY"},
                    "deepseek": {"api_key_env": "DEEPSEEK_API_KEY"},
                },
                "roles": {
                    "product_manager": {"provider": "anthropic", "model": "claude-opus-5"},
                    "architect": {"provider": "anthropic", "model": "claude-opus-5"},
                    "developer": {"provider": "deepseek", "model": "deepseek-v4"},
                    "qa": {"provider": "deepseek", "model": "deepseek-v4"},
                },
            }
        )

        assert config.litellm_model(PipelineRole.ARCHITECT) == "anthropic/claude-opus-5"
        assert config.litellm_model(PipelineRole.DEVELOPER) == "deepseek/deepseek-v4"


class TestResolveApiKey:
    def test_reads_from_the_environment(
        self, minimal_config: PlatformConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-123")

        assert minimal_config.resolve_api_key(PipelineRole.QA) == "sk-ant-test-123"

    def test_strips_surrounding_whitespace(
        self, minimal_config: PlatformConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A trailing newline from a .env file is a common silent failure."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-ant-test-123\n")

        assert minimal_config.resolve_api_key(PipelineRole.QA) == "sk-ant-test-123"

    def test_missing_variable_names_provider_and_variable(
        self, minimal_config: PlatformConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(MissingCredentialError) as excinfo:
            minimal_config.resolve_api_key(PipelineRole.QA)

        message = str(excinfo.value)
        assert "anthropic" in message
        assert "ANTHROPIC_API_KEY" in message
        assert "qa" in message

    def test_empty_variable_is_treated_as_missing(
        self, minimal_config: PlatformConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exported-but-empty var is the same failure as an unset one."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")

        with pytest.raises(MissingCredentialError):
            minimal_config.resolve_api_key(PipelineRole.QA)

    def test_resolution_is_lazy_not_at_load_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Loading must succeed without keys present, so CI can validate config."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        path = write_config(tmp_path, MINIMAL_YAML)

        config = load_config(path)  # must not raise

        assert config.providers["anthropic"].api_key_env == "ANTHROPIC_API_KEY"


class TestBudgetConfig:
    def test_defaults(self) -> None:
        budget = BudgetConfig()

        assert budget.per_task_usd == 5.0
        assert budget.warn_at_fraction == 0.8
        assert budget.warn_at_usd == pytest.approx(4.0)

    @pytest.mark.parametrize("usd", [0.0, -1.0])
    def test_rejects_non_positive_ceiling(self, usd: float) -> None:
        """A zero ceiling would pause every run immediately."""
        with pytest.raises(ValidationError):
            BudgetConfig.model_validate({"per_task_usd": usd})

    @pytest.mark.parametrize("fraction", [0.0, 1.5, -0.2])
    def test_rejects_out_of_range_warn_fraction(self, fraction: float) -> None:
        with pytest.raises(ValidationError):
            BudgetConfig.model_validate({"warn_at_fraction": fraction})

    def test_warn_at_one_means_warn_only_at_the_ceiling(self) -> None:
        assert BudgetConfig.model_validate({"warn_at_fraction": 1.0}).warn_at_usd == 5.0

    def test_config_file_can_override_budget(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, MINIMAL_YAML + "\nbudget:\n  per_task_usd: 12.5\n")

        config = load_config(path)

        assert config.budget.per_task_usd == 12.5
        assert config.budget.warn_at_usd == pytest.approx(10.0)

    def test_pipeline_defaults_and_override(self, tmp_path: Path) -> None:
        default = load_config(write_config(tmp_path, MINIMAL_YAML, "a.yaml"))
        assert default.pipeline.max_qa_retries == 3

        override = load_config(
            write_config(tmp_path, MINIMAL_YAML + "\npipeline:\n  max_qa_retries: 0\n", "b.yaml")
        )
        assert override.pipeline.max_qa_retries == 0

    @pytest.mark.parametrize("retries", [-1, 11])
    def test_rejects_out_of_range_retries(self, retries: int) -> None:
        with pytest.raises(ValidationError):
            PlatformConfig.model_validate(
                {
                    "providers": {"p": {"api_key_env": "K"}},
                    "roles": {role.value: {"provider": "p", "model": "m"} for role in PipelineRole},
                    "pipeline": {"max_qa_retries": retries},
                }
            )


class TestImmutability:
    def test_config_is_frozen(self, minimal_config: PlatformConfig) -> None:
        """A config that can be mutated mid-run makes cost behaviour unauditable."""
        with pytest.raises(ValidationError):
            minimal_config.budget = BudgetConfig(per_task_usd=9999.0)  # type: ignore[misc]

    def test_nested_sections_are_frozen(self, minimal_config: PlatformConfig) -> None:
        with pytest.raises(ValidationError):
            minimal_config.budget.per_task_usd = 9999.0  # type: ignore[misc]


EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / "config.example.yaml"


class TestShippedExample:
    """Guards config.example.yaml, which nothing else exercises.

    Without this the example drifts: a renamed field or a retired model id
    leaves users copying a file that fails on load, and the failure lands on
    them rather than on the change that caused it.
    """

    def test_example_file_exists(self) -> None:
        assert EXAMPLE_CONFIG.exists(), f"missing {EXAMPLE_CONFIG.name}"

    def test_example_loads_and_validates(self) -> None:
        config = load_config(EXAMPLE_CONFIG)

        assert set(config.providers) == {"anthropic"}
        assert set(config.roles) == set(PipelineRole)

    def test_example_resolves_every_role(self) -> None:
        """Each role must resolve to a routable litellm model string."""
        config = load_config(EXAMPLE_CONFIG)

        for role in PipelineRole:
            resolved = config.litellm_model(role)
            assert "/" in resolved, f"{role.value} resolved to {resolved!r}"

    def test_example_sets_no_temperature(self) -> None:
        """Current-generation Claude models 400 on a non-default sampling value."""
        config = load_config(EXAMPLE_CONFIG)

        for role, role_config in config.roles.items():
            assert role_config.temperature is None, f"{role.value} sets temperature"

    def test_example_carries_no_secret_values(self) -> None:
        """The file is committed; a literal key in it is a key in git history.

        Checks the parsed config rather than grepping the text, since the
        comments legitimately mention the shape of a key as an illustration.
        What must be true is that no field anywhere holds a credential.
        """
        config = load_config(EXAMPLE_CONFIG)
        dumped = config.model_dump()

        def walk(node: object, path: str = "") -> Iterator[tuple[str, str]]:
            if isinstance(node, dict):
                for key, value in node.items():
                    yield from walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    yield from walk(value, f"{path}[{index}]")
            elif isinstance(node, str):
                yield path, node

        for path, value in walk(dumped):
            assert not value.startswith(("sk-", "ghp_", "xox")), f"{path} holds a credential"
            # Only variable NAMES belong here; they are short identifiers.
            assert len(value) < 60, f"{path} is too long to be a variable name: {value!r}"

        for provider in config.providers.values():
            assert provider.api_key_env.endswith("_KEY")
            assert provider.base_url is None or provider.base_url.startswith("http")

    def test_example_documents_the_env_var_it_needs(self) -> None:
        """A user must be able to tell which variable to export by reading it."""
        text = EXAMPLE_CONFIG.read_text(encoding="utf-8")

        assert "ANTHROPIC_API_KEY" in text
        assert "export" in text

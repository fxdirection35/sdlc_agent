"""Configuration: which model each role talks to, and what it may spend.

The platform supports several LLM providers at once, selected from a config
file rather than hardcoded, and lets each role use a different model - a
strong one where judgment matters, a cheap one for mechanical steps.

Secrets are the one thing this module deliberately never holds. The config
file is committed to the repository, so a provider entry names the
*environment variable* that carries its key instead of the key itself
(HARNESS_SPEC.md section 4.1 forbids credentials in agent-reachable context,
and a file under version control is about as reachable as it gets).
"""

from __future__ import annotations

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sdlc_agent.domain.base import PipelineRole, reject_blank

#: Environment variable names look like this; validating the shape up front
#: turns "I typo'd the var name" into a config error instead of a mysterious
#: missing-credential failure at the first LLM call.
_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_env_var_name(value: str) -> str:
    """Normalize an env-var name: trimmed, uppercased, shape-checked."""
    name = reject_blank(value).upper()
    if not _ENV_VAR_NAME.match(name):
        raise ValueError(f"{name!r} is not a valid environment variable name")
    return name


#: A provider may legitimately differ from the litellm prefix it needs, so the
#: config can override it. These are the ones known to differ.
DEFAULT_LITELLM_PREFIXES: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "deepseek": "deepseek",
    "azure": "azure",
    "ollama": "ollama",
}


class MissingCredentialError(RuntimeError):
    """Raised when a configured provider has no API key in the environment.

    The message names both the provider and the variable to set, so the fix is
    obvious without reading the config back.
    """


class EffortLevel(StrEnum):
    """Reasoning depth, for providers that expose it as a first-class control.

    These are Claude's ``output_config.effort`` values. Constraining to an enum
    rather than a free string means a typo fails at config load instead of as a
    400 from the API on the first call of every pipeline run.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class StructuredOutputMode(StrEnum):
    """How instructor transports the artifact schema to one provider.

    Per provider because the APIs measurably disagree (instructor 1.17 /
    litellm 1.105, probed against the real DeepSeek API):

    - ``json_schema`` puts the schema in the request itself and works for
      Anthropic; DeepSeek rejects the ``response_format`` type API-wide
      ("This response_format type is unavailable now").
    - ``tools`` is instructor's default but forces ``tool_choice='required'``,
      which litellm rejects client-side for Claude *and* which DeepSeek's
      thinking models reject server-side.
    - ``json`` (``json_object``) is the mode DeepSeek actually serves.

    An enum rather than a free string, like ``EffortLevel``, so a typo fails
    at config load instead of as a 400 on the first call of every run.
    """

    JSON_SCHEMA = "json_schema"
    JSON = "json"
    TOOLS = "tools"
    MD_JSON = "md_json"


class ProviderConfig(BaseModel):
    """One LLM provider and how to authenticate against it.

    Attributes:
        api_key_env: Name of the environment variable holding the key. The key
            itself is never stored here.
        litellm_prefix: Prefix litellm needs to route to this provider, when it
            differs from the config key. Defaults to the key itself.
        base_url: Optional endpoint override, for a self-hosted or proxied
            gateway. Left unset, litellm uses the provider's default.
        instructor_mode: Structured-output transport for this provider.
            ``None`` means the platform default (``json_schema``), which is
            what Anthropic serves; DeepSeek needs ``json``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key_env: str = Field(min_length=1)
    litellm_prefix: str | None = None
    base_url: str | None = None
    instructor_mode: StructuredOutputMode | None = None

    @field_validator("api_key_env")
    @classmethod
    def _validate_env_name(cls, value: str) -> str:
        return _validate_env_var_name(value)

    @field_validator("base_url", "litellm_prefix")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return reject_blank(value)


class RoleModelConfig(BaseModel):
    """Which model one pipeline role uses.

    Attributes:
        provider: Key into the top-level ``providers`` mapping.
        model: Model identifier as the provider names it, without any prefix.
        effort: Reasoning depth for providers that expose it (Claude's
            ``output_config.effort``). This is the primary quality-versus-cost
            lever: roles needing judgment run high, mechanical roles run low.
            ``None`` omits it, letting the provider apply its own default.
        temperature: Sampling temperature. ``None`` omits the parameter
            entirely, which is the safe default: current-generation Claude
            models reject any non-default sampling value with a 400, so setting
            this per role only makes sense for providers that actually accept
            it. Prefer ``effort``.
        max_output_tokens: Per-call cap. Bounds the cost of one runaway call
            independently of the task budget.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    effort: EffortLevel | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4096, gt=0)

    @field_validator("provider", "model")
    @classmethod
    def _strip(cls, value: str) -> str:
        return reject_blank(value)


class BudgetConfig(BaseModel):
    """Cost guardrails for a single pipeline run.

    Attributes:
        per_task_usd: Ceiling for one task, across all roles and retries. On
            breach the run pauses and raises for a human rather than quietly
            continuing to spend - the MetaGPT ``investment`` idea, kept because
            an unbounded multi-agent run is how a demo becomes a bill.
        warn_at_fraction: Fraction of the ceiling at which to emit a warning.
            Set below 1.0 so an approaching overrun is visible while there is
            still something to do about it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    per_task_usd: float = Field(default=5.0, gt=0.0)
    warn_at_fraction: float = Field(default=0.8, gt=0.0, le=1.0)

    @property
    def warn_at_usd(self) -> float:
        return self.per_task_usd * self.warn_at_fraction


class PipelineConfig(BaseModel):
    """Flow-level limits that are not model selection.

    Attributes:
        max_qa_retries: How many times a failed test run returns to the
            developer before escalating to a human. Must be finite: an
            unbounded fix-and-retest loop is a budget fire with no exit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_qa_retries: int = Field(default=3, ge=0, le=10)


class GithubConfig(BaseModel):
    """GitHub integration: delivering verified changes as real pull requests.

    The section's presence is the feature switch. This deliberately deviates
    from ``budget``/``pipeline``, which default in: a silent default here would
    flip a deployment into real-repository mode without anyone deciding to.

    Attributes:
        api_token_env: Name of the environment variable holding the
            fine-grained PAT (pull-request write on the target repositories).
            Never the token itself - the file is committed.
        api_base_url: API root; ``https://host/api/v3`` for GitHub Enterprise.
        git_host: The git-remote host matching that API root, so pull requests
            open against the host the token actually serves (the slug parser
            refuses remotes on any other host).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_token_env: str = Field(default="GH_TOKEN")
    api_base_url: str = Field(default="https://api.github.com")
    git_host: str = Field(default="github.com")

    @field_validator("api_token_env")
    @classmethod
    def _validate_env_name(cls, value: str) -> str:
        return _validate_env_var_name(value)

    @field_validator("api_base_url", "git_host")
    @classmethod
    def _strip(cls, value: str) -> str:
        return reject_blank(value)


class PlatformConfig(BaseModel):
    """The whole configuration file.

    Attributes:
        providers: Available providers, keyed by a short name the roles refer
            to. At least one is required; a config with none cannot run
            anything and should fail at load, not at first call.
        roles: Model selection per pipeline role. Every role must be present,
            enforced by a validator rather than by a default, so adding a role
            to the enum forces a deliberate choice here.
        budget: Per-task cost ceiling.
        pipeline: Flow limits.
        github: GitHub integration; ``None`` (no section) means code delivery
            is off and the pipeline runs the artifact-only topology.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Used when a provider entry does not name a mode: the one measured
    #: against Anthropic, the platform's first provider. A class attribute,
    #: not a field - it is a property of the code, not of any config file.
    DEFAULT_INSTRUCTOR_MODE: ClassVar[StructuredOutputMode] = StructuredOutputMode.JSON_SCHEMA

    providers: dict[str, ProviderConfig] = Field(min_length=1)
    roles: dict[PipelineRole, RoleModelConfig]
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    github: GithubConfig | None = None

    @model_validator(mode="after")
    def _validate_cross_references(self) -> Self:
        """Check references between sections and require full role coverage."""
        unknown = {
            role: config.provider
            for role, config in self.roles.items()
            if config.provider not in self.providers
        }
        if unknown:
            known = ", ".join(sorted(self.providers)) or "(none)"
            details = ", ".join(
                f"{role.value}->{provider!r}" for role, provider in sorted(unknown.items())
            )
            raise ValueError(
                f"role(s) reference undefined provider(s) [{details}]; defined: {known}"
            )

        missing = [role for role in PipelineRole if role not in self.roles]
        if missing:
            names = ", ".join(role.value for role in missing)
            raise ValueError(f"no model configured for role(s): {names}")

        return self

    def litellm_model(self, role: PipelineRole) -> str:
        """Resolve a role to the model string litellm expects.

        litellm routes on a ``provider/model`` string, so this is where the
        config's separate provider key and model name get joined.
        """
        role_config = self.roles[role]
        provider = self.providers[role_config.provider]
        prefix = provider.litellm_prefix or DEFAULT_LITELLM_PREFIXES.get(
            role_config.provider, role_config.provider
        )
        return f"{prefix}/{role_config.model}"

    def instructor_mode_for(self, role: PipelineRole) -> StructuredOutputMode:
        """Resolve a role to the structured-output mode its provider serves.

        Providers disagree on what they accept (see :class:`StructuredOutputMode`),
        so this is a property of the provider, resolved per role exactly like
        ``litellm_model`` - the two together are the full per-role wire shape.
        """
        provider = self.providers[self.roles[role].provider]
        return provider.instructor_mode or self.DEFAULT_INSTRUCTOR_MODE

    def resolve_api_key(self, role: PipelineRole) -> str:
        """Read the API key for a role's provider from the environment.

        Raises:
            MissingCredentialError: If the variable is unset or empty. Raised
                at call time rather than load time so a config can be validated
                in CI without every provider's key being present.
        """
        role_config = self.roles[role]
        provider = self.providers[role_config.provider]
        value = os.environ.get(provider.api_key_env, "").strip()
        if not value:
            raise MissingCredentialError(
                f"provider {role_config.provider!r} (used by role {role.value!r}) needs "
                f"environment variable {provider.api_key_env} to be set"
            )
        return value

    def resolve_github_token(self) -> str:
        """Read the GitHub token from the environment, for code delivery.

        Raises:
            MissingCredentialError: If there is no ``github`` section (the
                caller asked for a feature that is off) or the variable is
                unset or empty. Lazy exactly like :meth:`resolve_api_key`, so
                CI can validate config files without a token present.
        """
        if self.github is None:
            raise MissingCredentialError("no github section configured; code delivery is off")
        value = os.environ.get(self.github.api_token_env, "").strip()
        if not value:
            raise MissingCredentialError(
                f"github integration needs environment variable "
                f"{self.github.api_token_env} to be set"
            )
        return value


def load_config(path: Path | str) -> PlatformConfig:
    """Read and validate a YAML config file.

    Args:
        path: Location of the config file.

    Returns:
        A validated, immutable configuration.

    Raises:
        FileNotFoundError: If the path does not exist. The message names the
            path, since a wrong working directory is the usual cause.
        ValueError: If the file is not valid YAML, is not a mapping, or fails
            schema validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"config file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ValueError(
            f"config file must contain a mapping at the top level, got {type(raw).__name__}"
        )

    return PlatformConfig.model_validate(raw)

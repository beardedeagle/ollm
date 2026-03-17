"""Boundary-layer settings models and precedence helpers for oLLM."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ollm.runtime.config import (
    DEFAULT_DEVICE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_REFERENCE,
    GenerationConfig,
    RuntimeConfig,
    normalize_backend,
)

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000


class SettingsPrecedenceLayer(StrEnum):
    """Named layers for the canonical settings-precedence contract."""

    CLI = "cli"
    ENVIRONMENT = "env"
    CONFIG_FILE = "config_file"
    DEFAULTS = "defaults"


SETTINGS_PRECEDENCE = (
    SettingsPrecedenceLayer.CLI,
    SettingsPrecedenceLayer.ENVIRONMENT,
    SettingsPrecedenceLayer.CONFIG_FILE,
    SettingsPrecedenceLayer.DEFAULTS,
)


class RuntimeSettings(BaseModel):
    """Default runtime settings resolved before execution-specific overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_reference: str = DEFAULT_MODEL_REFERENCE
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    device: str = DEFAULT_DEVICE
    backend: str | None = None
    adapter_dir: Path | None = None
    multimodal: bool = False
    use_specialization: bool = True
    cache_dir: Path = Field(default_factory=lambda: Path("kv_cache"))
    use_cache: bool = True
    offload_cpu_layers: int = Field(default=0, ge=0)
    offload_gpu_layers: int = Field(default=0, ge=0)
    force_download: bool = False
    stats: bool = False
    verbose: bool = False
    quiet: bool = False

    @field_validator("backend")
    @classmethod
    def _normalize_backend(cls, backend: str | None) -> str | None:
        return normalize_backend(backend)


class GenerationSettings(BaseModel):
    """Default generation settings resolved before request-level overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stream: bool = True


class ServerSettings(BaseModel):
    """Default headless-server settings for future server surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = DEFAULT_SERVER_HOST
    port: int = Field(default=DEFAULT_SERVER_PORT, gt=0, le=65535)
    reload: bool = False
    log_level: str = "info"


class AppSettings(BaseSettings):
    """Top-level application settings schema for CLI and future server surfaces."""

    model_config = SettingsConfigDict(
        env_prefix="OLLM_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)


class RuntimeConfigOverrides(BaseModel):
    """Explicit runtime overrides layered on top of runtime settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_reference: str | None = None
    models_dir: Path | None = None
    device: str | None = None
    backend: str | None = None
    adapter_dir: Path | None = None
    multimodal: bool | None = None
    use_specialization: bool | None = None
    cache_dir: Path | None = None
    use_cache: bool | None = None
    offload_cpu_layers: int | None = Field(default=None, ge=0)
    offload_gpu_layers: int | None = Field(default=None, ge=0)
    force_download: bool | None = None
    stats: bool | None = None
    verbose: bool | None = None
    quiet: bool | None = None

    @field_validator("backend")
    @classmethod
    def _normalize_backend(cls, backend: str | None) -> str | None:
        return normalize_backend(backend)


class GenerationConfigOverrides(BaseModel):
    """Explicit generation overrides layered on top of generation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stream: bool | None = None


class ServerSettingsOverrides(BaseModel):
    """Explicit server overrides layered on top of server settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str | None = None
    port: int | None = Field(default=None, gt=0, le=65535)
    reload: bool | None = None
    log_level: str | None = None


@lru_cache(maxsize=1)
def default_app_settings() -> AppSettings:
    """Return built-in application settings without loading external sources.

    Returns:
        AppSettings: The canonical built-in defaults for runtime, generation,
            and server settings.
    """
    return AppSettings.model_validate(
        {
            "runtime": RuntimeSettings().model_dump(mode="python"),
            "generation": GenerationSettings().model_dump(mode="python"),
            "server": ServerSettings().model_dump(mode="python"),
        }
    )


def resolve_runtime_config(
    defaults: RuntimeSettings,
    overrides: RuntimeConfigOverrides | None = None,
) -> RuntimeConfig:
    """Materialize a runtime execution config from defaults plus explicit overrides.

    Args:
        defaults: The baseline runtime settings for the current application
            surface.
        overrides: Explicit higher-precedence runtime overrides.

    Returns:
        RuntimeConfig: A validated execution-ready runtime config.
    """
    override_values = (
        {}
        if overrides is None
        else overrides.model_dump(exclude_none=True, mode="python")
    )
    config = RuntimeConfig(
        **{
            **defaults.model_dump(mode="python"),
            **override_values,
        }
    )
    config.validate()
    return config


def resolve_generation_config(
    defaults: GenerationSettings,
    overrides: GenerationConfigOverrides | None = None,
) -> GenerationConfig:
    """Materialize a generation config from defaults plus explicit overrides.

    Args:
        defaults: The baseline generation settings for the current application
            surface.
        overrides: Explicit higher-precedence generation overrides.

    Returns:
        GenerationConfig: A validated execution-ready generation config.
    """
    override_values = (
        {}
        if overrides is None
        else overrides.model_dump(exclude_none=True, mode="python")
    )
    config = GenerationConfig(
        **{
            **defaults.model_dump(mode="python"),
            **override_values,
        }
    )
    config.validate()
    return config


def resolve_server_settings(
    defaults: ServerSettings,
    overrides: ServerSettingsOverrides | None = None,
) -> ServerSettings:
    """Materialize server settings from defaults plus explicit overrides.

    Args:
        defaults: The baseline server settings for the current application
            surface.
        overrides: Explicit higher-precedence server overrides.

    Returns:
        ServerSettings: A validated server settings model.
    """
    override_values = (
        {}
        if overrides is None
        else overrides.model_dump(exclude_none=True, mode="python")
    )
    return ServerSettings.model_validate(
        {
            **defaults.model_dump(mode="python"),
            **override_values,
        }
    )

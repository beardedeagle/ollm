"""Settings loading and override materialization helpers for oLLM."""

from functools import lru_cache
from pathlib import Path

from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.settings_schema import (
    DEFAULT_SETTINGS_FILE,
    AppSettings,
    GenerationConfigOverrides,
    GenerationSettings,
    RuntimeConfigOverrides,
    RuntimeSettings,
    ServerSettings,
    ServerSettingsOverrides,
)
from ollm.runtime.settings_sources import load_settings_payload


@lru_cache(maxsize=1)
def default_app_settings() -> AppSettings:
    """Return built-in application settings without external sources.

    Returns:
        AppSettings: Canonical built-in defaults for runtime, generation, and
            server settings.
    """
    return AppSettings.model_validate(
        {
            "runtime": RuntimeSettings().model_dump(mode="python"),
            "generation": GenerationSettings().model_dump(mode="python"),
            "server": ServerSettings().model_dump(mode="python"),
        }
    )


def load_app_settings(config_file: Path | None = None) -> AppSettings:
    """Load application settings from TOML and environment sources.

    Args:
        config_file (Path | None): Optional explicit config-file path.

    Returns:
        AppSettings: Merged application settings following the canonical
            precedence contract.
    """
    merged_settings = load_settings_payload(
        app_settings_cls=AppSettings,
        default_settings_file=DEFAULT_SETTINGS_FILE,
        config_file=config_file,
        default_payload=default_app_settings().model_dump(mode="python"),
    )
    return AppSettings.model_validate(merged_settings)


def resolve_runtime_config(
    defaults: RuntimeSettings,
    overrides: RuntimeConfigOverrides | None = None,
) -> RuntimeConfig:
    """Materialize a runtime config from defaults plus explicit overrides.

    Args:
        defaults (RuntimeSettings): Baseline runtime settings.
        overrides (RuntimeConfigOverrides | None): Higher-precedence overrides.

    Returns:
        RuntimeConfig: Validated execution-ready runtime config.
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
        defaults (GenerationSettings): Baseline generation settings.
        overrides (GenerationConfigOverrides | None): Higher-precedence
            generation overrides.

    Returns:
        GenerationConfig: Validated execution-ready generation config.
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
        defaults (ServerSettings): Baseline server settings.
        overrides (ServerSettingsOverrides | None): Higher-precedence server
            overrides.

    Returns:
        ServerSettings: Validated server settings.
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

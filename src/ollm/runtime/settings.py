"""Compatibility re-export surface for oLLM settings models and helpers."""

from ollm.runtime.settings_resolution import (
    default_app_settings,
    load_app_settings,
    resolve_generation_config,
    resolve_runtime_config,
    resolve_server_settings,
)
from ollm.runtime.settings_schema import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_SETTINGS_FILE,
    SETTINGS_PRECEDENCE,
    AppSettings,
    BenchmarkSettings,
    GenerationConfigOverrides,
    GenerationSettings,
    RuntimeConfigOverrides,
    RuntimeSettings,
    ServerSettings,
    ServerSettingsOverrides,
    SettingsPrecedenceLayer,
)

__all__ = [
    "AppSettings",
    "BenchmarkSettings",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "DEFAULT_SETTINGS_FILE",
    "GenerationConfigOverrides",
    "GenerationSettings",
    "RuntimeConfigOverrides",
    "RuntimeSettings",
    "SETTINGS_PRECEDENCE",
    "ServerSettings",
    "ServerSettingsOverrides",
    "SettingsPrecedenceLayer",
    "default_app_settings",
    "load_app_settings",
    "resolve_generation_config",
    "resolve_runtime_config",
    "resolve_server_settings",
]

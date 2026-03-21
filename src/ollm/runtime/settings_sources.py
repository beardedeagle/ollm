"""Settings-source helper functions shared by runtime settings loading."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic_settings import (
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)


def load_settings_payload(
    *,
    app_settings_cls,
    default_settings_file: Path,
    config_file: Path | None,
    default_payload: dict[str, object],
) -> dict[str, object]:
    """Merge default, TOML, and environment payloads for app settings."""

    merged_settings = dict(default_payload)
    merged_settings = _merge_settings_payload(
        merged_settings,
        _load_toml_settings_payload(
            app_settings_cls=app_settings_cls,
            default_settings_file=default_settings_file,
            config_file=config_file,
        ),
    )
    merged_settings = _merge_settings_payload(
        merged_settings,
        _load_env_settings_payload(app_settings_cls),
    )
    return merged_settings


def _resolve_settings_file(
    *,
    default_settings_file: Path,
    config_file: Path | None,
) -> tuple[Path, bool]:
    if config_file is not None:
        return config_file.expanduser().resolve(), True
    env_config_file = os.getenv("OLLM_CONFIG_FILE")
    if env_config_file:
        return Path(env_config_file).expanduser().resolve(), True
    return default_settings_file.expanduser().resolve(), False


def _settings_source_payload(
    source: PydanticBaseSettingsSource,
) -> dict[str, object]:
    return {key: value for key, value in source().items()}


def _merge_settings_payload(
    base: dict[str, object],
    overlay: Mapping[str, object],
) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _merge_settings_payload(
                cast(dict[str, object], existing),
                cast(Mapping[str, object], value),
            )
            continue
        merged[key] = value
    return merged


def _load_env_settings_payload(app_settings_cls) -> dict[str, object]:
    return _settings_source_payload(EnvSettingsSource(app_settings_cls))


def _load_toml_settings_payload(
    *,
    app_settings_cls,
    default_settings_file: Path,
    config_file: Path | None,
) -> dict[str, object]:
    resolved_config_file, config_file_is_explicit = _resolve_settings_file(
        default_settings_file=default_settings_file,
        config_file=config_file,
    )
    if not resolved_config_file.exists():
        if config_file_is_explicit:
            raise ValueError(f"Settings file '{resolved_config_file}' does not exist")
        return {}
    if not resolved_config_file.is_file():
        raise ValueError(
            f"Settings file '{resolved_config_file}' is not a regular file"
        )
    return _settings_source_payload(
        TomlConfigSettingsSource(app_settings_cls, toml_file=resolved_config_file)
    )

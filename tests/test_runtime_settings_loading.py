from pathlib import Path

from ollm.runtime.settings import (
    DEFAULT_SETTINGS_FILE,
    default_app_settings,
    load_app_settings,
)


def test_default_app_settings_ignore_ambient_env_sources(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OLLM_RUNTIME__MODEL_REFERENCE", "env-model")
    monkeypatch.setenv("OLLM_CONFIG_FILE", str(tmp_path / DEFAULT_SETTINGS_FILE))
    default_app_settings.cache_clear()

    settings = default_app_settings()

    assert settings.runtime.model_reference == "llama3-1B-chat"


def test_load_app_settings_honors_env_config_file_override(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "custom-ollm.toml"
    config_path.write_text(
        "\n".join(
            [
                "[runtime]",
                'model_reference = "file-model"',
                "",
                "[generation]",
                "max_new_tokens = 64",
                "",
                "[server]",
                "port = 9001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLLM_CONFIG_FILE", str(config_path))

    settings = load_app_settings()

    assert settings.runtime.model_reference == "file-model"
    assert settings.generation.max_new_tokens == 64
    assert settings.server.port == 9001


def test_load_app_settings_rejects_missing_explicit_env_config_file(
    monkeypatch, tmp_path: Path
) -> None:
    missing_path = tmp_path / "missing-ollm.toml"
    monkeypatch.setenv("OLLM_CONFIG_FILE", str(missing_path))

    try:
        load_app_settings()
    except ValueError as exc:
        assert str(missing_path) in str(exc)
        assert "does not exist" in str(exc)
    else:
        raise AssertionError(
            "load_app_settings should reject a missing explicit config"
        )

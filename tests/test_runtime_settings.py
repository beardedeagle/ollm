from pathlib import Path

from ollm.runtime.config import DEFAULT_DEVICE, DEFAULT_MAX_NEW_TOKENS
from ollm.runtime.settings import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    SETTINGS_PRECEDENCE,
    AppSettings,
    GenerationConfigOverrides,
    GenerationSettings,
    RuntimeConfigOverrides,
    RuntimeSettings,
    SettingsPrecedenceLayer,
    default_app_settings,
    resolve_generation_config,
    resolve_runtime_config,
)


def test_default_app_settings_match_current_runtime_defaults() -> None:
    default_app_settings.cache_clear()
    settings = default_app_settings()

    assert isinstance(settings, AppSettings)
    assert settings.runtime.model_reference == "llama3-1B-chat"
    assert settings.runtime.models_dir == Path("models")
    assert settings.runtime.device == DEFAULT_DEVICE
    assert settings.runtime.cache_dir == Path("kv_cache")
    assert settings.runtime.use_specialization is True
    assert settings.runtime.use_cache is True
    assert settings.generation.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert settings.generation.stream is True
    assert settings.server.host == DEFAULT_SERVER_HOST
    assert settings.server.port == DEFAULT_SERVER_PORT


def test_default_app_settings_ignore_ambient_env_sources(monkeypatch) -> None:
    monkeypatch.setenv("OLLM_RUNTIME__MODEL_REFERENCE", "env-model")
    default_app_settings.cache_clear()

    settings = default_app_settings()

    assert settings.runtime.model_reference == "llama3-1B-chat"


def test_settings_precedence_contract_is_explicit() -> None:
    assert SETTINGS_PRECEDENCE == (
        SettingsPrecedenceLayer.CLI,
        SettingsPrecedenceLayer.ENVIRONMENT,
        SettingsPrecedenceLayer.CONFIG_FILE,
        SettingsPrecedenceLayer.DEFAULTS,
    )


def test_resolve_runtime_config_prefers_explicit_overrides() -> None:
    defaults = RuntimeSettings(
        model_reference="default-model",
        models_dir=Path("/tmp/default-models"),
        device="cpu",
        cache_dir=Path("/tmp/default-cache"),
        use_cache=True,
        verbose=False,
    )

    resolved = resolve_runtime_config(
        defaults,
        RuntimeConfigOverrides(
            model_reference="override-model",
            device="mps",
            use_cache=False,
            verbose=True,
        ),
    )

    assert resolved.model_reference == "override-model"
    assert resolved.models_dir == Path("/tmp/default-models")
    assert resolved.device == "mps"
    assert resolved.cache_dir == Path("/tmp/default-cache")
    assert resolved.use_cache is False
    assert resolved.verbose is True


def test_resolve_generation_config_prefers_explicit_overrides() -> None:
    defaults = GenerationSettings(
        max_new_tokens=64,
        temperature=0.0,
        top_p=None,
        top_k=None,
        seed=None,
        stream=True,
    )

    resolved = resolve_generation_config(
        defaults,
        GenerationConfigOverrides(
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            stream=False,
        ),
    )

    assert resolved.max_new_tokens == 32
    assert resolved.temperature == 0.7
    assert resolved.top_p == 0.9
    assert resolved.stream is False


def test_runtime_settings_validate_backend_identifiers() -> None:
    try:
        RuntimeSettings(backend="bad-backend")
    except ValueError as exc:
        assert "--backend must be one of" in str(exc)
    else:
        raise AssertionError("RuntimeSettings should reject unknown backends")

    try:
        RuntimeConfigOverrides(backend="bad-backend")
    except ValueError as exc:
        assert "--backend must be one of" in str(exc)
    else:
        raise AssertionError("RuntimeConfigOverrides should reject unknown backends")

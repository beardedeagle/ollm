from pathlib import Path

from ollm.runtime.config import DEFAULT_DEVICE, DEFAULT_MAX_NEW_TOKENS
from ollm.runtime.settings import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_SETTINGS_FILE,
    SETTINGS_PRECEDENCE,
    AppSettings,
    GenerationConfigOverrides,
    GenerationSettings,
    RuntimeConfigOverrides,
    RuntimeSettings,
    SettingsPrecedenceLayer,
    default_app_settings,
    load_app_settings,
    resolve_generation_config,
    resolve_runtime_config,
)


def _write_settings_file(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[runtime]",
                'model_reference = "file-model"',
                'models_dir = "file-models"',
                'device = "cpu"',
                'backend = "transformers-generic"',
                'cache_dir = "file-cache"',
                "use_cache = false",
                'kv_cache_strategy = "streamed-segmented"',
                "",
                "[generation]",
                "max_new_tokens = 64",
                "temperature = 0.25",
                "stream = false",
                "",
                "[server]",
                'host = "0.0.0.0"',
                "port = 9001",
                "reload = true",
            ]
        )
        + "\n",
        encoding="utf-8",
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
    assert settings.runtime.kv_cache_strategy == "chunked"
    assert settings.runtime.kv_cache_lifecycle == "runtime-scoped"
    assert settings.runtime.kv_cache_adaptation_mode == "observe-only"
    assert settings.runtime.kv_cache_window_tokens is None
    assert settings.generation.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert settings.generation.stream is True
    assert settings.server.host == DEFAULT_SERVER_HOST
    assert settings.server.port == DEFAULT_SERVER_PORT


def test_default_app_settings_ignore_ambient_env_sources(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OLLM_RUNTIME__MODEL_REFERENCE", "env-model")
    monkeypatch.setenv("OLLM_CONFIG_FILE", str(tmp_path / DEFAULT_SETTINGS_FILE))
    default_app_settings.cache_clear()

    settings = default_app_settings()

    assert settings.runtime.model_reference == "llama3-1B-chat"


def test_load_app_settings_reads_toml_defaults_from_explicit_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "ollm.toml"
    _write_settings_file(config_path)

    settings = load_app_settings(config_file=config_path)

    assert settings.runtime.model_reference == "file-model"
    assert settings.runtime.models_dir == Path("file-models")
    assert settings.runtime.device == "cpu"
    assert settings.runtime.backend == "transformers-generic"
    assert settings.runtime.cache_dir == Path("file-cache")
    assert settings.runtime.use_cache is False
    assert settings.runtime.kv_cache_strategy == "streamed-segmented"
    assert settings.runtime.kv_cache_lifecycle == "runtime-scoped"
    assert settings.runtime.kv_cache_adaptation_mode == "observe-only"
    assert settings.runtime.kv_cache_window_tokens is None
    assert settings.generation.max_new_tokens == 64
    assert settings.generation.temperature == 0.25
    assert settings.generation.stream is False
    assert settings.server.host == "0.0.0.0"
    assert settings.server.port == 9001
    assert settings.server.reload is True


def test_load_app_settings_reads_nested_environment_sources(monkeypatch) -> None:
    monkeypatch.setenv("OLLM_RUNTIME__MODEL_REFERENCE", "env-model")
    monkeypatch.setenv("OLLM_RUNTIME__DEVICE", "mps")
    monkeypatch.setenv("OLLM_GENERATION__MAX_NEW_TOKENS", "42")
    monkeypatch.setenv("OLLM_GENERATION__STREAM", "false")
    monkeypatch.setenv("OLLM_SERVER__PORT", "8123")

    settings = load_app_settings()

    assert settings.runtime.model_reference == "env-model"
    assert settings.runtime.device == "mps"
    assert settings.generation.max_new_tokens == 42
    assert settings.generation.stream is False
    assert settings.server.port == 8123


def test_load_app_settings_environment_overrides_config_file(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "ollm.toml"
    _write_settings_file(config_path)
    monkeypatch.setenv("OLLM_RUNTIME__MODEL_REFERENCE", "env-model")
    monkeypatch.setenv("OLLM_GENERATION__MAX_NEW_TOKENS", "11")

    settings = load_app_settings(config_file=config_path)

    assert settings.runtime.model_reference == "env-model"
    assert settings.generation.max_new_tokens == 11
    assert settings.runtime.device == "cpu"
    assert settings.server.port == 9001


def test_load_app_settings_honors_env_config_file_override(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "custom-ollm.toml"
    _write_settings_file(config_path)
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
        kv_cache_strategy="chunked",
        kv_cache_lifecycle="runtime-scoped",
        kv_cache_adaptation_mode="observe-only",
        verbose=False,
    )

    resolved = resolve_runtime_config(
        defaults,
        RuntimeConfigOverrides(
            model_reference="override-model",
            device="mps",
            use_cache=False,
            kv_cache_strategy="streamed-segmented",
            kv_cache_lifecycle="persistent",
            kv_cache_adaptation_mode="automatic",
            verbose=True,
        ),
    )

    assert resolved.model_reference == "override-model"
    assert resolved.models_dir == Path("/tmp/default-models")
    assert resolved.device == "mps"
    assert resolved.cache_dir == Path("/tmp/default-cache")
    assert resolved.use_cache is False
    assert resolved.kv_cache_strategy == "streamed-segmented"
    assert resolved.kv_cache_lifecycle == "persistent"
    assert resolved.kv_cache_adaptation_mode == "automatic"
    assert resolved.verbose is True


def test_resolve_runtime_config_cli_overrides_beat_loaded_settings(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "ollm.toml"
    _write_settings_file(config_path)
    settings = load_app_settings(config_file=config_path)

    resolved = resolve_runtime_config(
        settings.runtime,
        RuntimeConfigOverrides(
            model_reference="cli-model",
            device="mps",
            use_cache=True,
            kv_cache_strategy="chunked",
            kv_cache_lifecycle="persistent",
            kv_cache_adaptation_mode="disabled",
        ),
    )

    assert resolved.model_reference == "cli-model"
    assert resolved.device == "mps"
    assert resolved.use_cache is True
    assert resolved.kv_cache_strategy == "chunked"
    assert resolved.kv_cache_lifecycle == "persistent"
    assert resolved.kv_cache_adaptation_mode == "disabled"
    assert resolved.backend == "transformers-generic"


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


def test_load_app_settings_does_not_leak_between_calls(monkeypatch) -> None:
    monkeypatch.setenv("OLLM_RUNTIME__MODEL_REFERENCE", "first-model")
    first = load_app_settings()
    monkeypatch.delenv("OLLM_RUNTIME__MODEL_REFERENCE")

    second = load_app_settings()

    assert first.runtime.model_reference == "first-model"
    assert second.runtime.model_reference == "llama3-1B-chat"


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


def test_runtime_settings_accept_tiered_write_back_strategy() -> None:
    settings = RuntimeSettings(kv_cache_strategy="tiered-write-back")
    overrides = RuntimeConfigOverrides(kv_cache_strategy="tiered-write-back")

    assert settings.kv_cache_strategy == "tiered-write-back"
    assert overrides.kv_cache_strategy == "tiered-write-back"


def test_runtime_settings_accept_log_structured_journal_strategy() -> None:
    settings = RuntimeSettings(kv_cache_strategy="log-structured-journal")
    overrides = RuntimeConfigOverrides(kv_cache_strategy="log-structured-journal")

    assert settings.kv_cache_strategy == "log-structured-journal"
    assert overrides.kv_cache_strategy == "log-structured-journal"


def test_runtime_settings_accept_quantized_cold_tier_strategy() -> None:
    settings = RuntimeSettings(kv_cache_strategy="quantized-cold-tier")
    overrides = RuntimeConfigOverrides(kv_cache_strategy="quantized-cold-tier")

    assert settings.kv_cache_strategy == "quantized-cold-tier"
    assert overrides.kv_cache_strategy == "quantized-cold-tier"


def test_runtime_settings_accept_sliding_window_strategy_and_tokens() -> None:
    settings = RuntimeSettings(
        kv_cache_strategy="sliding-window-ring-buffer",
        kv_cache_window_tokens=96,
    )
    overrides = RuntimeConfigOverrides(
        kv_cache_strategy="sliding-window-ring-buffer",
        kv_cache_window_tokens=96,
    )

    assert settings.kv_cache_strategy == "sliding-window-ring-buffer"
    assert settings.kv_cache_window_tokens == 96
    assert overrides.kv_cache_strategy == "sliding-window-ring-buffer"
    assert overrides.kv_cache_window_tokens == 96


def test_runtime_settings_reject_window_tokens_for_full_history_strategy() -> None:
    try:
        RuntimeSettings(kv_cache_strategy="chunked", kv_cache_window_tokens=64)
    except ValueError as exc:
        assert "--kv-cache-window-tokens requires --kv-cache-strategy" in str(exc)
    else:
        raise AssertionError(
            "Full-history runtime settings should reject explicit window tokens"
        )


def test_runtime_settings_accept_sliding_window_strategy_with_window_tokens() -> None:
    settings = RuntimeSettings(
        kv_cache_strategy="sliding-window-ring-buffer",
        kv_cache_window_tokens=48,
    )
    overrides = RuntimeConfigOverrides(
        kv_cache_strategy="sliding-window-ring-buffer",
        kv_cache_window_tokens=48,
    )

    assert settings.kv_cache_strategy == "sliding-window-ring-buffer"
    assert settings.kv_cache_window_tokens == 48
    assert overrides.kv_cache_strategy == "sliding-window-ring-buffer"
    assert overrides.kv_cache_window_tokens == 48


def test_runtime_settings_reject_window_tokens_without_sliding_strategy() -> None:
    try:
        RuntimeSettings(kv_cache_strategy="chunked", kv_cache_window_tokens=48)
    except ValueError as exc:
        assert "--kv-cache-window-tokens requires" in str(exc)
    else:
        raise AssertionError(
            "RuntimeSettings should reject window tokens without the sliding strategy"
        )

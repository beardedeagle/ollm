from pathlib import Path

from ollm.runtime.backends.openai_compatible import (
    OpenAICompatibleBackend,
    _resolve_provider_endpoint,
)
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.loader import RuntimeLoader

from tests.openai_compatible_server import OpenAICompatibleFixtureServer


def test_openai_compatible_backend_requires_explicit_endpoint(tmp_path: Path) -> None:
    loader = RuntimeLoader(backends=(OpenAICompatibleBackend(),))
    runtime_plan = loader.plan(
        RuntimeConfig(
            model_reference="openai-compatible:gpt-4.1-mini",
            models_dir=tmp_path / "models",
            device="cpu",
        )
    )

    assert runtime_plan.is_executable() is False
    assert "--provider-endpoint" in runtime_plan.reason


def test_openai_compatible_backend_plans_executable_lmstudio_reference(
    tmp_path: Path,
) -> None:
    server = OpenAICompatibleFixtureServer(
        models={"local-model": {"response_text": "hello"}}
    )
    server.start()
    try:
        loader = RuntimeLoader(backends=(OpenAICompatibleBackend(),))
        runtime_plan = loader.plan(
            RuntimeConfig(
                model_reference="lmstudio:local-model",
                models_dir=tmp_path / "models",
                device="cpu",
                provider_endpoint=server.base_url,
            )
        )
    finally:
        server.stop()

    assert runtime_plan.is_executable() is True
    assert runtime_plan.backend_id == "openai-compatible"
    assert runtime_plan.support_level.value == "provider-backed"
    assert runtime_plan.details["provider_endpoint"] == server.base_url


def test_lmstudio_endpoint_resolution_defaults_without_override() -> None:
    config = RuntimeConfig(
        model_reference="lmstudio:demo", models_dir=Path("models"), device="cpu"
    )

    assert _resolve_provider_endpoint("lmstudio", config) == "http://127.0.0.1:1234/v1"


def test_runtime_loader_discovers_lmstudio_models(tmp_path: Path) -> None:
    server = OpenAICompatibleFixtureServer(
        models={"local-model": {"response_text": "hello"}}
    )
    server.start()
    try:
        loader = RuntimeLoader(backends=(OpenAICompatibleBackend(),))
        discovered_models = loader.discover_provider_models(
            tmp_path / "models",
            ("lmstudio",),
            server.base_url,
            strict=True,
        )
    finally:
        server.stop()

    assert [item.model_reference for item in discovered_models] == [
        "lmstudio:local-model"
    ]
    assert discovered_models[0].provider_endpoint == server.base_url


def test_runtime_loader_requires_endpoint_for_openai_compatible_discovery(
    tmp_path: Path,
) -> None:
    loader = RuntimeLoader(backends=(OpenAICompatibleBackend(),))

    try:
        loader.discover_provider_models(
            tmp_path / "models",
            ("openai-compatible",),
            strict=True,
        )
        raise AssertionError(
            "Expected openai-compatible discovery without endpoint to fail"
        )
    except ValueError as exc:
        assert "--provider-endpoint" in str(exc)


def test_openai_compatible_backend_reports_audio_capability_for_generic_provider(
    tmp_path: Path,
) -> None:
    server = OpenAICompatibleFixtureServer(
        models={"audio-model": {"response_text": "hello"}}
    )
    server.start()
    try:
        loader = RuntimeLoader(backends=(OpenAICompatibleBackend(),))
        runtime_plan = loader.plan(
            RuntimeConfig(
                model_reference="openai-compatible:audio-model",
                models_dir=tmp_path / "models",
                device="cpu",
                provider_endpoint=server.base_url,
            )
        )
    finally:
        server.stop()

    assert runtime_plan.is_executable() is True
    assert [
        item.value for item in runtime_plan.resolved_model.capabilities.modalities
    ] == ["text"]
    assert runtime_plan.details["audio_input_support"] == "request-capable"


def test_lmstudio_backend_does_not_report_audio_capability(tmp_path: Path) -> None:
    server = OpenAICompatibleFixtureServer(
        models={"local-model": {"response_text": "hello"}}
    )
    server.start()
    try:
        loader = RuntimeLoader(backends=(OpenAICompatibleBackend(),))
        runtime_plan = loader.plan(
            RuntimeConfig(
                model_reference="lmstudio:local-model",
                models_dir=tmp_path / "models",
                device="cpu",
                provider_endpoint=server.base_url,
            )
        )
    finally:
        server.stop()

    assert runtime_plan.is_executable() is True
    assert [
        item.value for item in runtime_plan.resolved_model.capabilities.modalities
    ] == ["text"]
    assert runtime_plan.details["audio_input_support"] == "unsupported"

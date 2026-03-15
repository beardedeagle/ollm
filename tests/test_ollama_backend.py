from pathlib import Path

from ollm.runtime.backends.ollama import OllamaBackend
from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.loader import RuntimeLoader
from ollm.runtime.providers.ollama_client import OllamaClient

from tests.ollama_server import OllamaFixtureServer


def test_runtime_loader_plans_executable_ollama_reference(tmp_path: Path) -> None:
    server = OllamaFixtureServer(
        models={
            "llama3.2": {
                "capabilities": ["completion", "vision"],
                "response_text": "hello from ollama",
            }
        }
    )
    server.start()
    try:
        loader = RuntimeLoader(
            backends=(OllamaBackend(client=OllamaClient(base_url=server.base_url)),),
        )
        plan = loader.plan(
            RuntimeConfig(
                model_reference="ollama:llama3.2",
                models_dir=tmp_path / "models",
            )
        )
    finally:
        server.stop()

    assert plan.backend_id == "ollama"
    assert plan.is_executable() is True
    assert plan.support_level is SupportLevel.PROVIDER_BACKED
    assert (
        plan.resolved_model.capabilities.support_level is SupportLevel.PROVIDER_BACKED
    )
    assert plan.resolved_model.capabilities.modalities[-1].value == "image"
    assert plan.details["provider_endpoint"] == server.base_url
    assert plan.details["provider_backend"] == "ollama"


def test_runtime_loader_marks_missing_ollama_model_as_not_executable(
    tmp_path: Path,
) -> None:
    server = OllamaFixtureServer(models={})
    server.start()
    try:
        loader = RuntimeLoader(
            backends=(OllamaBackend(client=OllamaClient(base_url=server.base_url)),),
        )
        plan = loader.plan(
            RuntimeConfig(
                model_reference="ollama:missing-model",
                models_dir=tmp_path / "models",
            )
        )
    finally:
        server.stop()

    assert plan.backend_id is None
    assert plan.is_executable() is False
    assert plan.support_level is SupportLevel.PROVIDER_BACKED
    assert "not found" in plan.reason


def test_runtime_loader_discovers_ollama_models(tmp_path: Path) -> None:
    server = OllamaFixtureServer(
        models={
            "llama3.2": {"capabilities": ["completion"], "response_text": "ready"},
            "llava": {
                "capabilities": ["completion", "vision"],
                "response_text": "vision",
            },
        }
    )
    server.start()
    try:
        loader = RuntimeLoader(
            backends=(OllamaBackend(client=OllamaClient(base_url=server.base_url)),),
        )
        discovered_models = loader.discover_provider_models(
            tmp_path / "models",
            ("ollama",),
            strict=True,
        )
    finally:
        server.stop()

    assert [item.model_reference for item in discovered_models] == [
        "ollama:llama3.2",
        "ollama:llava",
    ]
    assert all(item.provider_endpoint == server.base_url for item in discovered_models)


def test_runtime_loader_plans_executable_msty_reference_with_explicit_endpoint(
    tmp_path: Path,
) -> None:
    server = OllamaFixtureServer(
        models={
            "llama3.2": {
                "capabilities": ["completion"],
                "response_text": "hello from msty",
            }
        }
    )
    server.start()
    try:
        loader = RuntimeLoader(
            backends=(
                OllamaBackend(
                    client_factory=lambda endpoint: OllamaClient(base_url=endpoint)
                ),
            ),
        )
        plan = loader.plan(
            RuntimeConfig(
                model_reference="msty:llama3.2",
                models_dir=tmp_path / "models",
                provider_endpoint=server.base_url,
            )
        )
    finally:
        server.stop()

    assert plan.backend_id == "ollama"
    assert plan.is_executable() is True
    assert plan.support_level is SupportLevel.PROVIDER_BACKED
    assert plan.details["provider_endpoint"] == server.base_url
    assert plan.details["provider_backend"] == "ollama"
    assert plan.resolved_model.capabilities.details["provider"] == "msty"


def test_runtime_loader_marks_msty_reference_non_executable_without_endpoint(
    tmp_path: Path,
) -> None:
    loader = RuntimeLoader(backends=(OllamaBackend(),))
    plan = loader.plan(
        RuntimeConfig(
            model_reference="msty:llama3.2",
            models_dir=tmp_path / "models",
        )
    )

    assert plan.backend_id is None
    assert plan.is_executable() is False
    assert plan.support_level is SupportLevel.PROVIDER_BACKED
    assert "--provider-endpoint" in plan.reason


def test_runtime_loader_discovers_msty_models_with_explicit_endpoint(
    tmp_path: Path,
) -> None:
    server = OllamaFixtureServer(
        models={
            "llama3.2": {"capabilities": ["completion"], "response_text": "ready"},
        }
    )
    server.start()
    try:
        loader = RuntimeLoader(
            backends=(
                OllamaBackend(
                    client_factory=lambda endpoint: OllamaClient(base_url=endpoint)
                ),
            ),
        )
        discovered_models = loader.discover_provider_models(
            tmp_path / "models",
            ("msty",),
            server.base_url,
            strict=True,
        )
    finally:
        server.stop()

    assert [item.model_reference for item in discovered_models] == ["msty:llama3.2"]
    assert discovered_models[0].provider_name == "msty"
    assert discovered_models[0].provider_endpoint == server.base_url

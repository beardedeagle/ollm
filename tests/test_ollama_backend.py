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
	assert plan.resolved_model.capabilities.support_level is SupportLevel.PROVIDER_BACKED
	assert plan.resolved_model.capabilities.modalities[-1].value == "image"
	assert plan.details["provider_endpoint"] == server.base_url


def test_runtime_loader_marks_missing_ollama_model_as_not_executable(tmp_path: Path) -> None:
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

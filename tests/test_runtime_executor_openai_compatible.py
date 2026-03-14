from pathlib import Path

from ollm.app.types import Message, PromptRequest
from ollm.runtime.backends.openai_compatible import OpenAICompatibleBackend
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader
from ollm.runtime.streaming import StreamSink

from tests.openai_compatible_server import OpenAICompatibleFixtureServer


class RecordingSink(StreamSink):
	def __init__(self):
		self.status_messages: list[str] = []
		self.text_chunks: list[str] = []
		self.completed_text = ""

	def on_status(self, message: str) -> None:
		self.status_messages.append(message)

	def on_text(self, text: str) -> None:
		self.text_chunks.append(text)

	def on_complete(self, text: str) -> None:
		self.completed_text = text


def test_runtime_executor_streams_openai_compatible_provider_output(tmp_path: Path) -> None:
	server = OpenAICompatibleFixtureServer(
		models={
			"local-model": {
				"response_text": "hello from compatible provider",
				"stream_chunks": ["hello ", "from ", "compatible ", "provider"],
			}
		}
	)
	server.start()
	try:
		loader = RuntimeLoader(backends=(OpenAICompatibleBackend(),))
		runtime_config = RuntimeConfig(
			model_reference="openai-compatible:local-model",
			models_dir=tmp_path / "models",
			device="cpu",
			provider_endpoint=server.base_url,
		)
		runtime = loader.load(runtime_config)
		request = PromptRequest(
			runtime_config=runtime_config,
			generation_config=GenerationConfig(
				max_new_tokens=32,
				temperature=0.2,
				top_p=0.9,
				seed=321,
				stream=True,
			),
			messages=[
				Message.system_text("You are concise."),
				Message.user_text("Say hello."),
			],
		)
		sink = RecordingSink()
		response = RuntimeExecutor().execute(runtime, request, sink=sink)
	finally:
		server.stop()

	assert response.text == "hello from compatible provider"
	assert response.metadata["backend_id"] == "openai-compatible"
	assert response.metadata["provider"] == "openai-compatible"
	assert response.metadata["provider_backend"] == "openai-compatible"
	assert response.metadata["provider_endpoint"] == server.base_url
	assert sink.text_chunks == ["hello ", "from ", "compatible ", "provider"]
	assert sink.completed_text == "hello from compatible provider"
	assert sink.status_messages == [
		"Running openai-compatible:local-model via provider backend openai-compatible"
	]
	assert server.requests[0].method == "GET"
	assert server.requests[0].path == "/v1/models"
	assert server.requests[1].method == "POST"
	assert server.requests[1].path == "/v1/chat/completions"
	assert server.requests[1].payload == {
		"model": "local-model",
		"messages": [
			{"role": "system", "content": "You are concise."},
			{"role": "user", "content": "Say hello."},
		],
		"stream": True,
		"max_tokens": 32,
		"temperature": 0.2,
		"top_p": 0.9,
		"seed": 321,
	}


def test_runtime_executor_rejects_top_k_for_openai_compatible_provider(tmp_path: Path) -> None:
	server = OpenAICompatibleFixtureServer(models={"local-model": {"response_text": "hello"}})
	server.start()
	try:
		loader = RuntimeLoader(backends=(OpenAICompatibleBackend(),))
		runtime_config = RuntimeConfig(
			model_reference="lmstudio:local-model",
			models_dir=tmp_path / "models",
			device="cpu",
			provider_endpoint=server.base_url,
		)
		runtime = loader.load(runtime_config)
		request = PromptRequest(
			runtime_config=runtime_config,
			generation_config=GenerationConfig(max_new_tokens=16, top_k=32, stream=False),
			messages=[
				Message.system_text("You are concise."),
				Message.user_text("Say hello."),
			],
		)
	except Exception:
		server.stop()
		raise

	try:
		RuntimeExecutor().execute(runtime, request)
		raise AssertionError("Expected provider execution with top-k to fail")
	except ValueError as exc:
		assert "--top-k" in str(exc)
	finally:
		server.stop()


def test_runtime_executor_preserves_lmstudio_provider_identity(tmp_path: Path) -> None:
	server = OpenAICompatibleFixtureServer(models={"local-model": {"response_text": "hello from lmstudio"}})
	server.start()
	try:
		loader = RuntimeLoader(backends=(OpenAICompatibleBackend(),))
		runtime_config = RuntimeConfig(
			model_reference="lmstudio:local-model",
			models_dir=tmp_path / "models",
			device="cpu",
			provider_endpoint=server.base_url,
		)
		runtime = loader.load(runtime_config)
		request = PromptRequest(
			runtime_config=runtime_config,
			generation_config=GenerationConfig(max_new_tokens=16, stream=False),
			messages=[
				Message.system_text("You are concise."),
				Message.user_text("Say hello."),
			],
		)
		response = RuntimeExecutor().execute(runtime, request)
	finally:
		server.stop()

	assert response.text == "hello from lmstudio"
	assert response.metadata["provider"] == "lmstudio"
	assert response.metadata["provider_backend"] == "openai-compatible"
	assert response.metadata["provider_endpoint"] == server.base_url
	assert response.metadata["backend_id"] == "openai-compatible"

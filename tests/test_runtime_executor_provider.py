import base64
from pathlib import Path

import pytest

from ollm.app.types import ContentPart, Message, MessageRole, PromptRequest
from ollm.runtime.backends.ollama import OllamaBackend
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader
from ollm.runtime.streaming import StreamSink
from ollm.runtime.providers.ollama_client import OllamaClient

from tests.media_server import MediaFixtureServer, MediaResponse
from tests.ollama_server import OllamaFixtureServer


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


def test_runtime_executor_streams_ollama_provider_output(tmp_path: Path) -> None:
	server = OllamaFixtureServer(
		models={
			"llama3.2": {
				"capabilities": ["completion"],
				"response_text": "hello from ollama",
				"stream_chunks": ["hello ", "from ", "ollama"],
			}
		}
	)
	server.start()
	try:
		loader = RuntimeLoader(
			backends=(OllamaBackend(client=OllamaClient(base_url=server.base_url)),),
		)
		runtime_config = RuntimeConfig(
			model_reference="ollama:llama3.2",
			models_dir=tmp_path / "models",
			device="cuda:0",
		)
		runtime = loader.load(runtime_config)
		request = PromptRequest(
			runtime_config=runtime_config,
			generation_config=GenerationConfig(
				max_new_tokens=32,
				temperature=0.2,
				top_p=0.9,
				top_k=40,
				seed=123,
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

	assert response.text == "hello from ollama"
	assert response.metadata["backend_id"] == "ollama"
	assert response.metadata["provider"] == "ollama"
	assert response.metadata["provider_endpoint"] == server.base_url
	assert sink.text_chunks == ["hello ", "from ", "ollama"]
	assert sink.completed_text == "hello from ollama"
	assert sink.status_messages == ["Running ollama:llama3.2 via provider backend ollama"]
	assert server.requests[-1].path == "/api/chat"
	assert server.requests[-1].payload["stream"] is True
	assert server.requests[-1].payload["options"] == {
		"num_predict": 32,
		"temperature": 0.2,
		"top_p": 0.9,
		"top_k": 40,
		"seed": 123,
	}


def test_runtime_executor_sends_base64_image_to_ollama_vision_model(tmp_path: Path) -> None:
	server = OllamaFixtureServer(
		models={
			"llava": {
				"capabilities": ["completion", "vision"],
				"response_text": "vision ready",
			}
		}
	)
	image_path = tmp_path / "diagram.png"
	image_path.write_bytes(b"png-bytes")
	server.start()
	try:
		loader = RuntimeLoader(
			backends=(OllamaBackend(client=OllamaClient(base_url=server.base_url)),),
		)
		runtime_config = RuntimeConfig(
			model_reference="ollama:llava",
			models_dir=tmp_path / "models",
			device="cpu",
			multimodal=True,
		)
		runtime = loader.load(runtime_config)
		request = PromptRequest(
			runtime_config=runtime_config,
			generation_config=GenerationConfig(max_new_tokens=16, stream=False),
			messages=[
				Message(
					role=MessageRole.USER,
					content=[ContentPart.text("Describe the image"), ContentPart.image(str(image_path))],
				)
			],
		)
		response = RuntimeExecutor().execute(runtime, request)
	finally:
		server.stop()

	assert response.text == "vision ready"
	assert server.requests[-1].payload["messages"][0]["images"] == [
		base64.b64encode(b"png-bytes").decode("ascii")
	]


def test_runtime_executor_fetches_remote_image_url_for_ollama_vision_model(tmp_path: Path) -> None:
	server = OllamaFixtureServer(
		models={
			"llava": {
				"capabilities": ["completion", "vision"],
				"response_text": "vision ready",
			}
		}
	)
	media_server = MediaFixtureServer(
		responses={
			"/diagram.png": MediaResponse(body=b"remote-png-bytes", content_type="image/png"),
		}
	)
	server.start()
	media_server.start()
	try:
		loader = RuntimeLoader(
			backends=(OllamaBackend(client=OllamaClient(base_url=server.base_url)),),
		)
		runtime_config = RuntimeConfig(
			model_reference="ollama:llava",
			models_dir=tmp_path / "models",
			device="cpu",
			multimodal=True,
		)
		runtime = loader.load(runtime_config)
		request = PromptRequest(
			runtime_config=runtime_config,
			generation_config=GenerationConfig(max_new_tokens=16, stream=False),
			messages=[
				Message(
					role=MessageRole.USER,
					content=[
						ContentPart.text("Describe the image"),
						ContentPart.image(f"{media_server.base_url}/diagram.png"),
					],
				)
			],
		)
		response = RuntimeExecutor().execute(runtime, request)
	finally:
		media_server.stop()
		server.stop()

	assert response.text == "vision ready"
	assert server.requests[-1].payload["messages"][0]["images"] == [
		base64.b64encode(b"remote-png-bytes").decode("ascii")
	]


def test_runtime_executor_rejects_non_image_remote_url_for_ollama_vision_model(tmp_path: Path) -> None:
	server = OllamaFixtureServer(
		models={
			"llava": {
				"capabilities": ["completion", "vision"],
				"response_text": "vision ready",
			}
		}
	)
	media_server = MediaFixtureServer(
		responses={
			"/not-image": MediaResponse(body=b"nope", content_type="text/plain"),
		}
	)
	server.start()
	media_server.start()
	try:
		loader = RuntimeLoader(
			backends=(OllamaBackend(client=OllamaClient(base_url=server.base_url)),),
		)
		runtime_config = RuntimeConfig(
			model_reference="ollama:llava",
			models_dir=tmp_path / "models",
			device="cpu",
			multimodal=True,
		)
		runtime = loader.load(runtime_config)
		request = PromptRequest(
			runtime_config=runtime_config,
			generation_config=GenerationConfig(max_new_tokens=16, stream=False),
			messages=[
				Message(
					role=MessageRole.USER,
					content=[
						ContentPart.text("Describe the image"),
						ContentPart.image(f"{media_server.base_url}/not-image"),
					],
				)
			],
		)
		with pytest.raises(ValueError, match="image content type"):
			RuntimeExecutor().execute(runtime, request)
	finally:
		media_server.stop()
		server.stop()

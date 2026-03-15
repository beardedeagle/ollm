import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(slots=True)
class RecordedRequest:
	method: str
	path: str
	payload: dict[str, object] | None


@dataclass(slots=True)
class OpenAICompatibleFixtureServer:
	models: dict[str, dict[str, object]]
	requests: list[RecordedRequest] = field(default_factory=list)
	_server: ThreadingHTTPServer | None = field(init=False, default=None, repr=False)
	_thread: threading.Thread | None = field(init=False, default=None, repr=False)

	@property
	def base_url(self) -> str:
		assert self._server is not None
		return f"http://127.0.0.1:{self._server.server_port}/v1"

	def start(self) -> None:
		server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
		self._server = server
		thread = threading.Thread(target=server.serve_forever, daemon=True)
		thread.start()
		self._thread = thread

	def stop(self) -> None:
		if self._server is not None:
			self._server.shutdown()
			self._server.server_close()
		if self._thread is not None:
			self._thread.join(timeout=5)

	def _handler(self):
		fixture = self

		class OpenAICompatibleHandler(BaseHTTPRequestHandler):
			def do_GET(self) -> None:  # noqa: N802
				fixture.requests.append(RecordedRequest(method="GET", path=self.path, payload=None))
				if self.path == "/v1/models":
					_write_json(
						self,
						200,
						{
							"object": "list",
							"data": [
								{"id": model_name, "object": "model", "owned_by": "local"}
								for model_name in fixture.models
							],
						},
					)
					return
				_write_json(self, 404, {"error": {"message": f"unknown path: {self.path}"}})

			def do_POST(self) -> None:  # noqa: N802
				payload = _read_json_body(self)
				fixture.requests.append(RecordedRequest(method="POST", path=self.path, payload=payload))
				if self.path == "/v1/chat/completions":
					_handle_chat(self, fixture.models, payload)
					return
				_write_json(self, 404, {"error": {"message": f"unknown path: {self.path}"}})

			def log_message(self, format: str, *args) -> None:
				del format, args

		return OpenAICompatibleHandler


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
	content_length = int(handler.headers.get("Content-Length", "0"))
	body = handler.rfile.read(content_length).decode("utf-8")
	payload = json.loads(body)
	if not isinstance(payload, dict):
		raise ValueError("Expected a JSON object request payload")
	return payload


def _handle_chat(
	handler: BaseHTTPRequestHandler,
	models: dict[str, dict[str, object]],
	payload: dict[str, object],
) -> None:
	model_name = str(payload["model"])
	model = models.get(model_name)
	if model is None:
		_write_json(handler, 404, {"error": {"message": f"model '{model_name}' not found"}})
		return
	response_text = str(model.get("response_text", "hello from openai-compatible"))
	stream = bool(payload.get("stream", False))
	if not stream:
		_write_json(
			handler,
			200,
			{
				"id": "chatcmpl-test",
				"object": "chat.completion",
				"created": 1,
				"model": model_name,
				"choices": [
					{
						"index": 0,
						"message": {"role": "assistant", "content": response_text},
						"finish_reason": "stop",
					}
				],
			},
		)
		return

	handler.send_response(200)
	handler.send_header("Content-Type", "text/event-stream")
	handler.end_headers()
	raw_chunks = model.get("stream_chunks")
	stream_chunks = raw_chunks if isinstance(raw_chunks, list) else [response_text]
	for chunk in stream_chunks:
		handler.wfile.write(
			(
				"data: "
				+ json.dumps(
					{
						"id": "chatcmpl-test",
						"object": "chat.completion.chunk",
						"created": 1,
						"model": model_name,
						"choices": [{"index": 0, "delta": {"content": str(chunk)}, "finish_reason": None}],
					}
				)
				+ "\n\n"
			).encode("utf-8")
		)
		handler.wfile.flush()
	handler.wfile.write(
		(
			"data: "
			+ json.dumps(
				{
					"id": "chatcmpl-test",
					"object": "chat.completion.chunk",
					"created": 1,
					"model": model_name,
					"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
				}
			)
			+ "\n\n"
		).encode("utf-8")
	)
	handler.wfile.write(b"data: [DONE]\n\n")
	handler.wfile.flush()


def _write_json(handler: BaseHTTPRequestHandler, status_code: int, payload: dict[str, object]) -> None:
	body = json.dumps(payload).encode("utf-8")
	handler.send_response(status_code)
	handler.send_header("Content-Type", "application/json")
	handler.send_header("Content-Length", str(len(body)))
	handler.end_headers()
	handler.wfile.write(body)

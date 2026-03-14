import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(slots=True)
class RecordedRequest:
	path: str
	payload: dict[str, object]


@dataclass(slots=True)
class OllamaFixtureServer:
	models: dict[str, dict[str, object]]
	requests: list[RecordedRequest] = field(default_factory=list)
	_server: ThreadingHTTPServer | None = field(init=False, default=None, repr=False)
	_thread: threading.Thread | None = field(init=False, default=None, repr=False)

	@property
	def base_url(self) -> str:
		assert self._server is not None
		return f"http://127.0.0.1:{self._server.server_port}"

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

		class OllamaHandler(BaseHTTPRequestHandler):
			def do_GET(self) -> None:  # noqa: N802
				fixture.requests.append(RecordedRequest(path=self.path, payload={}))
				if self.path == "/api/tags":
					_write_json(
						self,
						200,
						{
							"models": [
								{"name": model_name}
								for model_name in fixture.models
							]
						},
					)
					return
				_write_json(self, 404, {"error": f"unknown path: {self.path}"})

			def do_POST(self) -> None:  # noqa: N802
				payload = _read_json_body(self)
				fixture.requests.append(RecordedRequest(path=self.path, payload=payload))
				if self.path == "/api/show":
					_handle_show(self, fixture.models, payload)
					return
				if self.path == "/api/chat":
					_handle_chat(self, fixture.models, payload)
					return
				_write_json(self, 404, {"error": f"unknown path: {self.path}"})

			def log_message(self, format_string: str, *args: object) -> None:
				del format_string, args

		return OllamaHandler


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
	content_length = int(handler.headers.get("Content-Length", "0"))
	body = handler.rfile.read(content_length).decode("utf-8")
	payload = json.loads(body)
	if not isinstance(payload, dict):
		raise ValueError("Expected a JSON object request payload")
	return payload


def _handle_show(
	handler: BaseHTTPRequestHandler,
	models: dict[str, dict[str, object]],
	payload: dict[str, object],
) -> None:
	model_name = str(payload["model"])
	model = models.get(model_name)
	if model is None:
		_write_json(handler, 404, {"error": f"model '{model_name}' not found"})
		return
	_write_json(
		handler,
		200,
		{
			"model": model_name,
			"capabilities": model.get("capabilities", ["completion"]),
			"details": model.get(
				"details",
				{
					"family": "llama",
					"parameter_size": "7B",
					"quantization_level": "Q4_0",
				},
			),
		},
	)


def _handle_chat(
	handler: BaseHTTPRequestHandler,
	models: dict[str, dict[str, object]],
	payload: dict[str, object],
) -> None:
	model_name = str(payload["model"])
	model = models.get(model_name)
	if model is None:
		_write_json(handler, 404, {"error": f"model '{model_name}' not found"})
		return
	response_text = str(model.get("response_text", "hello from ollama"))
	stream = bool(payload.get("stream", False))
	if not stream:
		_write_json(
			handler,
			200,
			{
				"model": model_name,
				"message": {"role": "assistant", "content": response_text},
				"done": True,
				"done_reason": "stop",
				"prompt_eval_count": 3,
				"eval_count": len(response_text.split()),
			},
		)
		return

	handler.send_response(200)
	handler.send_header("Content-Type", "application/x-ndjson")
	handler.end_headers()
	for chunk in model.get("stream_chunks", [response_text]):
		handler.wfile.write(
			(
				json.dumps(
					{
						"model": model_name,
						"message": {"role": "assistant", "content": str(chunk)},
						"done": False,
					}
				)
				+ "\n"
			).encode("utf-8")
		)
		handler.wfile.flush()
	handler.wfile.write(
		(
			json.dumps(
				{
					"model": model_name,
					"message": {"role": "assistant", "content": ""},
					"done": True,
					"done_reason": "stop",
					"prompt_eval_count": 3,
					"eval_count": len(response_text.split()),
				}
			)
			+ "\n"
		).encode("utf-8")
	)
	handler.wfile.flush()


def _write_json(handler: BaseHTTPRequestHandler, status_code: int, payload: dict[str, object]) -> None:
	body = json.dumps(payload).encode("utf-8")
	handler.send_response(status_code)
	handler.send_header("Content-Type", "application/json")
	handler.send_header("Content-Length", str(len(body)))
	handler.end_headers()
	handler.wfile.write(body)

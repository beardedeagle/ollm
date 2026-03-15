import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ollm.runtime.catalog import ModelModality

DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"


class OllamaClientError(RuntimeError):
	"""Base error for Ollama client failures."""


class OllamaConnectionError(OllamaClientError):
	"""Raised when the Ollama endpoint cannot be reached."""


@dataclass(frozen=True, slots=True)
class OllamaRequestError(OllamaClientError):
	status_code: int
	message: str

	def __str__(self) -> str:
		return self.message


@dataclass(frozen=True, slots=True)
class OllamaModelDetails:
	name: str
	modalities: tuple[ModelModality, ...]
	capabilities: tuple[str, ...]
	family: str | None
	parameter_size: str | None
	quantization_level: str | None


@dataclass(frozen=True, slots=True)
class OllamaChatResult:
	text: str
	metadata: dict[str, str]


class OllamaClient:
	def __init__(self, base_url: str = DEFAULT_OLLAMA_ENDPOINT, timeout_seconds: float = 300.0):
		self._base_url = base_url.rstrip("/")
		self._timeout_seconds = timeout_seconds

	@property
	def base_url(self) -> str:
		return self._base_url

	def list_models(self) -> tuple[str, ...]:
		payload = self._request_json("/api/tags", None)
		raw_models = payload.get("models")
		if not isinstance(raw_models, list):
			raise OllamaClientError("Expected 'models' list from Ollama /api/tags response")
		model_names: list[str] = []
		for item in raw_models:
			if not isinstance(item, dict):
				continue
			model_name = cast(dict[str, object], item).get("name")
			if isinstance(model_name, str) and model_name:
				model_names.append(model_name)
		return tuple(model_names)

	def show_model(self, model_name: str) -> OllamaModelDetails:
		payload = self._request_json("/api/show", {"model": model_name})
		raw_capabilities = tuple(str(item) for item in _payload_list(payload, "capabilities"))
		modalities = [ModelModality.TEXT]
		if "vision" in raw_capabilities:
			modalities.append(ModelModality.IMAGE)
		details_payload = _payload_dict(payload, "details")
		return OllamaModelDetails(
			name=model_name,
			modalities=tuple(modalities),
			capabilities=raw_capabilities,
			family=_payload_string(details_payload, "family"),
			parameter_size=_payload_string(details_payload, "parameter_size"),
			quantization_level=_payload_string(details_payload, "quantization_level"),
		)

	def chat(
		self,
		model_name: str,
		messages: list[dict[str, object]],
		options: dict[str, object],
		stream: bool,
		on_text: Callable[[str], None] | None = None,
	) -> OllamaChatResult:
		payload: dict[str, object] = {
			"model": model_name,
			"messages": messages,
			"stream": stream,
		}
		if options:
			payload["options"] = options

		if not stream:
			response_payload = self._request_json("/api/chat", payload)
			return OllamaChatResult(
				text=_chat_text(response_payload),
				metadata=_chat_metadata(response_payload, self._base_url, model_name),
			)

		text_chunks: list[str] = []
		final_payload: dict[str, object] = {}
		for chunk in self._request_json_stream("/api/chat", payload):
			final_payload = chunk
			text = _chat_text(chunk)
			if text:
				text_chunks.append(text)
				if on_text is not None:
					on_text(text)
		return OllamaChatResult(
			text="".join(text_chunks),
			metadata=_chat_metadata(final_payload, self._base_url, model_name),
		)

	def _request_json(self, path: str, payload: dict[str, object] | None) -> dict[str, object]:
		request = Request(
			url=f"{self._base_url}{path}",
			data=None if payload is None else json.dumps(payload).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="GET" if payload is None else "POST",
		)
		try:
			with urlopen(request, timeout=self._timeout_seconds) as response:
				return _load_json_bytes(response.read())
		except HTTPError as exc:
			raise OllamaRequestError(exc.code, _ollama_error_message(exc)) from exc
		except URLError as exc:
			raise OllamaConnectionError(
				f"Failed to reach Ollama at {self._base_url}: {exc.reason}"
			) from exc

	def _request_json_stream(
		self,
		path: str,
		payload: dict[str, object],
	) -> Iterator[dict[str, object]]:
		request = Request(
			url=f"{self._base_url}{path}",
			data=json.dumps(payload).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		try:
			with urlopen(request, timeout=self._timeout_seconds) as response:
				for raw_line in response:
					line = raw_line.decode("utf-8").strip()
					if not line:
						continue
					yield _load_json_text(line)
		except HTTPError as exc:
			raise OllamaRequestError(exc.code, _ollama_error_message(exc)) from exc
		except URLError as exc:
			raise OllamaConnectionError(
				f"Failed to reach Ollama at {self._base_url}: {exc.reason}"
			) from exc


def _load_json_bytes(raw_body: bytes) -> dict[str, object]:
	return _load_json_text(raw_body.decode("utf-8"))


def _load_json_text(raw_body: str) -> dict[str, object]:
	payload = json.loads(raw_body)
	if not isinstance(payload, dict):
		raise OllamaClientError("Expected a JSON object from Ollama")
	return payload


def _ollama_error_message(exc: HTTPError) -> str:
	body = exc.read().decode("utf-8", errors="replace")
	if body:
		try:
			payload = _load_json_text(body)
		except OllamaClientError:
			return f"Ollama request failed with status {exc.code}: {body}"
		error_text = _payload_string(payload, "error")
		if error_text is not None:
			return f"Ollama request failed with status {exc.code}: {error_text}"
	return f"Ollama request failed with status {exc.code}"


def _chat_text(payload: dict[str, object]) -> str:
	message = _payload_dict(payload, "message")
	content = _payload_string(message, "content")
	return "" if content is None else content


def _chat_metadata(
	payload: dict[str, object],
	base_url: str,
	model_name: str,
) -> dict[str, str]:
	metadata = {
		"provider": "ollama",
		"provider_endpoint": base_url,
		"provider_model": model_name,
	}
	for key in (
		"done",
		"done_reason",
		"total_duration",
		"load_duration",
		"prompt_eval_count",
		"prompt_eval_duration",
		"eval_count",
		"eval_duration",
	):
		value = payload.get(key)
		if value is None:
			continue
		metadata[key] = str(value)
	return metadata


def _payload_dict(payload: dict[str, object], key: str) -> dict[str, object]:
	value = payload.get(key)
	if isinstance(value, dict):
		return cast(dict[str, object], value)
	return {}


def _payload_list(payload: dict[str, object], key: str) -> list[object]:
	value = payload.get(key)
	if isinstance(value, list):
		return cast(list[object], value)
	return []


def _payload_string(payload: dict[str, object], key: str) -> str | None:
	value = payload.get(key)
	if isinstance(value, str):
		return value
	return None

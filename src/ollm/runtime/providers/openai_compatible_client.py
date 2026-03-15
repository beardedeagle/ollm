import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_LMSTUDIO_ENDPOINT = "http://127.0.0.1:1234/v1"


class OpenAICompatibleClientError(RuntimeError):
    """Base error for OpenAI-compatible client failures."""


class OpenAICompatibleConnectionError(OpenAICompatibleClientError):
    """Raised when the provider endpoint cannot be reached."""


@dataclass(frozen=True, slots=True)
class OpenAICompatibleRequestError(OpenAICompatibleClientError):
    status_code: int
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class OpenAICompatibleChatResult:
    text: str
    metadata: dict[str, str]


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 300.0,
        api_key: str | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._api_key = api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    def list_models(self) -> tuple[str, ...]:
        payload = self._request_json("GET", "/models")
        raw_models = payload.get("data")
        if not isinstance(raw_models, list):
            raise OpenAICompatibleClientError(
                "Expected 'data' list from provider /models response"
            )
        model_ids: list[str] = []
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            model_id = cast(dict[str, object], item).get("id")
            if isinstance(model_id, str) and model_id:
                model_ids.append(model_id)
        return tuple(model_ids)

    def chat_completions(
        self,
        provider_name: str,
        model_name: str,
        messages: list[dict[str, object]],
        options: dict[str, object],
        stream: bool,
        on_text: Callable[[str], None] | None = None,
    ) -> OpenAICompatibleChatResult:
        payload: dict[str, object] = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
        }
        payload.update(options)

        if not stream:
            response_payload = self._request_json("POST", "/chat/completions", payload)
            return OpenAICompatibleChatResult(
                text=_message_text(response_payload),
                metadata=_chat_metadata(
                    response_payload, self._base_url, provider_name, model_name
                ),
            )

        text_chunks: list[str] = []
        final_payload: dict[str, object] = {}
        for chunk in self._request_event_stream("/chat/completions", payload):
            final_payload = chunk
            text = _delta_text(chunk)
            if text:
                text_chunks.append(text)
                if on_text is not None:
                    on_text(text)
        return OpenAICompatibleChatResult(
            text="".join(text_chunks),
            metadata=_chat_metadata(
                final_payload, self._base_url, provider_name, model_name
            ),
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        request = Request(
            url=f"{self._base_url}{path}",
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                return _load_json_bytes(response.read())
        except HTTPError as exc:
            raise OpenAICompatibleRequestError(
                exc.code, _request_error_message(exc)
            ) from exc
        except URLError as exc:
            raise OpenAICompatibleConnectionError(
                f"Failed to reach OpenAI-compatible provider at {self._base_url}: {exc.reason}"
            ) from exc

    def _request_event_stream(
        self,
        path: str,
        payload: dict[str, object],
    ) -> Iterator[dict[str, object]]:
        request = Request(
            url=f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    yield _load_json_text(data)
        except HTTPError as exc:
            raise OpenAICompatibleRequestError(
                exc.code, _request_error_message(exc)
            ) from exc
        except URLError as exc:
            raise OpenAICompatibleConnectionError(
                f"Failed to reach OpenAI-compatible provider at {self._base_url}: {exc.reason}"
            ) from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None and self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers


def _load_json_bytes(raw_body: bytes) -> dict[str, object]:
    return _load_json_text(raw_body.decode("utf-8"))


def _load_json_text(raw_body: str) -> dict[str, object]:
    payload = json.loads(raw_body)
    if not isinstance(payload, dict):
        raise OpenAICompatibleClientError("Expected a JSON object from provider")
    return payload


def _request_error_message(exc: HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    if not body:
        return f"OpenAI-compatible request failed with status {exc.code}"
    try:
        payload = _load_json_text(body)
    except OpenAICompatibleClientError:
        return f"OpenAI-compatible request failed with status {exc.code}: {body}"
    error_payload = payload.get("error")
    if isinstance(error_payload, dict):
        error_message = cast(dict[str, object], error_payload).get("message")
        if isinstance(error_message, str) and error_message:
            return f"OpenAI-compatible request failed with status {exc.code}: {error_message}"
    if isinstance(error_payload, str) and error_payload:
        return (
            f"OpenAI-compatible request failed with status {exc.code}: {error_payload}"
        )
    return f"OpenAI-compatible request failed with status {exc.code}"


def _chat_metadata(
    payload: dict[str, object],
    base_url: str,
    provider_name: str,
    model_name: str,
) -> dict[str, str]:
    metadata = {
        "provider": provider_name,
        "provider_backend": "openai-compatible",
        "provider_endpoint": base_url,
        "provider_model": model_name,
    }
    for key in ("id", "object", "model", "created"):
        value = payload.get(key)
        if value is None:
            continue
        metadata[key] = str(value)
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            finish_reason = cast(dict[str, object], first_choice).get("finish_reason")
            if finish_reason is not None:
                metadata["finish_reason"] = str(finish_reason)
    return metadata


def _message_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = cast(dict[str, object], first_choice).get("message")
    if not isinstance(message, dict):
        return ""
    return _content_text(cast(dict[str, object], message).get("content"))


def _delta_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    delta = cast(dict[str, object], first_choice).get("delta")
    if not isinstance(delta, dict):
        return ""
    return _content_text(cast(dict[str, object], delta).get("content"))


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        typed_item = cast(dict[str, object], item)
        if typed_item.get("type") != "text":
            continue
        text_value = typed_item.get("text")
        if isinstance(text_value, str):
            text_parts.append(text_value)
    return "".join(text_parts)

import json
from collections.abc import Iterator
from importlib import import_module
from typing import Protocol, Self, cast

import pytest

from ollm.server.runtime import create_server_app
from tests.fakes import FakeRuntimeExecutor
from tests.server_support import build_application_service

pytest.importorskip("fastapi")


class TestClientProtocol(Protocol):
    def post(self, url: str, **kwargs) -> object: ...

    def get(self, url: str, **kwargs) -> object: ...

    def stream(self, method: str, url: str, **kwargs) -> object: ...


class StreamResponseProtocol(Protocol):
    status_code: int

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type, exc, tb) -> bool | None: ...

    def iter_lines(self) -> Iterator[str]: ...


class JsonResponseProtocol(Protocol):
    status_code: int

    def json(self) -> object: ...


def _test_client(app: object) -> TestClientProtocol:
    testclient_module = import_module("fastapi.testclient")
    return cast(TestClientProtocol, testclient_module.TestClient(app))


def _json_object(response: JsonResponseProtocol) -> dict[str, object]:
    return cast(dict[str, object], response.json())


def _payload_dict(value: object) -> dict[str, object]:
    return cast(dict[str, object], value)


def _payload_list(value: object) -> list[object]:
    return cast(list[object], value)


def test_openai_chat_completions_returns_non_streaming_response() -> None:
    application_service = build_application_service()
    app = create_server_app(application_service)
    client = _test_client(app)

    response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3-1B-chat",
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "assistant", "content": "prior reply"},
                    {"role": "user", "content": "hello"},
                ],
            },
        ),
    )

    runtime_executor = cast(
        FakeRuntimeExecutor,
        application_service.runtime_client.runtime_executor,
    )
    payload = _json_object(response)
    choices = _payload_list(payload["choices"])
    first_choice = _payload_dict(choices[0])
    message = _payload_dict(first_choice["message"])
    assert response.status_code == 200
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "llama3-1B-chat"
    assert message["role"] == "assistant"
    assert message["content"] == "echo:hello"
    assert [message.role.value for message in runtime_executor.message_batches[0]] == [
        "system",
        "assistant",
        "user",
    ]


def test_openai_chat_completions_accepts_structured_text_parts() -> None:
    application_service = build_application_service()
    app = create_server_app(application_service)
    client = _test_client(app)

    response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3-1B-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "text", "text": " world"},
                        ],
                    }
                ],
            },
        ),
    )

    payload = _json_object(response)
    choices = _payload_list(payload["choices"])
    first_choice = _payload_dict(choices[0])
    message = _payload_dict(first_choice["message"])
    assert response.status_code == 200
    assert message["content"] == "echo:hello world"


def test_openai_chat_completions_streams_openai_sse_chunks() -> None:
    application_service = build_application_service()
    app = create_server_app(application_service)
    client = _test_client(app)

    with cast(
        StreamResponseProtocol,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "llama3-1B-chat",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        ),
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    payload_lines = [line for line in lines if line.startswith("data: ")]
    event_payloads = [
        _payload_dict(json.loads(line.removeprefix("data: ")))
        for line in payload_lines
        if line != "data: [DONE]"
    ]
    first_chunk = event_payloads[0]
    second_chunk = event_payloads[1]
    final_chunk = event_payloads[-1]
    first_choices = _payload_list(first_chunk["choices"])
    second_choices = _payload_list(second_chunk["choices"])
    final_choices = _payload_list(final_chunk["choices"])
    first_delta = _payload_dict(_payload_dict(first_choices[0])["delta"])
    second_delta = _payload_dict(_payload_dict(second_choices[0])["delta"])
    final_choice = _payload_dict(final_choices[0])

    assert response.status_code == 200
    assert payload_lines[-1] == "data: [DONE]"
    assert first_chunk["object"] == "chat.completion.chunk"
    assert first_delta["role"] == "assistant"
    assert second_delta["content"] == "echo:hello"
    assert final_choice["finish_reason"] == "stop"


def test_openai_chat_completions_reports_invalid_request_with_error_envelope() -> None:
    app = create_server_app(build_application_service())
    client = _test_client(app)

    response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3-1B-chat",
                "messages": [{"role": "assistant", "content": "hello"}],
            },
        ),
    )

    payload = _json_object(response)
    error = _payload_dict(payload["error"])
    assert response.status_code == 400
    assert error["type"] == "invalid_request_error"
    assert "final message role" in cast(str, error["message"])


def test_openai_chat_completions_rejects_unsupported_content_parts_cleanly() -> None:
    app = create_server_app(build_application_service())
    client = _test_client(app)

    response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3-1B-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                    }
                ],
            },
        ),
    )

    payload = _json_object(response)
    error = _payload_dict(payload["error"])
    assert response.status_code == 400
    assert error["type"] == "invalid_request_error"
    assert "only text content parts" in cast(str, error["message"])


def test_openai_models_routes_use_openai_shapes() -> None:
    app = create_server_app(build_application_service())
    client = _test_client(app)

    list_response = cast(JsonResponseProtocol, client.get("/v1/models"))
    detail_response = cast(
        JsonResponseProtocol, client.get("/v1/models/llama3-1B-chat")
    )

    list_payload = _json_object(list_response)
    detail_payload = _json_object(detail_response)
    model_entries = _payload_list(list_payload["data"])
    first_entry = _payload_dict(model_entries[0])
    assert list_response.status_code == 200
    assert list_payload["object"] == "list"
    assert model_entries
    assert first_entry["object"] == "model"
    assert detail_response.status_code == 200
    assert detail_payload["object"] == "model"
    assert detail_payload["id"] == "llama3-1B-chat"

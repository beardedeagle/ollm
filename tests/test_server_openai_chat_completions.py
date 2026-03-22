import json
from typing import cast

import pytest

from ollm.server.runtime import create_server_app
from tests.fakes import FakeRuntimeExecutor
from tests.server_openai_support import (
    JsonResponseProtocol,
    StreamResponseProtocol,
    build_test_client,
    json_object,
    payload_dict,
    payload_list,
)
from tests.server_support import build_application_service

pytest.importorskip("fastapi")


def test_openai_chat_completions_returns_non_streaming_response() -> None:
    application_service = build_application_service()
    app = create_server_app(application_service)
    client = build_test_client(app)

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
    payload = json_object(response)
    choices = payload_list(payload["choices"])
    first_choice = payload_dict(choices[0])
    message = payload_dict(first_choice["message"])
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
    client = build_test_client(app)

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

    payload = json_object(response)
    choices = payload_list(payload["choices"])
    first_choice = payload_dict(choices[0])
    message = payload_dict(first_choice["message"])
    assert response.status_code == 200
    assert message["content"] == "echo:hello world"


def test_openai_chat_completions_streams_openai_sse_chunks() -> None:
    application_service = build_application_service()
    app = create_server_app(application_service)
    client = build_test_client(app)

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
        payload_dict(json.loads(line.removeprefix("data: ")))
        for line in payload_lines
        if line != "data: [DONE]"
    ]
    first_chunk = event_payloads[0]
    second_chunk = event_payloads[1]
    final_chunk = event_payloads[-1]
    first_choices = payload_list(first_chunk["choices"])
    second_choices = payload_list(second_chunk["choices"])
    final_choices = payload_list(final_chunk["choices"])
    first_delta = payload_dict(payload_dict(first_choices[0])["delta"])
    second_delta = payload_dict(payload_dict(second_choices[0])["delta"])
    final_choice = payload_dict(final_choices[0])

    assert response.status_code == 200
    assert payload_lines[-1] == "data: [DONE]"
    assert first_chunk["object"] == "chat.completion.chunk"
    assert first_delta["role"] == "assistant"
    assert second_delta["content"] == "echo:hello"
    assert final_choice["finish_reason"] == "stop"


def test_openai_chat_completions_reports_invalid_request_with_error_envelope() -> None:
    app = create_server_app(build_application_service())
    client = build_test_client(app)

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

    payload = json_object(response)
    error = payload_dict(payload["error"])
    assert response.status_code == 400
    assert error["type"] == "invalid_request_error"
    assert "final message role" in cast(str, error["message"])


def test_openai_chat_completions_rejects_unsupported_content_parts_cleanly() -> None:
    app = create_server_app(build_application_service())
    client = build_test_client(app)

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

    payload = json_object(response)
    error = payload_dict(payload["error"])
    assert response.status_code == 400
    assert error["type"] == "invalid_request_error"
    assert "only text content parts" in cast(str, error["message"])

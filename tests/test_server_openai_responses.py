from typing import cast

import pytest

from ollm.server.runtime import create_server_app
from tests.fakes import FakeRuntimeExecutor
from tests.server_openai_support import (
    JsonResponseProtocol,
    StreamResponseProtocol,
    build_test_client,
    configure_response_store,
    decode_stream_lines,
    json_object,
    payload_dict,
    payload_list,
)
from tests.server_support import build_application_service

pytest.importorskip("fastapi")


def test_openai_responses_accept_multimodal_input_parts(monkeypatch) -> None:
    configure_response_store(monkeypatch, backend="memory")
    application_service = build_application_service()
    app = create_server_app(application_service)
    client = build_test_client(app)

    response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/responses",
            json={
                "model": "llama3-1B-chat",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "describe"},
                            {
                                "type": "input_image",
                                "image_url": "file:///tmp/example.png",
                            },
                            {
                                "type": "input_audio",
                                "audio_url": "file:///tmp/example.wav",
                            },
                        ],
                    }
                ],
            },
        ),
    )

    runtime_executor = cast(
        FakeRuntimeExecutor,
        application_service.runtime_client.runtime_executor,
    )
    assert response.status_code == 200
    assert [
        part.kind.value for part in runtime_executor.message_batches[0][-1].content
    ] == ["text", "image", "audio"]


def test_openai_responses_create_and_retrieve_use_response_objects(
    monkeypatch,
) -> None:
    configure_response_store(monkeypatch, backend="memory")
    application_service = build_application_service()
    app = create_server_app(application_service)
    client = build_test_client(app)

    create_response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/responses",
            json={
                "model": "llama3-1B-chat",
                "instructions": "be brief",
                "input": "hello",
            },
        ),
    )

    create_payload = json_object(create_response)
    output = payload_list(create_payload["output"])
    output_message = payload_dict(output[0])
    output_content = payload_list(output_message["content"])
    output_text = payload_dict(output_content[0])
    response_id = cast(str, create_payload["id"])
    fetch_response = cast(
        JsonResponseProtocol,
        client.get(f"/v1/responses/{response_id}"),
    )
    fetch_payload = json_object(fetch_response)

    runtime_executor = cast(
        FakeRuntimeExecutor,
        application_service.runtime_client.runtime_executor,
    )
    assert create_response.status_code == 200
    assert create_payload["object"] == "response"
    assert create_payload["status"] == "completed"
    assert output_message["role"] == "assistant"
    assert output_text["type"] == "output_text"
    assert output_text["text"] == "echo:hello"
    assert [message.role.value for message in runtime_executor.message_batches[0]] == [
        "system",
        "user",
    ]
    assert fetch_response.status_code == 200
    assert fetch_payload["id"] == response_id


def test_openai_responses_support_previous_response_id_history(monkeypatch) -> None:
    configure_response_store(monkeypatch, backend="memory")
    application_service = build_application_service()
    app = create_server_app(application_service)
    client = build_test_client(app)

    first_response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/responses",
            json={"model": "llama3-1B-chat", "input": "hello"},
        ),
    )
    first_payload = json_object(first_response)
    response_id = cast(str, first_payload["id"])
    second_response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/responses",
            json={
                "model": "llama3-1B-chat",
                "previous_response_id": response_id,
                "input": "again",
            },
        ),
    )

    runtime_executor = cast(
        FakeRuntimeExecutor,
        application_service.runtime_client.runtime_executor,
    )
    second_messages = runtime_executor.message_batches[-1]
    assert second_response.status_code == 200
    assert [message.role.value for message in second_messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert second_messages[0].text_content() == "hello"
    assert second_messages[1].text_content() == "echo:hello"
    assert second_messages[2].text_content() == "again"


def test_openai_responses_stream_response_events(monkeypatch) -> None:
    configure_response_store(monkeypatch, backend="memory")
    app = create_server_app(build_application_service())
    client = build_test_client(app)

    with cast(
        StreamResponseProtocol,
        client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "llama3-1B-chat",
                "stream": True,
                "input": "hello",
            },
        ),
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    events, payloads = decode_stream_lines(lines)
    created_payload = payloads[0]
    output_item_added_payload = payloads[1]
    content_part_added_payload = payloads[2]
    delta_payload = payloads[3]
    done_payload = payloads[4]
    output_item_done_payload = payloads[5]
    completed_payload = payloads[6]

    assert response.status_code == 200
    assert events == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert created_payload["type"] == "response.created"
    assert output_item_added_payload["type"] == "response.output_item.added"
    assert content_part_added_payload["type"] == "response.content_part.added"
    assert delta_payload["type"] == "response.output_text.delta"
    assert delta_payload["delta"] == "echo:hello"
    assert done_payload["text"] == "echo:hello"
    assert output_item_done_payload["type"] == "response.output_item.done"
    assert completed_payload["type"] == "response.completed"


def test_openai_responses_report_missing_previous_response_cleanly(
    monkeypatch,
) -> None:
    configure_response_store(monkeypatch, backend="memory")
    app = create_server_app(build_application_service())
    client = build_test_client(app)

    response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/responses",
            json={
                "model": "llama3-1B-chat",
                "previous_response_id": "resp_missing",
                "input": "hello",
            },
        ),
    )

    payload = json_object(response)
    error = payload_dict(payload["error"])
    assert response.status_code == 400
    assert error["type"] == "invalid_request_error"
    assert "does not exist" in cast(str, error["message"])


def test_openai_responses_retrieval_is_disabled_by_default() -> None:
    app = create_server_app(build_application_service())
    client = build_test_client(app)

    response = cast(
        JsonResponseProtocol,
        client.get("/v1/responses/resp_missing"),
    )

    payload = json_object(response)
    error = payload_dict(payload["error"])
    assert response.status_code == 501
    assert error["code"] == "responses_storage_disabled"

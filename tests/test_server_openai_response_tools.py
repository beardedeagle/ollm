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


def test_openai_responses_return_function_call_output_items(monkeypatch) -> None:
    configure_response_store(monkeypatch, backend="memory")
    application_service = build_application_service()
    runtime_executor = cast(
        FakeRuntimeExecutor,
        application_service.runtime_client.runtime_executor,
    )
    runtime_executor.fixed_response_text = (
        '{"type":"function_call","calls":[{"name":"get_weather",'
        '"arguments":{"city":"Paris"}}]}'
    )
    app = create_server_app(application_service)
    client = build_test_client(app)

    response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/responses",
            json={
                "model": "llama3-1B-chat",
                "input": "weather?",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Look up current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            },
        ),
    )

    payload = json_object(response)
    output_items = payload_list(payload["output"])
    tool_call_item = payload_dict(output_items[0])

    assert response.status_code == 200
    assert tool_call_item["type"] == "function_call"
    assert tool_call_item["name"] == "get_weather"
    assert tool_call_item["arguments"] == '{"city":"Paris"}'
    assert payload["tool_choice"] == "auto"
    first_system_message = runtime_executor.message_batches[0][0]
    assert first_system_message.role.value == "system"
    assert "Responses API compatibility layer" in first_system_message.text_content()


def test_openai_responses_stream_function_call_events(monkeypatch) -> None:
    configure_response_store(monkeypatch, backend="memory")
    application_service = build_application_service()
    runtime_executor = cast(
        FakeRuntimeExecutor,
        application_service.runtime_client.runtime_executor,
    )
    runtime_executor.fixed_response_text = (
        '{"type":"function_call","calls":[{"name":"get_weather",'
        '"arguments":{"city":"Paris"}}]}'
    )
    app = create_server_app(application_service)
    client = build_test_client(app)

    with cast(
        StreamResponseProtocol,
        client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "llama3-1B-chat",
                "stream": True,
                "input": "weather?",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
            },
        ),
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    events, payloads = decode_stream_lines(lines)
    in_progress_payload = payloads[1]
    output_item_added_payload = payloads[2]
    arguments_delta_payload = payloads[3]
    arguments_done_payload = payloads[4]
    output_item_done_payload = payloads[5]
    completed_payload = payloads[6]

    assert response.status_code == 200
    assert events == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert in_progress_payload["type"] == "response.in_progress"
    assert payload_dict(output_item_added_payload["item"])["type"] == "function_call"
    assert payload_dict(output_item_added_payload["item"])["status"] == "in_progress"
    assert arguments_delta_payload["delta"] == '{"city":"Paris"}'
    assert arguments_done_payload["arguments"] == '{"city":"Paris"}'
    assert output_item_done_payload["type"] == "response.output_item.done"
    assert payload_dict(output_item_done_payload["item"])["status"] == "completed"
    assert completed_payload["type"] == "response.completed"
    assert [payload["sequence_number"] for payload in payloads] == list(
        range(1, len(payloads) + 1)
    )


def test_openai_responses_chain_previous_response_and_function_call_output(
    monkeypatch,
) -> None:
    configure_response_store(monkeypatch, backend="memory")
    application_service = build_application_service()
    runtime_executor = cast(
        FakeRuntimeExecutor,
        application_service.runtime_client.runtime_executor,
    )
    runtime_executor.fixed_response_text = (
        '{"type":"function_call","calls":[{"name":"get_weather",'
        '"arguments":{"city":"Paris"}}]}'
    )
    app = create_server_app(application_service)
    client = build_test_client(app)

    first_response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/responses",
            json={
                "model": "llama3-1B-chat",
                "input": "weather?",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
            },
        ),
    )
    first_payload = json_object(first_response)
    first_output = payload_list(first_payload["output"])
    first_tool_call = payload_dict(first_output[0])
    response_id = cast(str, first_payload["id"])
    call_id = cast(str, first_tool_call["call_id"])

    runtime_executor.fixed_response_text = (
        '{"type":"message","content":"It is 21C and sunny in Paris."}'
    )
    second_response = cast(
        JsonResponseProtocol,
        client.post(
            "/v1/responses",
            json={
                "model": "llama3-1B-chat",
                "previous_response_id": response_id,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": {"temperature_c": 21, "condition": "sunny"},
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
            },
        ),
    )

    second_payload = json_object(second_response)
    output = payload_list(second_payload["output"])
    output_message = payload_dict(output[0])
    output_content = payload_list(output_message["content"])
    output_text = payload_dict(output_content[0])
    second_messages = runtime_executor.message_batches[-1]

    assert second_response.status_code == 200
    assert output_message["type"] == "message"
    assert output_text["text"] == "It is 21C and sunny in Paris."
    assert [message.role.value for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "Responses API compatibility layer" in second_messages[0].text_content()
    assert "weather?" in second_messages[1].text_content()
    assert "function_call name=get_weather" in second_messages[2].text_content()
    assert "function_call_output" in second_messages[3].text_content()

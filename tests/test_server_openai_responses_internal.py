import json

import pytest

from ollm.app.types import Message
from ollm.server.openai_response_execution import (
    build_response_payload,
    build_response_prompt,
)
from ollm.server.openai_response_models import (
    OpenAIResponseFunctionCallOutputRequestModel,
    OpenAIResponseFunctionCallResponseModel,
    OpenAIResponseFunctionToolRequestModel,
)
from ollm.server.openai_response_store import (
    MemoryOpenAIResponseStore,
    StoredOpenAIResponse,
    StoredOpenAIResponseFunctionCall,
)
from ollm.server.openai_response_streaming import structured_output_events
from ollm.server.openai_response_tooling import parse_model_output


def _function_tool() -> OpenAIResponseFunctionToolRequestModel:
    return OpenAIResponseFunctionToolRequestModel(
        name="get_weather",
        description="Look up current weather",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )


def test_build_response_prompt_replays_previous_function_call_and_tool_output() -> None:
    store = MemoryOpenAIResponseStore()
    previous_response = build_response_payload(
        response_id="resp_prev",
        output_message_id="msg_prev",
        created_at=1,
        model="llama3-1B-chat",
        parsed_output=parse_model_output(
            '{"type":"function_call","calls":[{"name":"get_weather",'
            '"arguments":{"city":"Paris"},"call_id":"call_weather"}]}',
            tools=[_function_tool()],
            tool_choice="auto",
            parallel_tool_calls=True,
        ),
        instructions=None,
        previous_response_id=None,
        tools=[_function_tool()],
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    store.put(
        StoredOpenAIResponse(
            response=previous_response,
            conversation_items=[
                Message.user_text("weather?"),
                StoredOpenAIResponseFunctionCall(
                    item_id="fc_prev",
                    call_id="call_weather",
                    name="get_weather",
                    arguments='{"city":"Paris"}',
                ),
            ],
        )
    )

    prepared = build_response_prompt(
        [
            OpenAIResponseFunctionCallOutputRequestModel(
                call_id="call_weather",
                output={"temperature_c": 21, "condition": "sunny"},
            )
        ],
        previous_response_id="resp_prev",
        response_store=store,
    )

    assert [message.role.value for message in prepared.history_messages] == [
        "user",
        "assistant",
    ]
    assert (
        prepared.history_messages[1]
        .text_content()
        .startswith("[function_call name=get_weather")
    )
    assert prepared.prompt_parts[0].value.startswith(
        "[function_call_output call_id=call_weather]"
    )


def test_response_payload_builds_function_call_items() -> None:
    response = build_response_payload(
        response_id="resp_1",
        output_message_id="msg_1",
        created_at=1,
        model="llama3-1B-chat",
        parsed_output=parse_model_output(
            '{"type":"function_call","calls":[{"name":"get_weather",'
            '"arguments":{"city":"Paris"}}]}',
            tools=[_function_tool()],
            tool_choice="auto",
            parallel_tool_calls=True,
        ),
        instructions="be brief",
        previous_response_id=None,
        tools=[_function_tool()],
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    tool_call = response.output[0]
    assert isinstance(tool_call, OpenAIResponseFunctionCallResponseModel)
    assert tool_call.type == "function_call"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == '{"city":"Paris"}'
    assert response.tool_choice == "auto"
    assert response.tools[0].name == "get_weather"


def test_structured_output_events_emit_function_call_argument_events() -> None:
    response = build_response_payload(
        response_id="resp_1",
        output_message_id="msg_1",
        created_at=1,
        model="llama3-1B-chat",
        parsed_output=parse_model_output(
            '{"type":"function_call","calls":[{"name":"get_weather",'
            '"arguments":{"city":"Paris"}}]}',
            tools=[_function_tool()],
            tool_choice="auto",
            parallel_tool_calls=True,
        ),
        instructions=None,
        previous_response_id=None,
        tools=[_function_tool()],
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    lines = list(structured_output_events(response_id="resp_1", response=response))
    event_names = [line.splitlines()[0].removeprefix("event: ") for line in lines]
    payloads = [
        json.loads(line.splitlines()[1].removeprefix("data: ")) for line in lines
    ]

    assert event_names == [
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
    ]
    assert payloads[1]["delta"] == '{"city":"Paris"}'
    assert payloads[2]["arguments"] == '{"city":"Paris"}'


def test_parse_model_output_rejects_missing_required_tool_call() -> None:
    with pytest.raises(ValueError, match="tool_choice=required"):
        parse_model_output(
            '{"type":"message","content":"hello"}',
            tools=[_function_tool()],
            tool_choice="required",
            parallel_tool_calls=True,
        )

"""Streaming adapters for OpenAI-compatible responses."""

import time
from collections.abc import Iterator, Sequence
from queue import Queue
from threading import Thread

from ollm.app.types import Message
from ollm.server.openai_response_execution import (
    build_conversation_items,
    build_response_payload,
    new_output_message_id,
)
from ollm.server.openai_response_models import (
    OpenAIResponseCompletedEventModel,
    OpenAIResponseContentPartAddedEventModel,
    OpenAIResponseContentPartDoneEventModel,
    OpenAIResponseCreatedEventModel,
    OpenAIResponseFailedEventModel,
    OpenAIResponseFunctionCallOutputRequestModel,
    OpenAIResponseFunctionToolChoiceRequestModel,
    OpenAIResponseFunctionToolRequestModel,
    OpenAIResponseInProgressEventModel,
    OpenAIResponseInputMessageRequestModel,
    OpenAIResponseOutputItemAddedEventModel,
    OpenAIResponseOutputItemDoneEventModel,
    OpenAIResponseOutputMessageResponseModel,
    OpenAIResponseOutputTextDoneEventModel,
    OpenAIResponseOutputTextResponseModel,
    OpenAIResponseResponseModel,
)
from ollm.server.openai_response_store import (
    OpenAIResponseStore,
    StoredOpenAIResponse,
    StoredOpenAIResponseFunctionCall,
    StoredOpenAIResponseFunctionCallOutput,
)
from ollm.server.openai_response_streaming_support import (
    OpenAIResponsesStreamSink,
    sse_event,
    structured_output_events,
)
from ollm.server.openai_response_tooling import (
    ParsedOpenAIResponseOutput,
    parse_model_output,
)


def response_event_iterator(
    execute,
    *,
    model: str,
    response_id: str,
    output_message_id: str,
    created_at: int,
    request_input: str
    | list[
        OpenAIResponseInputMessageRequestModel
        | OpenAIResponseFunctionCallOutputRequestModel
    ],
    instructions: str | None,
    max_output_tokens: int | None,
    previous_response_id: str | None,
    response_store: OpenAIResponseStore,
    conversation_items: Sequence[
        Message
        | StoredOpenAIResponseFunctionCall
        | StoredOpenAIResponseFunctionCallOutput
    ],
    temperature: float | None,
    tools: list[OpenAIResponseFunctionToolRequestModel],
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel,
    top_p: float | None,
    parallel_tool_calls: bool,
) -> Iterator[str]:
    """Stream a plain-text response through the Responses SSE event family."""

    queue: Queue[str | None] = Queue()
    sequence_number = 0

    def next_sequence_number() -> int:
        nonlocal sequence_number
        sequence_number += 1
        return sequence_number

    sink = OpenAIResponsesStreamSink(
        queue,
        response_id=response_id,
        output_message_id=output_message_id,
        next_sequence_number=next_sequence_number,
    )

    output_text_part = OpenAIResponseOutputTextResponseModel(text="")
    initial_output_item = OpenAIResponseOutputMessageResponseModel(
        id=output_message_id,
        status="in_progress",
        content=[output_text_part],
    )
    initial_response = build_response_payload(
        response_id=response_id,
        output_message_id=output_message_id,
        created_at=created_at,
        model=model,
        parsed_output=ParsedOpenAIResponseOutput(message_text="", function_calls=()),
        input_items=request_input,
        instructions=instructions,
        max_output_tokens=max_output_tokens,
        previous_response_id=previous_response_id,
        store=response_store.enabled,
        temperature=temperature,
        tools=tools,
        tool_choice=tool_choice,
        top_p=top_p,
        parallel_tool_calls=parallel_tool_calls,
        status="in_progress",
        output_item_status="in_progress",
        output_items=[initial_output_item],
    )
    queue.put(
        sse_event(
            "response.created",
            OpenAIResponseCreatedEventModel(
                response=initial_response.model_dump(exclude_none=True),
                sequence_number=next_sequence_number(),
            ).model_dump(exclude_none=True),
        )
    )
    queue.put(
        sse_event(
            "response.in_progress",
            OpenAIResponseInProgressEventModel(
                response=initial_response.model_dump(exclude_none=True),
                sequence_number=next_sequence_number(),
            ).model_dump(exclude_none=True),
        )
    )
    output_item = initial_response.output[0]
    if not isinstance(output_item, OpenAIResponseOutputMessageResponseModel):
        raise ValueError("responses text streaming requires a message output item")
    queue.put(
        sse_event(
            "response.output_item.added",
            OpenAIResponseOutputItemAddedEventModel(
                response_id=response_id,
                item=output_item,
                sequence_number=next_sequence_number(),
            ).model_dump(exclude_none=True),
        )
    )
    queue.put(
        sse_event(
            "response.content_part.added",
            OpenAIResponseContentPartAddedEventModel(
                response_id=response_id,
                item_id=output_message_id,
                part=output_text_part,
                sequence_number=next_sequence_number(),
            ).model_dump(exclude_none=True),
        )
    )

    def run() -> None:
        try:
            prompt_response = execute(sink)
            final_response = build_response_payload(
                response_id=response_id,
                output_message_id=output_message_id,
                created_at=created_at,
                completed_at=int(time.time()),
                model=model,
                parsed_output=ParsedOpenAIResponseOutput(
                    message_text=prompt_response.text,
                    function_calls=(),
                ),
                input_items=request_input,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                previous_response_id=previous_response_id,
                store=response_store.enabled,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                top_p=top_p,
                parallel_tool_calls=parallel_tool_calls,
            )
            response_store.put(
                StoredOpenAIResponse(
                    response=final_response,
                    conversation_items=build_conversation_items(
                        base_items=conversation_items,
                        output_items=final_response.output,
                    ),
                )
            )
            queue.put(
                sse_event(
                    "response.output_text.done",
                    OpenAIResponseOutputTextDoneEventModel(
                        response_id=response_id,
                        item_id=output_message_id,
                        text=prompt_response.text,
                        sequence_number=next_sequence_number(),
                    ).model_dump(exclude_none=True),
                )
            )
            queue.put(
                sse_event(
                    "response.content_part.done",
                    OpenAIResponseContentPartDoneEventModel(
                        response_id=response_id,
                        item_id=output_message_id,
                        part=OpenAIResponseOutputTextResponseModel(
                            text=prompt_response.text
                        ),
                        sequence_number=next_sequence_number(),
                    ).model_dump(exclude_none=True),
                )
            )
            queue.put(
                sse_event(
                    "response.output_item.done",
                    OpenAIResponseOutputItemDoneEventModel(
                        response_id=response_id,
                        item=final_response.output[0],
                        sequence_number=next_sequence_number(),
                    ).model_dump(exclude_none=True),
                )
            )
            queue.put(
                sse_event(
                    "response.completed",
                    OpenAIResponseCompletedEventModel(
                        response=final_response.model_dump(exclude_none=True),
                        sequence_number=next_sequence_number(),
                    ).model_dump(exclude_none=True),
                )
            )
        except Exception as exc:
            failed_response = build_response_payload(
                response_id=response_id,
                output_message_id=output_message_id,
                created_at=created_at,
                model=model,
                parsed_output=ParsedOpenAIResponseOutput(
                    message_text="",
                    function_calls=(),
                ),
                input_items=request_input,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                previous_response_id=previous_response_id,
                store=response_store.enabled,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                top_p=top_p,
                parallel_tool_calls=parallel_tool_calls,
                status="failed",
                error={
                    "code": "server_error",
                    "message": str(exc),
                },
                output_items=[],
            )
            queue.put(
                sse_event(
                    "response.failed",
                    OpenAIResponseFailedEventModel(
                        response=failed_response.model_dump(exclude_none=True),
                        sequence_number=next_sequence_number(),
                    ).model_dump(exclude_none=True),
                )
            )
        finally:
            queue.put(None)

    Thread(target=run, daemon=True).start()

    while True:
        item = queue.get()
        if item is None:
            return
        yield item


def structured_response_event_iterator(
    execute,
    *,
    model: str,
    response_id: str,
    created_at: int,
    request_input: str
    | list[
        OpenAIResponseInputMessageRequestModel
        | OpenAIResponseFunctionCallOutputRequestModel
    ],
    instructions: str | None,
    max_output_tokens: int | None,
    previous_response_id: str | None,
    response_store: OpenAIResponseStore,
    conversation_items: Sequence[
        Message
        | StoredOpenAIResponseFunctionCall
        | StoredOpenAIResponseFunctionCallOutput
    ],
    temperature: float | None,
    tools: list[OpenAIResponseFunctionToolRequestModel],
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel,
    top_p: float | None,
    parallel_tool_calls: bool,
) -> Iterator[str]:
    """Stream a structured tool-capable response through the Responses SSE API."""

    queue: Queue[str | None] = Queue()
    sequence_number = 0

    def next_sequence_number() -> int:
        nonlocal sequence_number
        sequence_number += 1
        return sequence_number

    initial_response = OpenAIResponseResponseModel(
        id=response_id,
        created_at=created_at,
        status="in_progress",
        input=request_input,
        max_output_tokens=max_output_tokens,
        model=model,
        output=[],
        instructions=instructions,
        previous_response_id=previous_response_id,
        store=response_store.enabled,
        temperature=temperature,
        top_p=top_p,
        tools=list(tools),
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    queue.put(
        sse_event(
            "response.created",
            OpenAIResponseCreatedEventModel(
                response=initial_response.model_dump(exclude_none=True),
                sequence_number=next_sequence_number(),
            ).model_dump(exclude_none=True),
        )
    )
    queue.put(
        sse_event(
            "response.in_progress",
            OpenAIResponseInProgressEventModel(
                response=initial_response.model_dump(exclude_none=True),
                sequence_number=next_sequence_number(),
            ).model_dump(exclude_none=True),
        )
    )

    def run() -> None:
        try:
            prompt_response = execute()
            parsed_output = parse_model_output(
                prompt_response.text,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )
            final_response = build_response_payload(
                response_id=response_id,
                output_message_id=new_output_message_id(),
                created_at=created_at,
                completed_at=int(time.time()),
                model=model,
                parsed_output=parsed_output,
                input_items=request_input,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                previous_response_id=previous_response_id,
                store=response_store.enabled,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                top_p=top_p,
                parallel_tool_calls=parallel_tool_calls,
            )
            response_store.put(
                StoredOpenAIResponse(
                    response=final_response,
                    conversation_items=build_conversation_items(
                        base_items=conversation_items,
                        output_items=final_response.output,
                    ),
                )
            )
            for event in structured_output_events(
                response_id=response_id,
                response=final_response,
                next_sequence_number=next_sequence_number,
            ):
                queue.put(event)
            queue.put(
                sse_event(
                    "response.completed",
                    OpenAIResponseCompletedEventModel(
                        response=final_response.model_dump(exclude_none=True),
                        sequence_number=next_sequence_number(),
                    ).model_dump(exclude_none=True),
                )
            )
        except Exception as exc:
            failed_response = OpenAIResponseResponseModel(
                id=response_id,
                created_at=created_at,
                status="failed",
                input=request_input,
                max_output_tokens=max_output_tokens,
                model=model,
                output=[],
                instructions=instructions,
                previous_response_id=previous_response_id,
                store=response_store.enabled,
                temperature=temperature,
                top_p=top_p,
                tools=list(tools),
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                error={
                    "code": "server_error",
                    "message": str(exc),
                },
            )
            queue.put(
                sse_event(
                    "response.failed",
                    OpenAIResponseFailedEventModel(
                        response=failed_response.model_dump(exclude_none=True),
                        sequence_number=next_sequence_number(),
                    ).model_dump(exclude_none=True),
                )
            )
        finally:
            queue.put(None)

    Thread(target=run, daemon=True).start()

    while True:
        item = queue.get()
        if item is None:
            return
        yield item

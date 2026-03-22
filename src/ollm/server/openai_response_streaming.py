"""Streaming adapters for OpenAI-compatible responses."""

import json
from collections.abc import Iterator, Sequence
from queue import Queue
from threading import Thread

from ollm.app.types import Message
from ollm.runtime.streaming import StreamSink
from ollm.server.openai_response_execution import (
    build_conversation_items,
    build_response_payload,
    new_output_message_id,
)
from ollm.server.openai_response_models import (
    OpenAIResponseCompletedEventModel,
    OpenAIResponseContentPartAddedEventModel,
    OpenAIResponseCreatedEventModel,
    OpenAIResponseFunctionCallArgumentsDeltaEventModel,
    OpenAIResponseFunctionCallArgumentsDoneEventModel,
    OpenAIResponseFunctionToolChoiceRequestModel,
    OpenAIResponseFunctionToolRequestModel,
    OpenAIResponseOutputItemAddedEventModel,
    OpenAIResponseOutputItemDoneEventModel,
    OpenAIResponseOutputMessageResponseModel,
    OpenAIResponseOutputTextDeltaEventModel,
    OpenAIResponseOutputTextDoneEventModel,
    OpenAIResponseResponseModel,
)
from ollm.server.openai_response_store import (
    OpenAIResponseStore,
    StoredOpenAIResponse,
    StoredOpenAIResponseFunctionCall,
    StoredOpenAIResponseFunctionCallOutput,
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
    instructions: str | None,
    previous_response_id: str | None,
    response_store: OpenAIResponseStore,
    conversation_items: Sequence[
        Message
        | StoredOpenAIResponseFunctionCall
        | StoredOpenAIResponseFunctionCallOutput
    ],
    tools: list[OpenAIResponseFunctionToolRequestModel],
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel,
    parallel_tool_calls: bool,
) -> Iterator[str]:
    """Stream a plain-text response through the Responses SSE event family."""

    queue: Queue[str | None] = Queue()
    sink = OpenAIResponsesStreamSink(
        queue,
        response_id=response_id,
        output_message_id=output_message_id,
    )

    initial_response = build_response_payload(
        response_id=response_id,
        output_message_id=output_message_id,
        created_at=created_at,
        model=model,
        parsed_output=ParsedOpenAIResponseOutput(message_text="", function_calls=()),
        instructions=instructions,
        previous_response_id=previous_response_id,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        status="in_progress",
    )
    queue.put(
        sse_event(
            "response.created",
            OpenAIResponseCreatedEventModel(response=initial_response).model_dump(
                exclude_none=True
            ),
        )
    )
    output_item = initial_response.output[0]
    if not isinstance(output_item, OpenAIResponseOutputMessageResponseModel):
        raise ValueError("responses text streaming requires a message output item")
    output_text_part = output_item.content[0]
    queue.put(
        sse_event(
            "response.output_item.added",
            OpenAIResponseOutputItemAddedEventModel(
                response_id=response_id,
                item=output_item,
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
                model=model,
                parsed_output=ParsedOpenAIResponseOutput(
                    message_text=prompt_response.text,
                    function_calls=(),
                ),
                instructions=instructions,
                previous_response_id=previous_response_id,
                tools=tools,
                tool_choice=tool_choice,
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
                    ).model_dump(exclude_none=True),
                )
            )
            queue.put(
                sse_event(
                    "response.output_item.done",
                    OpenAIResponseOutputItemDoneEventModel(
                        response_id=response_id,
                        item=final_response.output[0],
                    ).model_dump(exclude_none=True),
                )
            )
            queue.put(
                sse_event(
                    "response.completed",
                    OpenAIResponseCompletedEventModel(
                        response=final_response
                    ).model_dump(exclude_none=True),
                )
            )
        except Exception as exc:
            queue.put(
                sse_event(
                    "error",
                    {
                        "type": "error",
                        "message": str(exc),
                    },
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
    instructions: str | None,
    previous_response_id: str | None,
    response_store: OpenAIResponseStore,
    conversation_items: Sequence[
        Message
        | StoredOpenAIResponseFunctionCall
        | StoredOpenAIResponseFunctionCallOutput
    ],
    tools: list[OpenAIResponseFunctionToolRequestModel],
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel,
    parallel_tool_calls: bool,
) -> Iterator[str]:
    """Stream a structured tool-capable response through the Responses SSE API."""

    queue: Queue[str | None] = Queue()
    initial_response = OpenAIResponseResponseModel(
        id=response_id,
        created_at=created_at,
        status="in_progress",
        model=model,
        output=[],
        instructions=instructions,
        previous_response_id=previous_response_id,
        tools=list(tools),
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    queue.put(
        sse_event(
            "response.created",
            OpenAIResponseCreatedEventModel(response=initial_response).model_dump(
                exclude_none=True
            ),
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
                model=model,
                parsed_output=parsed_output,
                instructions=instructions,
                previous_response_id=previous_response_id,
                tools=tools,
                tool_choice=tool_choice,
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
            ):
                queue.put(event)
            queue.put(
                sse_event(
                    "response.completed",
                    OpenAIResponseCompletedEventModel(
                        response=final_response
                    ).model_dump(exclude_none=True),
                )
            )
        except Exception as exc:
            queue.put(
                sse_event(
                    "error",
                    {
                        "type": "error",
                        "message": str(exc),
                    },
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


def structured_output_events(
    *,
    response_id: str,
    response: OpenAIResponseResponseModel,
) -> Iterator[str]:
    """Emit item-scoped SSE events for a completed structured response."""

    for output_index, output_item in enumerate(response.output):
        yield sse_event(
            "response.output_item.added",
            OpenAIResponseOutputItemAddedEventModel(
                response_id=response_id,
                output_index=output_index,
                item=output_item,
            ).model_dump(exclude_none=True),
        )
        if isinstance(output_item, OpenAIResponseOutputMessageResponseModel):
            content_part = output_item.content[0]
            yield sse_event(
                "response.content_part.added",
                OpenAIResponseContentPartAddedEventModel(
                    response_id=response_id,
                    item_id=output_item.id,
                    output_index=output_index,
                    part=content_part,
                ).model_dump(exclude_none=True),
            )
            if content_part.text:
                yield sse_event(
                    "response.output_text.delta",
                    OpenAIResponseOutputTextDeltaEventModel(
                        response_id=response_id,
                        item_id=output_item.id,
                        output_index=output_index,
                        delta=content_part.text,
                    ).model_dump(exclude_none=True),
                )
            yield sse_event(
                "response.output_text.done",
                OpenAIResponseOutputTextDoneEventModel(
                    response_id=response_id,
                    item_id=output_item.id,
                    output_index=output_index,
                    text=content_part.text,
                ).model_dump(exclude_none=True),
            )
        else:
            if output_item.arguments:
                yield sse_event(
                    "response.function_call_arguments.delta",
                    OpenAIResponseFunctionCallArgumentsDeltaEventModel(
                        response_id=response_id,
                        item_id=output_item.id,
                        output_index=output_index,
                        delta=output_item.arguments,
                    ).model_dump(exclude_none=True),
                )
            yield sse_event(
                "response.function_call_arguments.done",
                OpenAIResponseFunctionCallArgumentsDoneEventModel(
                    response_id=response_id,
                    item_id=output_item.id,
                    output_index=output_index,
                    arguments=output_item.arguments,
                ).model_dump(exclude_none=True),
            )
        yield sse_event(
            "response.output_item.done",
            OpenAIResponseOutputItemDoneEventModel(
                response_id=response_id,
                output_index=output_index,
                item=output_item,
            ).model_dump(exclude_none=True),
        )


class OpenAIResponsesStreamSink(StreamSink):
    """Stream sink that serializes text deltas into Responses SSE events."""

    def __init__(
        self,
        queue: Queue[str | None],
        *,
        response_id: str,
        output_message_id: str,
    ) -> None:
        self._queue = queue
        self._response_id = response_id
        self._output_message_id = output_message_id

    def on_status(self, message: str) -> None:
        del message

    def on_text(self, text: str) -> None:
        if not text:
            return
        self._queue.put(
            sse_event(
                "response.output_text.delta",
                OpenAIResponseOutputTextDeltaEventModel(
                    response_id=self._response_id,
                    item_id=self._output_message_id,
                    delta=text,
                ).model_dump(exclude_none=True),
            )
        )

    def on_complete(self, text: str) -> None:
        del text


def sse_event(event: str, payload: dict[str, object]) -> str:
    """Serialize one SSE event line pair."""

    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

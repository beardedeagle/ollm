"""Shared streaming helpers for OpenAI-compatible Responses events."""

import json
from collections.abc import Callable, Iterator
from queue import Queue

from ollm.runtime.streaming import StreamSink
from ollm.server.openai_response_models import (
    OpenAIResponseContentPartAddedEventModel,
    OpenAIResponseContentPartDoneEventModel,
    OpenAIResponseFunctionCallArgumentsDeltaEventModel,
    OpenAIResponseFunctionCallArgumentsDoneEventModel,
    OpenAIResponseFunctionCallResponseModel,
    OpenAIResponseOutputItemAddedEventModel,
    OpenAIResponseOutputItemDoneEventModel,
    OpenAIResponseOutputMessageResponseModel,
    OpenAIResponseOutputTextDeltaEventModel,
    OpenAIResponseOutputTextDoneEventModel,
    OpenAIResponseResponseModel,
)


def structured_output_events(
    *,
    response_id: str,
    response: OpenAIResponseResponseModel,
    next_sequence_number: Callable[[], int] | None = None,
) -> Iterator[str]:
    """Emit item-scoped SSE events for a completed structured response."""

    local_sequence_number = 0

    def fallback_sequence_number() -> int:
        nonlocal local_sequence_number
        local_sequence_number += 1
        return local_sequence_number

    resolved_next_sequence_number = (
        fallback_sequence_number
        if next_sequence_number is None
        else next_sequence_number
    )

    for output_index, output_item in enumerate(response.output):
        yield sse_event(
            "response.output_item.added",
            OpenAIResponseOutputItemAddedEventModel(
                response_id=response_id,
                output_index=output_index,
                item=_in_progress_output_item(output_item),
                sequence_number=resolved_next_sequence_number(),
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
                    sequence_number=resolved_next_sequence_number(),
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
                        sequence_number=resolved_next_sequence_number(),
                    ).model_dump(exclude_none=True),
                )
            yield sse_event(
                "response.output_text.done",
                OpenAIResponseOutputTextDoneEventModel(
                    response_id=response_id,
                    item_id=output_item.id,
                    output_index=output_index,
                    text=content_part.text,
                    sequence_number=resolved_next_sequence_number(),
                ).model_dump(exclude_none=True),
            )
            yield sse_event(
                "response.content_part.done",
                OpenAIResponseContentPartDoneEventModel(
                    response_id=response_id,
                    item_id=output_item.id,
                    output_index=output_index,
                    part=content_part,
                    sequence_number=resolved_next_sequence_number(),
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
                        sequence_number=resolved_next_sequence_number(),
                    ).model_dump(exclude_none=True),
                )
            yield sse_event(
                "response.function_call_arguments.done",
                OpenAIResponseFunctionCallArgumentsDoneEventModel(
                    response_id=response_id,
                    item_id=output_item.id,
                    output_index=output_index,
                    arguments=output_item.arguments,
                    sequence_number=resolved_next_sequence_number(),
                ).model_dump(exclude_none=True),
            )
        yield sse_event(
            "response.output_item.done",
            OpenAIResponseOutputItemDoneEventModel(
                response_id=response_id,
                output_index=output_index,
                item=output_item,
                sequence_number=resolved_next_sequence_number(),
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
        next_sequence_number: Callable[[], int],
    ) -> None:
        self._queue = queue
        self._response_id = response_id
        self._output_message_id = output_message_id
        self._next_sequence_number = next_sequence_number

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
                    sequence_number=self._next_sequence_number(),
                ).model_dump(exclude_none=True),
            )
        )

    def on_complete(self, text: str) -> None:
        del text


def sse_event(event: str, payload: dict[str, object]) -> str:
    """Serialize one SSE event line pair."""

    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _in_progress_output_item(
    output_item: OpenAIResponseOutputMessageResponseModel
    | OpenAIResponseFunctionCallResponseModel,
) -> OpenAIResponseOutputMessageResponseModel | OpenAIResponseFunctionCallResponseModel:
    if isinstance(output_item, OpenAIResponseOutputMessageResponseModel):
        return output_item.model_copy(update={"status": "in_progress", "content": []})
    return output_item.model_copy(update={"status": "in_progress"})

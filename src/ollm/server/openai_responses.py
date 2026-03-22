"""OpenAI-compatible Responses API adapters and route registration."""

import json
import time
import uuid
from collections.abc import Iterator, Sequence
from queue import Queue
from threading import Thread

from ollm.app.service import ApplicationService
from ollm.app.types import ContentPart, Message, MessageRole
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.settings import (
    GenerationConfigOverrides,
    RuntimeConfigOverrides,
    load_app_settings,
    resolve_generation_config,
    resolve_runtime_config,
)
from ollm.runtime.streaming import StreamSink
from ollm.server.openai_compat import build_openai_error_response
from ollm.server.openai_response_models import (
    OpenAIResponseCompletedEventModel,
    OpenAIResponseContentPartAddedEventModel,
    OpenAIResponseCreatedEventModel,
    OpenAIResponseCreateRequestModel,
    OpenAIResponseInputContentPartRequestModel,
    OpenAIResponseInputMessageRequestModel,
    OpenAIResponseOutputItemAddedEventModel,
    OpenAIResponseOutputItemDoneEventModel,
    OpenAIResponseOutputMessageResponseModel,
    OpenAIResponseOutputTextDeltaEventModel,
    OpenAIResponseOutputTextDoneEventModel,
    OpenAIResponseOutputTextResponseModel,
    OpenAIResponseResponseModel,
)
from ollm.server.openai_response_store import OpenAIResponseStore, StoredOpenAIResponse
from ollm.server.streaming import build_sse_response_from_iterator


def register_openai_responses_routes(
    app,
    *,
    application_service: ApplicationService,
    response_store: OpenAIResponseStore,
) -> None:
    """Register the OpenAI-compatible Responses API routes."""

    @app.post(
        "/v1/responses",
        response_model=OpenAIResponseResponseModel,
        summary="Create a response (OpenAI-compatible)",
        tags=["openai-compatible"],
    )
    def create_response(
        request: OpenAIResponseCreateRequestModel,
    ) -> OpenAIResponseResponseModel | object:
        try:
            runtime_config = _build_runtime_config(request.model)
            generation_config = _build_generation_config(request)
            history, prompt_parts = _build_response_prompt(
                request.input,
                previous_response_id=request.previous_response_id,
                response_store=response_store,
            )
        except ValueError as exc:
            return build_openai_error_response(status_code=400, message=str(exc))

        created_at = int(time.time())
        response_id = _new_response_id()
        output_message_id = _new_output_message_id()

        if request.stream:
            return build_sse_response_from_iterator(
                _response_event_iterator(
                    lambda sink: application_service.prompt_parts(
                        prompt_parts,
                        runtime_config=runtime_config,
                        generation_config=generation_config,
                        system_prompt=request.instructions or "",
                        history=history,
                        sink=sink,
                    ),
                    model=request.model,
                    response_id=response_id,
                    output_message_id=output_message_id,
                    created_at=created_at,
                    instructions=request.instructions,
                    previous_response_id=request.previous_response_id,
                    response_store=response_store,
                    history=history,
                    prompt_parts=prompt_parts,
                )
            )

        try:
            prompt_response = application_service.prompt_parts(
                prompt_parts,
                runtime_config=runtime_config,
                generation_config=generation_config,
                system_prompt=request.instructions or "",
                history=history,
            )
        except ValueError as exc:
            return build_openai_error_response(status_code=400, message=str(exc))

        response = _response_payload(
            response_id=response_id,
            output_message_id=output_message_id,
            created_at=created_at,
            model=request.model,
            text=prompt_response.text,
            instructions=request.instructions,
            previous_response_id=request.previous_response_id,
        )
        response_store.put(
            StoredOpenAIResponse(
                response=response,
                conversation_messages=_conversation_messages(
                    history=history,
                    prompt_parts=prompt_parts,
                    assistant_text=prompt_response.text,
                ),
            )
        )
        return response

    @app.get(
        "/v1/responses/{response_id}",
        response_model=OpenAIResponseResponseModel,
        summary="Retrieve a response (OpenAI-compatible)",
        tags=["openai-compatible"],
    )
    def get_response(response_id: str) -> OpenAIResponseResponseModel | object:
        if not response_store.enabled:
            return build_openai_error_response(
                status_code=501,
                message="Responses retrieval requires a configured response-store backend",
                error_type="server_error",
                code="responses_storage_disabled",
            )
        try:
            return response_store.require(response_id).response
        except ValueError as exc:
            return build_openai_error_response(
                status_code=404,
                message=str(exc),
                error_type="not_found_error",
            )


def _build_runtime_config(model_reference: str) -> RuntimeConfig:
    settings = load_app_settings()
    return resolve_runtime_config(
        settings.runtime,
        RuntimeConfigOverrides(model_reference=model_reference),
    )


def _build_generation_config(
    request: OpenAIResponseCreateRequestModel,
) -> GenerationConfig:
    settings = load_app_settings()
    return resolve_generation_config(
        settings.generation,
        GenerationConfigOverrides(
            max_new_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
            stream=request.stream,
        ),
    )


def _build_response_prompt(
    input_value: str | list[OpenAIResponseInputMessageRequestModel],
    *,
    previous_response_id: str | None,
    response_store: OpenAIResponseStore,
) -> tuple[list[Message], list[ContentPart]]:
    history: list[Message] = []
    if previous_response_id is not None:
        if not response_store.enabled:
            raise ValueError(
                "previous_response_id requires a configured response-store backend"
            )
        stored_response = response_store.require(previous_response_id)
        history.extend(stored_response.conversation_messages)

    if isinstance(input_value, str):
        if not input_value:
            raise ValueError("responses input must not be empty")
        return history, [ContentPart.text(input_value)]

    if not input_value:
        raise ValueError("responses input must include at least one message")

    input_messages = [_translate_input_message(message) for message in input_value]
    current_message = input_messages[-1]
    if current_message.role is not MessageRole.USER:
        raise ValueError(
            "responses input currently requires the final role to be 'user'"
        )
    history.extend(input_messages[:-1])
    return history, current_message.content


def _translate_input_message(
    message: OpenAIResponseInputMessageRequestModel,
) -> Message:
    parts = _translate_input_content(message.content)
    if message.role == "system":
        return Message(role=MessageRole.SYSTEM, content=parts)
    if message.role == "assistant":
        return Message(role=MessageRole.ASSISTANT, content=parts)
    if message.role == "user":
        return Message(role=MessageRole.USER, content=parts)
    raise ValueError("responses input role must be one of: system, user, assistant")


def _translate_input_content(
    content: str | list[OpenAIResponseInputContentPartRequestModel],
) -> list[ContentPart]:
    if isinstance(content, str):
        if not content:
            raise ValueError("responses input content must not be empty")
        return [ContentPart.text(content)]
    parts = [_translate_input_part(part) for part in content]
    if not parts:
        raise ValueError("responses input content must include at least one text part")
    return parts


def _translate_input_part(
    part: OpenAIResponseInputContentPartRequestModel,
) -> ContentPart:
    if part.type in {"text", "input_text"}:
        if not part.text:
            raise ValueError("responses text parts must not be empty")
        return ContentPart.text(part.text)
    if part.type in {"image", "input_image"}:
        if not part.image_url:
            raise ValueError("responses image parts require image_url")
        return ContentPart.image(part.image_url)
    if part.type in {"audio", "input_audio"}:
        if not part.audio_url:
            raise ValueError("responses audio parts require audio_url")
        return ContentPart.audio(part.audio_url)
    raise ValueError(
        "responses input currently supports only text, image, and audio content parts"
    )


def _new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _new_output_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _response_payload(
    *,
    response_id: str,
    output_message_id: str,
    created_at: int,
    model: str,
    text: str,
    instructions: str | None,
    previous_response_id: str | None,
    status: str = "completed",
) -> OpenAIResponseResponseModel:
    return OpenAIResponseResponseModel(
        id=response_id,
        created_at=created_at,
        status=status,
        model=model,
        output=[
            OpenAIResponseOutputMessageResponseModel(
                id=output_message_id,
                content=[OpenAIResponseOutputTextResponseModel(text=text)],
            )
        ],
        instructions=instructions,
        previous_response_id=previous_response_id,
    )


def _conversation_messages(
    *,
    history: Sequence[Message],
    prompt_parts: Sequence[ContentPart],
    assistant_text: str,
) -> list[Message]:
    return [
        *history,
        Message(role=MessageRole.USER, content=list(prompt_parts)),
        Message.assistant_text(assistant_text),
    ]


def _response_event_iterator(
    execute,
    *,
    model: str,
    response_id: str,
    output_message_id: str,
    created_at: int,
    instructions: str | None,
    previous_response_id: str | None,
    response_store: OpenAIResponseStore,
    history: Sequence[Message],
    prompt_parts: Sequence[ContentPart],
) -> Iterator[str]:
    queue: Queue[str | None] = Queue()
    sink = _OpenAIResponsesStreamSink(
        queue,
        response_id=response_id,
        output_message_id=output_message_id,
    )

    initial_response = _response_payload(
        response_id=response_id,
        output_message_id=output_message_id,
        created_at=created_at,
        model=model,
        text="",
        instructions=instructions,
        previous_response_id=previous_response_id,
        status="in_progress",
    )
    queue.put(
        _sse_event(
            "response.created",
            OpenAIResponseCreatedEventModel(response=initial_response).model_dump(
                exclude_none=True
            ),
        )
    )
    output_item = initial_response.output[0]
    output_text_part = output_item.content[0]
    queue.put(
        _sse_event(
            "response.output_item.added",
            OpenAIResponseOutputItemAddedEventModel(
                response_id=response_id,
                item=output_item,
            ).model_dump(exclude_none=True),
        )
    )
    queue.put(
        _sse_event(
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
            final_response = _response_payload(
                response_id=response_id,
                output_message_id=output_message_id,
                created_at=created_at,
                model=model,
                text=prompt_response.text,
                instructions=instructions,
                previous_response_id=previous_response_id,
            )
            response_store.put(
                StoredOpenAIResponse(
                    response=final_response,
                    conversation_messages=_conversation_messages(
                        history=history,
                        prompt_parts=prompt_parts,
                        assistant_text=prompt_response.text,
                    ),
                )
            )
            queue.put(
                _sse_event(
                    "response.output_text.done",
                    OpenAIResponseOutputTextDoneEventModel(
                        response_id=response_id,
                        item_id=output_message_id,
                        text=prompt_response.text,
                    ).model_dump(exclude_none=True),
                )
            )
            queue.put(
                _sse_event(
                    "response.output_item.done",
                    OpenAIResponseOutputItemDoneEventModel(
                        response_id=response_id,
                        item=final_response.output[0],
                    ).model_dump(exclude_none=True),
                )
            )
            queue.put(
                _sse_event(
                    "response.completed",
                    OpenAIResponseCompletedEventModel(
                        response=final_response
                    ).model_dump(exclude_none=True),
                )
            )
        except Exception as exc:
            queue.put(
                _sse_event(
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


class _OpenAIResponsesStreamSink(StreamSink):
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
            _sse_event(
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


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

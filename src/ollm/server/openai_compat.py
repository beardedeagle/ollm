"""OpenAI-compatible server adapters for the local oLLM server."""

import json
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from importlib import import_module
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Protocol, cast

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.runtime.streaming import StreamSink
from ollm.server.dependencies import SERVER_EXTRA_INSTALL_HINT, ServerDependenciesError
from ollm.server.models import (
    ModelInfoResponseModel,
    OpenAIChatCompletionChoiceResponseModel,
    OpenAIChatCompletionChunkChoiceResponseModel,
    OpenAIChatCompletionChunkResponseModel,
    OpenAIChatCompletionDeltaResponseModel,
    OpenAIChatCompletionMessageResponseModel,
    OpenAIChatCompletionResponseModel,
    OpenAIChatMessageContentPartRequestModel,
    OpenAIChatMessageRequestModel,
    OpenAIErrorEnvelopeResponseModel,
    OpenAIErrorResponseModel,
    OpenAIModelResponseModel,
    OpenAIModelsListResponseModel,
)
from ollm.server.streaming import build_sse_response_from_iterator


class JSONResponseFactory(Protocol):
    """Protocol for the imported JSONResponse constructor."""

    def __call__(self, content: object, *, status_code: int) -> object: ...


def build_openai_model_payload(
    model_info: ModelInfoResponseModel,
) -> OpenAIModelResponseModel:
    """Convert a native model info payload into an OpenAI-style model object."""
    return OpenAIModelResponseModel(
        id=model_info.model_reference,
        created=_model_created_timestamp(model_info.path),
        owned_by=_model_owner(model_info),
    )


def build_openai_models_list_payload(
    model_entries: Sequence[ModelInfoResponseModel],
) -> OpenAIModelsListResponseModel:
    """Convert native model entries into an OpenAI-style model list."""
    return OpenAIModelsListResponseModel(
        data=[build_openai_model_payload(entry) for entry in model_entries]
    )


def parse_openai_chat_messages(
    messages: Sequence[OpenAIChatMessageRequestModel],
) -> tuple[list[Message], list[ContentPart]]:
    """Translate OpenAI chat messages into oLLM history plus the current prompt."""
    if not messages:
        raise ValueError("chat completions requires at least one message")

    translated_messages = [_translate_chat_message(message) for message in messages]
    current_message = translated_messages[-1]
    if current_message.role is not MessageRole.USER:
        raise ValueError(
            "chat completions currently requires the final message role to be 'user'"
        )
    if not current_message.content:
        raise ValueError("chat completions requires a non-empty final user message")
    return translated_messages[:-1], current_message.content


def build_openai_chat_completion_response(
    *,
    model: str,
    text: str,
    response_id: str | None = None,
    created: int | None = None,
) -> OpenAIChatCompletionResponseModel:
    """Build a non-streaming OpenAI-compatible chat completion response."""
    resolved_response_id = (
        new_openai_chat_completion_id() if response_id is None else response_id
    )
    resolved_created = int(time.time()) if created is None else created
    return OpenAIChatCompletionResponseModel(
        id=resolved_response_id,
        created=resolved_created,
        model=model,
        choices=[
            OpenAIChatCompletionChoiceResponseModel(
                index=0,
                message=OpenAIChatCompletionMessageResponseModel(
                    role="assistant",
                    content=text,
                ),
                finish_reason="stop",
            )
        ],
    )


def build_openai_error_response(
    *,
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> object:
    """Build a JSON response with an OpenAI-style error envelope."""
    json_response = _load_json_response_factory()
    payload = OpenAIErrorEnvelopeResponseModel(
        error=OpenAIErrorResponseModel(
            message=message,
            type=error_type,
            code=code,
        )
    )
    return json_response(
        payload.model_dump(exclude_none=True),
        status_code=status_code,
    )


def build_openai_chat_sse_response(
    execute: Callable[[StreamSink], None],
    *,
    model: str,
    response_id: str | None = None,
    created: int | None = None,
) -> object:
    """Build an OpenAI-compatible SSE response for chat completions."""
    stream_response_id = (
        new_openai_chat_completion_id() if response_id is None else response_id
    )
    stream_created = int(time.time()) if created is None else created
    return build_sse_response_from_iterator(
        _openai_chat_event_iterator(
            execute,
            model=model,
            response_id=stream_response_id,
            created=stream_created,
        )
    )


def new_openai_chat_completion_id() -> str:
    """Generate a stable-looking OpenAI-style chat completion id."""
    return f"chatcmpl-{uuid.uuid4().hex}"


def _translate_chat_message(message: OpenAIChatMessageRequestModel) -> Message:
    parts = _translate_message_content(message.content)
    if message.role == "system":
        return Message(role=MessageRole.SYSTEM, content=parts)
    if message.role == "assistant":
        return Message(role=MessageRole.ASSISTANT, content=parts)
    return Message(role=MessageRole.USER, content=parts)


def _translate_message_content(
    content: str | list[OpenAIChatMessageContentPartRequestModel],
) -> list[ContentPart]:
    if isinstance(content, str):
        if not content:
            raise ValueError("chat message content must not be empty")
        return [ContentPart.text(content)]

    parts = [_translate_text_part(part) for part in content]
    if not parts:
        raise ValueError("chat message content must include at least one text part")
    return parts


def _translate_text_part(part: OpenAIChatMessageContentPartRequestModel) -> ContentPart:
    if part.type != "text":
        raise ValueError("chat completions currently supports only text content parts")
    if not part.text:
        raise ValueError("chat message text parts must not be empty")
    return ContentPart.text(part.text)


def _model_created_timestamp(path_value: str | None) -> int:
    if path_value is None:
        return 0
    path = Path(path_value)
    if not path.exists():
        return 0
    return int(path.stat().st_mtime)


def _model_owner(model_info: ModelInfoResponseModel) -> str:
    repo_id = model_info.repo_id
    if repo_id:
        return repo_id.split("/", 1)[0]
    return "ollm"


def _load_json_response_factory() -> JSONResponseFactory:
    try:
        responses_module = import_module("fastapi.responses")
    except ModuleNotFoundError as exc:
        raise ServerDependenciesError(SERVER_EXTRA_INSTALL_HINT) from exc
    return cast(JSONResponseFactory, responses_module.JSONResponse)


def _openai_chat_event_iterator(
    execute: Callable[[StreamSink], None],
    *,
    model: str,
    response_id: str,
    created: int,
) -> Iterator[str]:
    queue: Queue[str | None] = Queue()
    sink = OpenAIChatCompletionStreamSink(
        queue,
        model=model,
        response_id=response_id,
        created=created,
    )

    def run() -> None:
        try:
            execute(sink)
        except Exception as exc:
            queue.put(
                _sse_data(
                    OpenAIErrorEnvelopeResponseModel(
                        error=OpenAIErrorResponseModel(
                            message=str(exc),
                            type="server_error",
                        )
                    ).model_dump(exclude_none=True)
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


class OpenAIChatCompletionStreamSink(StreamSink):
    """Stream sink that serializes runtime text into OpenAI chat chunks."""

    def __init__(
        self,
        queue: Queue[str | None],
        *,
        model: str,
        response_id: str,
        created: int,
    ) -> None:
        self._queue = queue
        self._model = model
        self._response_id = response_id
        self._created = created
        self._started = False

    def on_status(self, message: str) -> None:
        del message

    def on_text(self, text: str) -> None:
        if not text:
            return
        self._emit_role_chunk()
        self._queue.put(
            _sse_data(
                OpenAIChatCompletionChunkResponseModel(
                    id=self._response_id,
                    created=self._created,
                    model=self._model,
                    choices=[
                        OpenAIChatCompletionChunkChoiceResponseModel(
                            index=0,
                            delta=OpenAIChatCompletionDeltaResponseModel(content=text),
                        )
                    ],
                ).model_dump(exclude_none=True)
            )
        )

    def on_complete(self, text: str) -> None:
        del text
        self._emit_role_chunk()
        self._queue.put(
            _sse_data(
                OpenAIChatCompletionChunkResponseModel(
                    id=self._response_id,
                    created=self._created,
                    model=self._model,
                    choices=[
                        OpenAIChatCompletionChunkChoiceResponseModel(
                            index=0,
                            delta=OpenAIChatCompletionDeltaResponseModel(),
                            finish_reason="stop",
                        )
                    ],
                ).model_dump(exclude_none=True)
            )
        )
        self._queue.put("data: [DONE]\n\n")

    def _emit_role_chunk(self) -> None:
        if self._started:
            return
        self._started = True
        self._queue.put(
            _sse_data(
                OpenAIChatCompletionChunkResponseModel(
                    id=self._response_id,
                    created=self._created,
                    model=self._model,
                    choices=[
                        OpenAIChatCompletionChunkChoiceResponseModel(
                            index=0,
                            delta=OpenAIChatCompletionDeltaResponseModel(
                                role="assistant"
                            ),
                        )
                    ],
                ).model_dump(exclude_none=True)
            )
        )


def _sse_data(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"

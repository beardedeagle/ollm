"""Request translation and payload assembly for OpenAI-compatible responses."""

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.server.openai_response_models import (
    OpenAIResponseFunctionCallOutputRequestModel,
    OpenAIResponseFunctionCallResponseModel,
    OpenAIResponseFunctionToolChoiceRequestModel,
    OpenAIResponseFunctionToolRequestModel,
    OpenAIResponseIncompleteDetailsResponseModel,
    OpenAIResponseInputContentPartRequestModel,
    OpenAIResponseInputMessageRequestModel,
    OpenAIResponseOutputMessageResponseModel,
    OpenAIResponseOutputTextResponseModel,
    OpenAIResponseResponseModel,
)
from ollm.server.openai_response_store import (
    OpenAIResponseStore,
    StoredOpenAIResponseFunctionCall,
    StoredOpenAIResponseFunctionCallOutput,
)
from ollm.server.openai_response_tooling import (
    ParsedOpenAIResponseOutput,
    build_runtime_tool_call_text,
    build_runtime_tool_output_text,
    build_tool_system_prompt,
)


@dataclass(frozen=True, slots=True)
class PreparedOpenAIResponseRequest:
    """Prepared request state for a Responses API invocation."""

    history_messages: list[Message]
    prompt_parts: list[ContentPart]
    conversation_items: list[
        Message
        | StoredOpenAIResponseFunctionCall
        | StoredOpenAIResponseFunctionCallOutput
    ]


def resolve_response_system_prompt(
    *,
    instructions: str | None,
    tools: list[OpenAIResponseFunctionToolRequestModel],
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel,
) -> str:
    """Resolve the system prompt for a Responses request."""

    if not tools:
        return instructions or ""
    return build_tool_system_prompt(
        instructions=instructions,
        tools=tools,
        tool_choice=tool_choice,
    )


def build_response_prompt(
    input_value: str
    | list[
        OpenAIResponseInputMessageRequestModel
        | OpenAIResponseFunctionCallOutputRequestModel
    ],
    *,
    previous_response_id: str | None,
    response_store: OpenAIResponseStore,
) -> PreparedOpenAIResponseRequest:
    """Prepare runtime history, prompt parts, and stored conversation items."""

    conversation_items: list[
        Message
        | StoredOpenAIResponseFunctionCall
        | StoredOpenAIResponseFunctionCallOutput
    ] = []
    runtime_messages: list[Message] = []
    if previous_response_id is not None:
        if not response_store.enabled:
            raise ValueError(
                "previous_response_id requires a configured response-store backend"
            )
        stored_response = response_store.require(previous_response_id)
        conversation_items.extend(stored_response.conversation_items)
        runtime_messages.extend(
            runtime_messages_from_conversation_items(stored_response.conversation_items)
        )

    if isinstance(input_value, str):
        if not input_value:
            raise ValueError("responses input must not be empty")
        current_message = Message.user_text(input_value)
        conversation_items.append(current_message)
        runtime_messages.append(current_message)
        history_messages, prompt_parts = split_history_and_prompt(runtime_messages)
        return PreparedOpenAIResponseRequest(
            history_messages=history_messages,
            prompt_parts=prompt_parts,
            conversation_items=conversation_items,
        )

    if not input_value:
        raise ValueError("responses input must include at least one message")

    for item in input_value:
        translated_item, runtime_message = translate_input_item(item)
        conversation_items.append(translated_item)
        runtime_messages.append(runtime_message)

    history_messages, prompt_parts = split_history_and_prompt(runtime_messages)
    return PreparedOpenAIResponseRequest(
        history_messages=history_messages,
        prompt_parts=prompt_parts,
        conversation_items=conversation_items,
    )


def translate_input_item(
    item: OpenAIResponseInputMessageRequestModel
    | OpenAIResponseFunctionCallOutputRequestModel,
) -> tuple[
    Message | StoredOpenAIResponseFunctionCallOutput,
    Message,
]:
    """Translate a Responses input item into stored and runtime forms."""

    if isinstance(item, OpenAIResponseFunctionCallOutputRequestModel):
        return translate_function_call_output_item(item)
    message = translate_input_message(item)
    return message, message


def translate_input_message(
    message: OpenAIResponseInputMessageRequestModel,
) -> Message:
    """Translate a Responses message item into a runtime message."""

    parts = translate_input_content(message.content)
    if message.role in {"system", "developer"}:
        return Message(role=MessageRole.SYSTEM, content=parts)
    if message.role == "assistant":
        return Message(role=MessageRole.ASSISTANT, content=parts)
    if message.role == "user":
        return Message(role=MessageRole.USER, content=parts)
    raise ValueError(
        "responses input role must be one of: system, developer, user, assistant"
    )


def translate_function_call_output_item(
    item: OpenAIResponseFunctionCallOutputRequestModel,
) -> tuple[StoredOpenAIResponseFunctionCallOutput, Message]:
    """Translate a tool-result input item into stored and runtime forms."""

    output_text = serialize_function_call_output(item.output)
    stored_item = StoredOpenAIResponseFunctionCallOutput(
        call_id=item.call_id,
        output_text=output_text,
    )
    runtime_message = Message.user_text(
        build_runtime_tool_output_text(
            call_id=item.call_id,
            output_text=output_text,
        )
    )
    return stored_item, runtime_message


def translate_input_content(
    content: str | list[OpenAIResponseInputContentPartRequestModel],
) -> list[ContentPart]:
    """Translate Responses content into runtime content parts."""

    if isinstance(content, str):
        if not content:
            raise ValueError("responses input content must not be empty")
        return [ContentPart.text(content)]
    parts = [translate_input_part(part) for part in content]
    if not parts:
        raise ValueError("responses input content must include at least one text part")
    return parts


def translate_input_part(
    part: OpenAIResponseInputContentPartRequestModel,
) -> ContentPart:
    """Translate a single Responses content part into a runtime part."""

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
    if part.type == "input_file":
        if part.file_url:
            return ContentPart.text(f"[input_file url={part.file_url}]")
        if part.file_id:
            return ContentPart.text(f"[input_file id={part.file_id}]")
        raise ValueError("responses file parts require file_url or file_id")
    raise ValueError(
        "responses input currently supports text, image, audio, and file content parts"
    )


def new_output_message_id() -> str:
    """Generate an OpenAI-style message item id."""

    return f"msg_{uuid.uuid4().hex}"


def build_response_payload(
    *,
    response_id: str,
    output_message_id: str,
    created_at: int,
    model: str,
    parsed_output: ParsedOpenAIResponseOutput,
    input_items: str
    | list[
        OpenAIResponseInputMessageRequestModel
        | OpenAIResponseFunctionCallOutputRequestModel
    ]
    | None = None,
    instructions: str | None = None,
    previous_response_id: str | None = None,
    tools: list[OpenAIResponseFunctionToolRequestModel],
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel,
    parallel_tool_calls: bool,
    status: str = "completed",
    completed_at: int | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    store: bool = False,
    reasoning_effort: str | None = None,
    truncation: str = "disabled",
    metadata: dict[str, object] | None = None,
    error: object | None = None,
    incomplete_reason: str | None = None,
    output_item_status: str = "completed",
    output_items: list[
        OpenAIResponseOutputMessageResponseModel
        | OpenAIResponseFunctionCallResponseModel
    ]
    | None = None,
) -> OpenAIResponseResponseModel:
    """Build the top-level response object from parsed model output."""

    resolved_output_items = (
        build_output_items(
            parsed_output=parsed_output,
            output_message_id=output_message_id,
            item_status=output_item_status,
        )
        if output_items is None
        else output_items
    )
    return OpenAIResponseResponseModel(
        id=response_id,
        created_at=created_at,
        status=status,
        completed_at=completed_at,
        incomplete_details=(
            OpenAIResponseIncompleteDetailsResponseModel(reason=incomplete_reason)
            if incomplete_reason is not None
            else None
        ),
        input=input_items,
        max_output_tokens=max_output_tokens,
        model=model,
        output=resolved_output_items,
        instructions=instructions,
        previous_response_id=previous_response_id,
        tools=list(tools),
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        reasoning_effort=reasoning_effort,
        store=store,
        temperature=temperature,
        top_p=top_p,
        truncation=truncation,
        metadata={} if metadata is None else dict(metadata),
        error=error,
    )


def build_output_items(
    *,
    parsed_output: ParsedOpenAIResponseOutput,
    output_message_id: str,
    item_status: str,
) -> list[
    OpenAIResponseOutputMessageResponseModel | OpenAIResponseFunctionCallResponseModel
]:
    """Build response output items from parsed model output."""

    if parsed_output.is_message:
        return [
            OpenAIResponseOutputMessageResponseModel(
                id=output_message_id,
                status=item_status,
                content=[
                    OpenAIResponseOutputTextResponseModel(
                        text=parsed_output.message_text or ""
                    )
                ],
            )
        ]
    return [
        OpenAIResponseFunctionCallResponseModel(
            id=_new_function_call_item_id(),
            call_id=function_call.call_id,
            name=function_call.name,
            arguments=function_call.arguments,
            status=item_status,
        )
        for function_call in parsed_output.function_calls
    ]


def build_conversation_items(
    *,
    base_items: Sequence[
        Message
        | StoredOpenAIResponseFunctionCall
        | StoredOpenAIResponseFunctionCallOutput
    ],
    output_items: Sequence[
        OpenAIResponseOutputMessageResponseModel
        | OpenAIResponseFunctionCallResponseModel
    ],
) -> list[
    Message | StoredOpenAIResponseFunctionCall | StoredOpenAIResponseFunctionCallOutput
]:
    """Append response output items to a stored conversation history."""

    items = list(base_items)
    for output_item in output_items:
        if isinstance(output_item, OpenAIResponseOutputMessageResponseModel):
            items.append(
                Message.assistant_text(
                    "".join(part.text for part in output_item.content)
                )
            )
            continue
        items.append(
            StoredOpenAIResponseFunctionCall(
                item_id=output_item.id,
                call_id=output_item.call_id,
                name=output_item.name,
                arguments=output_item.arguments,
            )
        )
    return items


def runtime_messages_from_conversation_items(
    items: Sequence[
        Message
        | StoredOpenAIResponseFunctionCall
        | StoredOpenAIResponseFunctionCallOutput
    ],
) -> list[Message]:
    """Convert stored conversation items into runtime messages."""

    runtime_messages: list[Message] = []
    for item in items:
        if isinstance(item, Message):
            runtime_messages.append(item)
            continue
        if isinstance(item, StoredOpenAIResponseFunctionCall):
            runtime_messages.append(
                Message.assistant_text(
                    build_runtime_tool_call_text(
                        name=item.name,
                        arguments=item.arguments,
                        call_id=item.call_id,
                    )
                )
            )
            continue
        runtime_messages.append(
            Message.user_text(
                build_runtime_tool_output_text(
                    call_id=item.call_id,
                    output_text=item.output_text,
                )
            )
        )
    return runtime_messages


def split_history_and_prompt(
    runtime_messages: Sequence[Message],
) -> tuple[list[Message], list[ContentPart]]:
    """Split runtime messages into history and the current prompt payload."""

    if not runtime_messages:
        raise ValueError("responses input must include at least one input item")
    prompt_start = len(runtime_messages)
    while (
        prompt_start > 0 and runtime_messages[prompt_start - 1].role is MessageRole.USER
    ):
        prompt_start -= 1
    if prompt_start == len(runtime_messages):
        raise ValueError(
            "responses input must end in a user message or function_call_output item"
        )
    prompt_parts: list[ContentPart] = []
    for message in runtime_messages[prompt_start:]:
        prompt_parts.extend(message.content)
    if not prompt_parts:
        raise ValueError("responses input must include at least one content part")
    return list(runtime_messages[:prompt_start]), prompt_parts


def serialize_function_call_output(
    output: str | dict[str, object] | list[object] | int | float | bool | None,
) -> str:
    """Serialize a function-call output item into text for runtime replay."""

    if isinstance(output, str):
        if not output:
            raise ValueError("responses function_call_output output must not be empty")
        return output
    return json.dumps(output, separators=(",", ":"), sort_keys=True)


def _new_function_call_item_id() -> str:
    return f"fc_{uuid.uuid4().hex}"

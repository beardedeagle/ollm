"""Helpers for OpenAI-compatible Responses function-tool handling."""

import json
import uuid
from dataclasses import dataclass
from typing import cast

from ollm.server.openai_response_models import (
    OpenAIResponseFunctionToolChoiceRequestModel,
    OpenAIResponseFunctionToolRequestModel,
)

_SUPPORTED_TOOL_CHOICE_VALUES = {"auto", "none", "required"}


@dataclass(frozen=True, slots=True)
class ParsedOpenAIResponseFunctionCall:
    """Parsed function-call output item."""

    name: str
    arguments: str
    call_id: str


@dataclass(frozen=True, slots=True)
class ParsedOpenAIResponseOutput:
    """Normalized model output for the Responses compatibility layer."""

    message_text: str | None
    function_calls: tuple[ParsedOpenAIResponseFunctionCall, ...]

    @property
    def is_message(self) -> bool:
        return self.message_text is not None


def normalize_tool_choice(
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel | None,
    *,
    tools: list[OpenAIResponseFunctionToolRequestModel],
) -> str | OpenAIResponseFunctionToolChoiceRequestModel:
    """Normalize and validate the requested tool-choice mode."""

    _validate_tools(tools)
    if tool_choice is None:
        return "auto" if tools else "none"
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip().lower()
        if normalized not in _SUPPORTED_TOOL_CHOICE_VALUES:
            allowed = ", ".join(sorted(_SUPPORTED_TOOL_CHOICE_VALUES))
            raise ValueError(f"responses tool_choice must be one of: {allowed}")
        if normalized != "none" and not tools:
            raise ValueError("responses tool_choice requires at least one tool")
        return normalized
    if not tools:
        raise ValueError("responses tool_choice requires at least one tool")
    _require_known_tool_name(tool_choice.name, tools)
    return tool_choice


def build_tool_system_prompt(
    *,
    instructions: str | None,
    tools: list[OpenAIResponseFunctionToolRequestModel],
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel,
) -> str:
    """Build a strict JSON-output prompt for function-tool compatibility."""

    base_sections: list[str] = []
    if instructions and instructions.strip():
        base_sections.append(instructions.strip())

    tool_payload = [
        tool.model_dump(exclude_none=True, exclude_defaults=False) for tool in tools
    ]
    base_sections.append(
        "You are serving the OpenAI Responses API compatibility layer for oLLM."
    )
    base_sections.append(
        "When tools are available, reply with exactly one JSON object and no "
        "markdown, code fences, or explanatory text."
    )
    base_sections.append(
        "Valid reply forms are:\n"
        '{"type":"message","content":"plain assistant reply"}\n'
        "or\n"
        '{"type":"function_call","calls":[{"name":"tool_name","arguments":{}}]}'
    )
    base_sections.append(
        "Function-call arguments must be valid JSON. Use only the tools listed "
        f"here: {json.dumps(tool_payload, separators=(',', ':'), sort_keys=True)}"
    )
    base_sections.append(_tool_choice_instruction(tool_choice))
    base_sections.append(
        "Tool results may appear in prior user messages using the exact format:\n"
        "[function_call_output call_id=<id>]\n"
        "<tool output text>\n"
        "[/function_call_output]"
    )
    return "\n\n".join(base_sections)


def parse_model_output(
    raw_text: str,
    *,
    tools: list[OpenAIResponseFunctionToolRequestModel],
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel,
    parallel_tool_calls: bool,
) -> ParsedOpenAIResponseOutput:
    """Parse model text into a message or one or more function-call items."""

    if not tools:
        message_text = raw_text.strip()
        if not message_text:
            raise ValueError("responses model output must not be empty")
        return ParsedOpenAIResponseOutput(message_text=message_text, function_calls=())

    json_candidate = _extract_json_candidate(raw_text)
    if json_candidate is None:
        if _tool_call_required(tool_choice):
            raise ValueError(
                "responses tool call generation requires structured JSON output"
            )
        message_text = raw_text.strip()
        if not message_text:
            raise ValueError("responses model output must not be empty")
        return ParsedOpenAIResponseOutput(message_text=message_text, function_calls=())

    payload = _load_json_object(json_candidate)
    parsed = _parse_structured_payload(
        payload,
        tools=tools,
        parallel_tool_calls=parallel_tool_calls,
    )
    _validate_tool_choice_against_output(parsed, tool_choice=tool_choice)
    return parsed


def build_runtime_tool_call_text(*, name: str, arguments: str, call_id: str) -> str:
    """Render a function-call item into a stable runtime-history message."""

    return (
        f"[function_call name={name} call_id={call_id}]\n{arguments}\n[/function_call]"
    )


def build_runtime_tool_output_text(*, call_id: str, output_text: str) -> str:
    """Render a tool-result item into a stable runtime-history message."""

    return (
        f"[function_call_output call_id={call_id}]\n"
        f"{output_text}\n"
        "[/function_call_output]"
    )


def _validate_tools(tools: list[OpenAIResponseFunctionToolRequestModel]) -> None:
    seen_names: set[str] = set()
    for tool in tools:
        if not tool.name.strip():
            raise ValueError("responses tools must have a non-empty name")
        normalized_name = tool.name.strip()
        if normalized_name in seen_names:
            raise ValueError(f"duplicate responses tool name '{normalized_name}'")
        seen_names.add(normalized_name)


def _tool_choice_instruction(
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel,
) -> str:
    if isinstance(tool_choice, str):
        if tool_choice == "none":
            return "Do not call tools. Always emit a message JSON object."
        if tool_choice == "required":
            return "You must emit at least one function_call JSON item."
        if tool_choice == "auto":
            return (
                "Choose between a direct message and one or more function_call items "
                "based on whether tool use is necessary."
            )
    if not isinstance(tool_choice, OpenAIResponseFunctionToolChoiceRequestModel):
        raise ValueError("responses tool_choice did not resolve to a supported value")
    return (
        "You must emit at least one function_call item and every call must use "
        f"the tool named '{tool_choice.name}'."
    )


def _extract_json_candidate(raw_text: str) -> str | None:
    stripped = raw_text.strip()
    if not stripped:
        return None
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = "\n".join(stripped.splitlines()[1:-1]).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    first_open = stripped.find("{")
    last_close = stripped.rfind("}")
    if first_open == -1 or last_close == -1 or last_close <= first_open:
        return None
    return stripped[first_open : last_close + 1]


def _load_json_object(value: str) -> dict[str, object]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("responses structured output must be valid JSON") from exc
    if not isinstance(loaded, dict):
        raise ValueError("responses structured output must be a JSON object")
    return cast(dict[str, object], loaded)


def _parse_structured_payload(
    payload: dict[str, object],
    *,
    tools: list[OpenAIResponseFunctionToolRequestModel],
    parallel_tool_calls: bool,
) -> ParsedOpenAIResponseOutput:
    payload_type = payload.get("type")
    if payload_type == "message":
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "responses message payloads require a non-empty string content"
            )
        return ParsedOpenAIResponseOutput(
            message_text=content.strip(),
            function_calls=(),
        )
    if payload_type == "function_call":
        calls_value = payload.get("calls")
        calls_payload = _parse_calls_payload(payload, calls_value)
        if not parallel_tool_calls and len(calls_payload) > 1:
            raise ValueError(
                "responses parallel_tool_calls=false does not allow multiple function calls"
            )
        return ParsedOpenAIResponseOutput(
            message_text=None,
            function_calls=tuple(
                _parse_function_call_item(call_payload, tools=tools)
                for call_payload in calls_payload
            ),
        )
    raise ValueError(
        "responses structured output must use type 'message' or 'function_call'"
    )


def _parse_calls_payload(
    payload: dict[str, object],
    calls_value: object,
) -> list[dict[str, object]]:
    if isinstance(calls_value, list):
        calls_payload = []
        for index, item in enumerate(calls_value):
            if not isinstance(item, dict):
                raise ValueError(
                    f"responses function_call calls[{index}] must be an object"
                )
            calls_payload.append(cast(dict[str, object], item))
        if not calls_payload:
            raise ValueError(
                "responses function_call payload must include at least one call"
            )
        return calls_payload
    if "name" in payload and "arguments" in payload:
        return [payload]
    raise ValueError("responses function_call payload must include a calls list")


def _parse_function_call_item(
    payload: dict[str, object],
    *,
    tools: list[OpenAIResponseFunctionToolRequestModel],
) -> ParsedOpenAIResponseFunctionCall:
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("responses function_call items require a non-empty name")
    normalized_name = name.strip()
    _require_known_tool_name(normalized_name, tools)
    arguments = _normalize_arguments(payload.get("arguments"))
    call_id_value = payload.get("call_id")
    call_id = (
        call_id_value.strip()
        if isinstance(call_id_value, str) and call_id_value.strip()
        else f"call_{uuid.uuid4().hex}"
    )
    return ParsedOpenAIResponseFunctionCall(
        name=normalized_name,
        arguments=arguments,
        call_id=call_id,
    )


def _normalize_arguments(value: object) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("responses function_call arguments must not be empty")
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            return json.dumps(stripped)
        return json.dumps(loaded, separators=(",", ":"), sort_keys=True)
    if isinstance(value, dict) or isinstance(value, list):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    if value is None or isinstance(value, (int, float, bool)):
        return json.dumps(value)
    raise ValueError("responses function_call arguments must be JSON-compatible data")


def _require_known_tool_name(
    tool_name: str,
    tools: list[OpenAIResponseFunctionToolRequestModel],
) -> None:
    available_names = {tool.name for tool in tools}
    if tool_name not in available_names:
        allowed_names = ", ".join(sorted(available_names))
        raise ValueError(
            f"responses function_call requested unknown tool '{tool_name}'. "
            f"Known tools: {allowed_names}"
        )


def _tool_call_required(
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel,
) -> bool:
    return tool_choice == "required" or isinstance(
        tool_choice, OpenAIResponseFunctionToolChoiceRequestModel
    )


def _validate_tool_choice_against_output(
    parsed: ParsedOpenAIResponseOutput,
    *,
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel,
) -> None:
    if tool_choice == "none" and not parsed.is_message:
        raise ValueError("responses tool_choice=none does not allow function calls")
    if tool_choice == "required" and not parsed.function_calls:
        raise ValueError("responses tool_choice=required requires a function call")
    if isinstance(tool_choice, OpenAIResponseFunctionToolChoiceRequestModel):
        if not parsed.function_calls:
            raise ValueError(
                "responses explicit function tool_choice requires a function call"
            )
        for call in parsed.function_calls:
            if call.name != tool_choice.name:
                raise ValueError(
                    "responses explicit function tool_choice requires every call to "
                    f"use '{tool_choice.name}'"
                )

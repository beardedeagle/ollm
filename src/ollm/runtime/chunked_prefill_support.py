"""Support helpers for chunked prompt-ingestion strategies."""

import re
from collections.abc import Iterable
from inspect import Parameter, signature

import torch

from ollm.app.types import ContentKind, Message
from ollm.runtime.errors import PromptExecutionError
from ollm.runtime.generation_support import render_plain_prompt
from ollm.runtime.loaded_runtime import LoadedRuntime
from ollm.runtime.output_control import suppress_module_prints


def render_prompt_text(runtime: LoadedRuntime, messages: list[Message]) -> str:
    transformers_messages = [
        message.as_transformers_message(
            structured_content=runtime.processor is not None
        )
        for message in messages
    ]
    if runtime.processor is not None and hasattr(
        runtime.processor, "apply_chat_template"
    ):
        rendered = runtime.processor.apply_chat_template(
            transformers_messages,
            add_generation_prompt=True,
            tokenize=False,
            return_dict=False,
            return_tensors=None,
        )
        if isinstance(rendered, str):
            return rendered
    if hasattr(runtime.tokenizer, "apply_chat_template"):
        try:
            rendered = runtime.tokenizer.apply_chat_template(
                transformers_messages,
                tokenize=False,
                add_generation_prompt=True,
                return_dict=False,
            )
            if isinstance(rendered, str):
                return rendered
        except (TypeError, ValueError, AttributeError):
            pass
    return render_plain_prompt(messages)


def resolve_stream_tokenizer(runtime: LoadedRuntime):
    if (
        runtime.processor is not None
        and getattr(runtime.processor, "tokenizer", None) is not None
    ):
        return runtime.processor.tokenizer
    return runtime.tokenizer


def prepare_static_inputs(
    runtime: LoadedRuntime,
    messages: list[Message],
) -> dict[str, object]:
    if runtime.processor is None:
        return {}
    custom_builder = getattr(
        runtime.processor, "prepare_chunked_prefill_static_inputs", None
    )
    if callable(custom_builder):
        prepared = custom_builder(messages, runtime.device)
        return dict(prepared)
    image_values = [
        part.value
        for message in messages
        for part in message.content
        if part.kind is ContentKind.IMAGE
    ]
    audio_values = [
        part.value
        for message in messages
        for part in message.content
        if part.kind is ContentKind.AUDIO
    ]
    if not image_values and not audio_values:
        return {}
    prepared = call_processor_for_static_inputs(
        processor=runtime.processor,
        image_values=image_values,
        audio_values=audio_values,
        device=runtime.device,
    )
    prepared.pop("input_ids", None)
    prepared.pop("attention_mask", None)
    prepared.pop("token_type_ids", None)
    return prepared


def call_processor_for_static_inputs(
    *,
    processor,
    image_values: list[str],
    audio_values: list[str],
    device: torch.device,
) -> dict[str, object]:
    processor_signature = signature(processor.__call__)
    accepts_kwargs = any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in processor_signature.parameters.values()
    )
    kwargs: dict[str, object] = {"return_tensors": "pt"}
    if image_values and ("images" in processor_signature.parameters or accepts_kwargs):
        kwargs["images"] = image_values
    if audio_values:
        if "audios" in processor_signature.parameters or accepts_kwargs:
            kwargs["audios"] = audio_values
        elif "audio" in processor_signature.parameters:
            kwargs["audio"] = audio_values
    if "text" in processor_signature.parameters or accepts_kwargs:
        kwargs["text"] = [""]
    prepared = processor(**kwargs)
    return move_input_mapping(prepared, device)


def move_input_mapping(value: object, device: torch.device) -> dict[str, object]:
    to_method = getattr(value, "to", None)
    if callable(to_method):
        moved = to_method(device)
        if isinstance(moved, dict):
            return dict(moved)
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            if isinstance(item, torch.Tensor):
                result[str(key)] = item.to(device)
            else:
                result[str(key)] = item
        return result
    raise PromptExecutionError("Chunked prompt ingestion expected mapping-like inputs.")


def prompt_token_id_pieces(tokenizer, rendered_prompt: str) -> Iterable[list[int]]:
    custom_streamer = getattr(tokenizer, "stream_tokenize_prompt", None)
    if callable(custom_streamer):
        for piece in custom_streamer(rendered_prompt):
            ids = list(piece)
            if ids:
                yield ids
        return
    for piece_text in prompt_piece_texts(tokenizer, rendered_prompt):
        piece_ids = tokenize_prompt_piece(tokenizer, piece_text)
        if piece_ids:
            yield piece_ids


def prompt_piece_texts(tokenizer, rendered_prompt: str) -> tuple[str, ...]:
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    pre_tokenizer = (
        None
        if backend_tokenizer is None
        else getattr(backend_tokenizer, "pre_tokenizer", None)
    )
    if pre_tokenizer is not None and hasattr(pre_tokenizer, "pre_tokenize_str"):
        pieces = pre_tokenizer.pre_tokenize_str(rendered_prompt)
        if pieces:
            return tuple(text for text, _offset in pieces if text)
    regex_pieces = tuple(
        match.group(0)
        for match in re.finditer(r"\S+\s*|\s+", rendered_prompt)
        if match.group(0)
    )
    if regex_pieces:
        return regex_pieces
    return (rendered_prompt,)


def tokenize_prompt_piece(tokenizer, piece_text: str) -> list[int]:
    try:
        encoded = tokenizer(
            piece_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
    except TypeError:
        encoded = tokenizer(piece_text)
    if isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return [int(token_id) for token_id in input_ids[0]]
            return [int(token_id) for token_id in input_ids]
        if isinstance(input_ids, torch.Tensor):
            return [int(token_id) for token_id in input_ids.reshape(-1).tolist()]
    if isinstance(encoded, list):
        return [int(token_id) for token_id in encoded]
    raise PromptExecutionError(
        "Chunked prompt tokenization produced unsupported input ids."
    )


def run_causal_prefill_chunk(
    *,
    runtime: LoadedRuntime,
    forward_method,
    static_inputs: dict[str, object],
    chunk_ids: list[int],
    prefill_cache: object,
    prefix_token_count: int,
    strategy_label: str,
) -> object:
    forward_inputs: dict[str, object] = dict(static_inputs)
    forward_inputs["input_ids"] = token_tensor(chunk_ids, device=runtime.device)
    forward_inputs["attention_mask"] = ones_attention_mask(
        token_count=prefix_token_count + len(chunk_ids),
        device=runtime.device,
    )
    forward_inputs["use_cache"] = True
    forward_inputs["cache_position"] = torch.arange(
        prefix_token_count,
        prefix_token_count + len(chunk_ids),
        device=runtime.device,
        dtype=torch.long,
    )
    if prefill_cache is not None:
        forward_inputs["past_key_values"] = prefill_cache
    filtered_inputs = filter_supported_forward_inputs(forward_method, forward_inputs)
    with torch.inference_mode():
        with suppress_module_prints(runtime.backend.print_suppression_modules):
            outputs = forward_method(**filtered_inputs)
    next_cache = getattr(outputs, "past_key_values", None)
    if next_cache is None:
        raise PromptExecutionError(
            "Chunked prompt-ingestion strategy "
            f"{strategy_label!r} requires a runtime that returns past_key_values."
        )
    return next_cache


def filter_supported_forward_inputs(
    forward_method,
    inputs: dict[str, object],
) -> dict[str, object]:
    method_signature = signature(forward_method)
    if any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in method_signature.parameters.values()
    ):
        return inputs
    supported_keys = set(method_signature.parameters)
    return {key: value for key, value in inputs.items() if key in supported_keys}


def token_tensor(token_ids: list[int], *, device: torch.device) -> torch.Tensor:
    return torch.tensor([token_ids], device=device, dtype=torch.long)


def ones_attention_mask(token_count: int, *, device: torch.device) -> torch.Tensor:
    return torch.ones((1, token_count), device=device, dtype=torch.long)


def prompt_token_count(inputs: dict[str, object]) -> int:
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise PromptExecutionError(
            "Chunked prompt preparation expected tensor-backed input_ids."
        )
    return int(input_ids.shape[-1])

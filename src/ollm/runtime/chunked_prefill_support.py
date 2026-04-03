"""Support helpers for chunked prompt-ingestion strategies."""

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import cast

import torch

from ollm.app.types import ContentKind, Message
from ollm.runtime.errors import PromptExecutionError
from ollm.runtime.generation_support import render_plain_prompt
from ollm.runtime.loaded_runtime import LoadedRuntime
from ollm.runtime.output_control import suppress_module_prints

_MAX_STREAM_TOKENIZER_BATCH_PIECES = 32


@dataclass(slots=True)
class StreamedTokenBuffer:
    total_token_count: int = 0
    _tokens: list[int] = field(default_factory=list)
    _head_index: int = 0

    @property
    def buffered_token_count(self) -> int:
        return len(self._tokens) - self._head_index

    def append_piece(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        self._tokens.extend(token_ids)
        self.total_token_count += len(token_ids)

    def pop_chunk(self, token_count: int) -> list[int]:
        if token_count < 1:
            raise ValueError("token_count must be at least 1")
        if token_count > self.buffered_token_count:
            raise ValueError("token_count exceeds buffered token count")
        chunk = self._tokens[self._head_index : self._head_index + token_count]
        self._head_index += token_count
        self._compact()
        return chunk

    def remaining_tokens(self) -> list[int]:
        return self._tokens[self._head_index :]

    def _compact(self) -> None:
        if self._head_index < 1024:
            return
        if self._head_index * 2 < len(self._tokens):
            return
        self._tokens = self._tokens[self._head_index :]
        self._head_index = 0


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
        try:
            rendered = runtime.processor.apply_chat_template(
                transformers_messages,
                add_generation_prompt=True,
                tokenize=False,
                return_dict=False,
                return_tensors=None,
            )
            if isinstance(rendered, str):
                return rendered
        except (TypeError, ValueError, AttributeError):
            pass
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
    try:
        processor_signature = signature(processor.__call__)
    except (TypeError, ValueError):
        processor_signature = None
        accepts_kwargs = True
    else:
        accepts_kwargs = any(
            parameter.kind is Parameter.VAR_KEYWORD
            for parameter in processor_signature.parameters.values()
        )
    kwargs: dict[str, object] = {}
    if accepts_kwargs or (
        processor_signature is not None
        and "return_tensors" in processor_signature.parameters
    ):
        kwargs["return_tensors"] = "pt"
    if image_values and (
        accepts_kwargs
        or (
            processor_signature is not None
            and "images" in processor_signature.parameters
        )
    ):
        kwargs["images"] = image_values
    if audio_values:
        if accepts_kwargs or (
            processor_signature is not None
            and "audios" in processor_signature.parameters
        ):
            kwargs["audios"] = audio_values
        elif (
            processor_signature is not None
            and "audio" in processor_signature.parameters
        ):
            kwargs["audio"] = audio_values
    if accepts_kwargs or (
        processor_signature is not None and "text" in processor_signature.parameters
    ):
        kwargs["text"] = [""]
    prepared = processor(**kwargs)
    return move_input_mapping(prepared, device)


def move_input_mapping(value: object, device: torch.device) -> dict[str, object]:
    to_method = getattr(value, "to", None)
    if callable(to_method):
        moved = to_method(device)
        if isinstance(moved, Mapping):
            return dict(moved)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if isinstance(item, torch.Tensor):
                result[str(key)] = item.to(device)
            else:
                result[str(key)] = item
        return result
    raise PromptExecutionError("Chunked prompt ingestion expected mapping-like inputs.")


def prompt_token_id_pieces(
    tokenizer,
    rendered_prompt: str,
    *,
    piece_batch_limit: int = 1,
) -> Iterable[list[int]]:
    if piece_batch_limit < 1:
        raise ValueError("piece_batch_limit must be at least 1")
    custom_streamer = getattr(tokenizer, "stream_tokenize_prompt", None)
    if callable(custom_streamer):
        for piece in custom_streamer(rendered_prompt):
            ids = list(piece)
            if ids:
                yield ids
        return
    piece_batch: list[str] = []
    for piece_text in prompt_piece_texts(tokenizer, rendered_prompt):
        piece_batch.append(piece_text)
        if len(piece_batch) < piece_batch_limit:
            continue
        yield from tokenize_prompt_piece_batch(tokenizer, piece_batch)
        piece_batch.clear()
    if piece_batch:
        yield from tokenize_prompt_piece_batch(tokenizer, piece_batch)


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
        encode_method = getattr(tokenizer, "encode", None)
        if callable(encode_method):
            encoded = encode_method(piece_text, add_special_tokens=False)
        else:
            encoded = tokenizer(piece_text)
    token_batches = _extract_token_batches(encoded, expected_batch_size=1)
    if token_batches is not None:
        return token_batches[0]
    raise PromptExecutionError(
        "Chunked prompt tokenization produced unsupported input ids."
    )


def tokenize_prompt_piece_batch(
    tokenizer,
    piece_texts: list[str],
) -> tuple[list[int], ...]:
    if not piece_texts:
        return ()
    if len(piece_texts) == 1:
        return (tokenize_prompt_piece(tokenizer, piece_texts[0]),)
    try:
        encoded = tokenizer(
            piece_texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )
    except TypeError:
        token_batches = _encode_piece_batch_with_backend_tokenizer(
            tokenizer, piece_texts
        )
        if token_batches is not None:
            return token_batches
        return tuple(
            tokenize_prompt_piece(tokenizer, piece_text) for piece_text in piece_texts
        )
    token_batches = _extract_token_batches(
        encoded,
        expected_batch_size=len(piece_texts),
    )
    if token_batches is not None:
        return token_batches
    return tuple(
        tokenize_prompt_piece(tokenizer, piece_text) for piece_text in piece_texts
    )


def stream_tokenizer_piece_batch_limit(chunk_tokens: int) -> int:
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be at least 1")
    return max(1, min(_MAX_STREAM_TOKENIZER_BATCH_PIECES, chunk_tokens // 4 or 1))


def _encode_piece_batch_with_backend_tokenizer(
    tokenizer,
    piece_texts: list[str],
) -> tuple[list[int], ...] | None:
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    encode_batch = (
        None
        if backend_tokenizer is None
        else getattr(backend_tokenizer, "encode_batch", None)
    )
    if not callable(encode_batch):
        return None
    try:
        encodings = encode_batch(piece_texts, add_special_tokens=False)
    except TypeError:
        try:
            encodings = encode_batch(piece_texts)
        except TypeError:
            return None
    token_batches: list[list[int]] = []
    for encoding in encodings:
        token_ids = getattr(encoding, "ids", None)
        if token_ids is None:
            return None
        token_batches.append([int(token_id) for token_id in token_ids])
    return tuple(token_batches)


def _extract_token_batches(
    encoded: object,
    *,
    expected_batch_size: int,
) -> tuple[list[int], ...] | None:
    input_ids = encoded
    if isinstance(encoded, Mapping):
        mapping = cast(Mapping[str, object], encoded)
        input_ids = mapping.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 1:
            if expected_batch_size != 1:
                return None
            return ([int(token_id) for token_id in input_ids.tolist()],)
        if input_ids.ndim == 2 and input_ids.shape[0] == expected_batch_size:
            return tuple(
                [int(token_id) for token_id in row.tolist()] for row in input_ids
            )
        return None
    if isinstance(input_ids, list):
        if not input_ids:
            return tuple([] for _ in range(expected_batch_size))
        if isinstance(input_ids[0], list):
            row_batches = cast(list[list[object]], input_ids)
            if len(input_ids) != expected_batch_size:
                return None
            return tuple(_coerce_token_id_list(row) for row in row_batches)
        if expected_batch_size != 1:
            return None
        token_ids = cast(list[object], input_ids)
        return (_coerce_token_id_list(token_ids),)
    return None


def _coerce_token_id_list(values: list[object]) -> list[int]:
    token_ids: list[int] = []
    for value in values:
        if not isinstance(value, int):
            raise PromptExecutionError(
                "Chunked prompt tokenization produced non-integer token ids."
            )
        token_ids.append(value)
    return token_ids


def run_causal_prefill_chunk(
    *,
    runtime: LoadedRuntime,
    forward_method,
    forward_input_filter: Callable[[dict[str, object]], dict[str, object]],
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
    filtered_inputs = forward_input_filter(forward_inputs)
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


def build_forward_input_filter(
    forward_method,
) -> Callable[[dict[str, object]], dict[str, object]]:
    try:
        method_signature = signature(forward_method)
    except (TypeError, ValueError):
        return lambda inputs: inputs
    if any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in method_signature.parameters.values()
    ):
        return lambda inputs: inputs
    supported_keys = frozenset(method_signature.parameters)
    return lambda inputs: {
        key: value for key, value in inputs.items() if key in supported_keys
    }


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

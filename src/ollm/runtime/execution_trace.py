"""Typed runtime execution tracing for benchmark-style measurements."""

import time
from dataclasses import dataclass

import torch

from ollm.app.types import PromptRequest
from ollm.kv_cache.state import KVCacheStateSnapshot
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.chunked_prefill import ChunkedPrefillScopeSurface
from ollm.runtime.errors import PromptExecutionError
from ollm.runtime.generation import (
    build_runtime_generate_kwargs,
    decode_runtime_response,
    prepare_runtime_generate_inputs,
    validate_runtime_request,
)
from ollm.runtime.generation_config_support import temporary_generation_config
from ollm.runtime.generation_support import (
    extract_cache_state_snapshot,
    normalize_generate_inputs,
    require_tensor,
)
from ollm.runtime.loaded_runtime import LoadedRuntime
from ollm.runtime.output_control import suppress_module_prints
from ollm.runtime.streaming import BufferedTextStreamer


@dataclass(frozen=True, slots=True)
class RuntimeExecutionTrace:
    generation_started_at: float
    prompt_token_count: int
    decode_prefix_token_count: int
    output_token_count: int
    response_text: str
    cache_state: KVCacheStateSnapshot | None
    chunked_prefill: ChunkedPrefillScopeSurface


def execute_request_with_trace(
    *,
    runtime: LoadedRuntime,
    request: PromptRequest,
    streamer: BufferedTextStreamer | None = None,
) -> RuntimeExecutionTrace:
    """Execute one prompt request and return only benchmark-relevant facts."""

    validate_runtime_request(runtime, request)
    if request.generation_config.seed is not None:
        torch.manual_seed(request.generation_config.seed)

    generate_kwargs, generation_config = build_runtime_generate_kwargs(
        runtime, request, streamer
    )
    generation_started_at = time.perf_counter()
    prepared_result = prepare_runtime_generate_inputs(
        runtime,
        request,
        generate_kwargs,
    )
    prompt_token_count = prepared_result.prompt_token_count
    prepared_inputs = normalize_generate_inputs(prepared_result.inputs)
    prepared_generate_kwargs = prepared_result.generate_kwargs
    chunked_prefill = prepared_result.scope
    outputs, effective_generate_kwargs = _generate_outputs(
        runtime=runtime,
        prepared_inputs=prepared_inputs,
        prepared_generate_kwargs=prepared_generate_kwargs,
        generation_config=generation_config,
    )
    if hasattr(outputs, "detach"):
        outputs = outputs.detach()
    cpu_outputs = outputs.cpu()
    response_text = decode_runtime_response(runtime, prepared_inputs, cpu_outputs)
    if streamer is not None and not response_text.strip():
        response_text = streamer.text
    decode_prefix_token_count = _decode_prefix_token_count(prepared_inputs)
    output_token_count = _count_output_tokens(
        runtime=runtime,
        outputs=cpu_outputs,
        decode_prefix_token_count=decode_prefix_token_count,
    )
    cache_state = extract_cache_state_snapshot(
        effective_generate_kwargs.get("past_key_values")
    )
    return RuntimeExecutionTrace(
        generation_started_at=generation_started_at,
        prompt_token_count=prompt_token_count,
        decode_prefix_token_count=decode_prefix_token_count,
        output_token_count=output_token_count,
        response_text=response_text,
        cache_state=cache_state,
        chunked_prefill=chunked_prefill,
    )


def _generate_outputs(
    *,
    runtime: LoadedRuntime,
    prepared_inputs: dict[str, object],
    prepared_generate_kwargs: dict[str, object],
    generation_config: object,
) -> tuple[torch.Tensor, dict[str, object]]:
    try:
        return (
            _run_model_generate(
                runtime=runtime,
                prepared_inputs=prepared_inputs,
                prepared_generate_kwargs=prepared_generate_kwargs,
                generation_config=generation_config,
            ),
            prepared_generate_kwargs,
        )
    except TypeError as exc:
        if "streamer" not in str(exc):
            raise
        retry_generate_kwargs = dict(prepared_generate_kwargs)
        retry_generate_kwargs.pop("streamer", None)
        return (
            _run_model_generate(
                runtime=runtime,
                prepared_inputs=prepared_inputs,
                prepared_generate_kwargs=retry_generate_kwargs,
                generation_config=generation_config,
            ),
            retry_generate_kwargs,
        )


def _run_model_generate(
    *,
    runtime: LoadedRuntime,
    prepared_inputs: dict[str, object],
    prepared_generate_kwargs: dict[str, object],
    generation_config: object,
) -> torch.Tensor:
    with torch.inference_mode():
        with suppress_module_prints(runtime.backend.print_suppression_modules):
            with temporary_generation_config(runtime.model, generation_config):
                return runtime.model.generate(
                    **prepared_inputs,
                    **prepared_generate_kwargs,
                )


def _count_prompt_tokens(inputs: dict[str, object]) -> int:
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise PromptExecutionError("Benchmark probe expected tensor-backed input_ids")
    return int(input_ids.shape[0] if input_ids.ndim == 1 else input_ids.shape[-1])


def _decode_prefix_token_count(inputs: dict[str, object]) -> int:
    input_ids = require_tensor(inputs["input_ids"])
    return int(input_ids.shape[-1])


def _count_output_tokens(
    *,
    runtime: LoadedRuntime,
    outputs: torch.Tensor,
    decode_prefix_token_count: int,
) -> int:
    if runtime.plan.generic_model_kind is GenericModelKind.SEQ2SEQ_LM:
        return int(outputs.shape[-1])
    return max(0, int(outputs.shape[-1] - decode_prefix_token_count))

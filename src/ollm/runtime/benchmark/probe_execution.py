"""Private helpers for runtime benchmark probe execution."""

import time
from typing import cast

import torch

from ollm.app.types import Message, PromptRequest
from ollm.kv_cache.matrix import (
    build_kv_cache_adaptation_surface,
    resolve_kv_cache_base_dir,
)
from ollm.kv_cache.state import KVCacheStateSnapshot
from ollm.kv_cache.strategy import is_disk_backed_kv_cache_strategy, kv_cache_root
from ollm.runtime.benchmark.probe_types import (
    EventTimingSummary,
    NativeRuntimeProfile,
    RequestProbeExecution,
    RequestProbeMetrics,
)
from ollm.runtime.benchmark.resources import cache_dir_size_mb, measure_stage
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.errors import PromptExecutionError
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.generation_config_support import temporary_generation_config
from ollm.runtime.generation_support import normalize_generate_inputs
from ollm.runtime.loaded_runtime import LoadedRuntime
from ollm.runtime.output_control import suppress_module_prints
from ollm.runtime.streaming import BufferedTextStreamer
from ollm.utils import Stats

_STORAGE_PATH_BY_EVENT = {
    "gds_read": "gds",
    "safetensor_read": "safetensor-io",
    "safetensor_pread": "safetensor-io",
    "offloaded_cpu_to_cuda": "cpu-offloaded-artifacts",
    "kvload": "disk-kv-cache",
    "kvsave": "disk-kv-cache",
    "kvcompact": "disk-kv-cache",
    "torch_file_load": "torch-artifact-io",
}


class TimedBufferedTextStreamer(BufferedTextStreamer):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer,
            sink=_NullStreamSink(),
            skip_prompt=True,
            skip_special_tokens=False,
        )
        self._token_timestamps: list[float] = []

    @property
    def token_timestamps(self) -> tuple[float, ...]:
        return tuple(self._token_timestamps)

    def put(self, value) -> None:
        tensor_value = value
        if hasattr(tensor_value, "shape"):
            if len(tensor_value.shape) > 1 and tensor_value.shape[0] > 1:
                raise ValueError("TimedBufferedTextStreamer only supports batch size 1")
            if len(tensor_value.shape) > 1:
                tensor_value = tensor_value[0]
        skip_prompt_tokens = bool(self.skip_prompt and self.next_tokens_are_prompt)
        if not skip_prompt_tokens:
            token_count = (
                tensor_value.numel()
                if isinstance(tensor_value, torch.Tensor)
                else len(list(tensor_value))
            )
            now = time.perf_counter()
            self._token_timestamps.extend(now for _ in range(token_count))
        super().put(value)


class _NullStreamSink:
    def on_status(self, message: str) -> None:
        del message

    def on_text(self, text: str) -> None:
        del text

    def on_complete(self, text: str) -> None:
        del text


def execute_request_probe(
    *,
    runtime: LoadedRuntime,
    request: PromptRequest,
) -> RequestProbeExecution:
    executor = RuntimeExecutor()
    executor._validate_request(runtime, request)
    inputs = executor._build_inputs(runtime, request.messages)
    prompt_tokens = _count_prompt_tokens(inputs)
    streamer = TimedBufferedTextStreamer(runtime.tokenizer)
    generate_kwargs, generation_config = executor._build_generate_kwargs(
        runtime, request, streamer
    )
    cache_mode = _cache_mode(runtime, request)
    kv_cache_strategy = _kv_cache_strategy(runtime, request)
    _clear_backend_stats(runtime)
    normalized_inputs = normalize_generate_inputs(inputs)
    generation_result, generation_ms, generation_resources = measure_stage(
        runtime.config.device,
        lambda: _generate_outputs(
            executor,
            runtime,
            request,
            normalized_inputs,
            generate_kwargs,
            generation_config,
        ),
        sample_accelerator_utilization=True,
    )
    generated = cast(
        tuple[object, float, dict[str, object], dict[str, object]], generation_result
    )
    output_tensor = cast(torch.Tensor, generated[0])
    generation_started = generated[1]
    prepared_inputs = generated[2]
    prepared_generate_kwargs = generated[3]
    if hasattr(output_tensor, "detach"):
        output_tensor = output_tensor.detach()
    cpu_outputs = output_tensor.cpu()
    response_text = executor._decode_response(runtime, prepared_inputs, cpu_outputs)
    cache_state = _extract_cache_state_snapshot(
        prepared_generate_kwargs.get("past_key_values")
    )
    offload_cpu_applied_indices = tuple(
        int(layer_idx)
        for layer_idx in runtime.plan.details.get(
            "offload_cpu_applied_indices", ""
        ).split(",")
        if layer_idx
    )
    output_tokens = _count_output_tokens(runtime, prepared_inputs, cpu_outputs)
    time_to_first_token_ms = None
    if streamer.token_timestamps:
        time_to_first_token_ms = round(
            (streamer.token_timestamps[0] - generation_started) * 1000.0,
            6,
        )
    prompt_tokens_per_second = None
    if time_to_first_token_ms is not None and time_to_first_token_ms > 0:
        prompt_tokens_per_second = round(
            prompt_tokens / (time_to_first_token_ms / 1000.0),
            6,
        )
    output_tokens_per_second = None
    if generation_ms > 0:
        output_tokens_per_second = round(
            output_tokens / (generation_ms / 1000.0),
            6,
        )
    cache_dir_size = None
    if kv_cache_strategy is not None and is_disk_backed_kv_cache_strategy(
        kv_cache_strategy
    ):
        cache_base_dir = resolve_kv_cache_base_dir(
            cache_dir=request.runtime_config.resolved_cache_dir(),
            lifecycle=request.runtime_config.resolved_kv_cache_lifecycle(),
            model_reference=runtime.resolved_model.reference.raw,
            normalized_name=runtime.resolved_model.normalized_name,
            backend_id=runtime.plan.backend_id or "unknown",
            specialization_provider_id=runtime.plan.specialization_provider_id,
        )
        cache_dir_size = cache_dir_size_mb(
            kv_cache_root(cache_base_dir, kv_cache_strategy)
        )
    allocator_gap_mb = None
    allocator_gap_ratio = None
    if (
        generation_resources.accelerator_peak_reserved_mb is not None
        and generation_resources.accelerator_peak_mb is not None
    ):
        allocator_gap_mb = round(
            generation_resources.accelerator_peak_reserved_mb
            - generation_resources.accelerator_peak_mb,
            6,
        )
        if generation_resources.accelerator_peak_reserved_mb > 0:
            allocator_gap_ratio = round(
                allocator_gap_mb / generation_resources.accelerator_peak_reserved_mb,
                6,
            )
    native_runtime_profile = _collect_native_runtime_profile(runtime)
    kv_cache_adaptation = None
    if kv_cache_strategy is not None and cache_state is not None:
        kv_cache_adaptation = build_kv_cache_adaptation_surface(
            adaptation_mode=request.runtime_config.resolved_kv_cache_adaptation_mode(),
            current_strategy=kv_cache_strategy,
            persisted_artifact_count=cache_state.persisted_artifact_count,
            spill_count=cache_state.spill_count,
            resident_bytes=cache_state.resident_bytes,
            hot_bytes=cache_state.hot_bytes,
        )
    return RequestProbeExecution(
        metrics=RequestProbeMetrics(
            total_ms=round(generation_ms, 6),
            generation_ms=round(generation_ms, 6),
            time_to_first_token_ms=time_to_first_token_ms,
            inter_token_latencies_ms=_inter_token_latencies(streamer.token_timestamps),
            prompt_tokens=prompt_tokens,
            prompt_tokens_per_second=prompt_tokens_per_second,
            output_tokens=output_tokens,
            output_tokens_per_second=output_tokens_per_second,
            cache_mode=cache_mode,
            kv_cache_strategy=kv_cache_strategy,
            strategy_selector_profile=runtime.plan.details.get(
                "strategy_selector_profile"
            ),
            strategy_selector_rule_id=runtime.plan.details.get(
                "strategy_selector_rule_id"
            ),
            strategy_selector_requested_override=runtime.plan.details.get(
                "strategy_selector_requested_override"
            ),
            strategy_selector_selected_kv_cache_strategy=runtime.plan.details.get(
                "strategy_selector_selected_kv_cache_strategy"
            ),
            strategy_selector_applied_kv_cache_strategy=runtime.plan.details.get(
                "strategy_selector_applied_kv_cache_strategy"
            ),
            strategy_selector_fallback_chain=tuple(
                item
                for item in runtime.plan.details.get(
                    "strategy_selector_fallback_chain", ""
                ).split(",")
                if item
            ),
            offload_cpu_policy=runtime.plan.details.get("offload_cpu_resolved_policy"),
            offload_cpu_requested_layers=_optional_int_detail(
                runtime.plan.details, "offload_cpu_requested_layers"
            ),
            offload_cpu_applied_layers=_optional_int_detail(
                runtime.plan.details, "offload_cpu_applied_layers"
            ),
            offload_cpu_applied_indices=offload_cpu_applied_indices,
            kv_cache_adaptation=kv_cache_adaptation,
            cache_dir_size_mb=cache_dir_size,
            cache_state=cache_state,
            allocator_gap_mb=allocator_gap_mb,
            allocator_gap_ratio=allocator_gap_ratio,
            native_runtime_profile=native_runtime_profile,
            resources=generation_resources,
            text_excerpt=_clip_text(response_text, max_chars=120),
        ),
        response_text=response_text,
    )


def build_prompt_request(
    *,
    runtime_config: RuntimeConfig,
    generation_config: GenerationConfig,
    messages: list[Message],
) -> PromptRequest:
    return PromptRequest(
        runtime_config=runtime_config,
        generation_config=generation_config,
        messages=messages,
    )


def build_scaling_prompt(*, target_prompt_tokens: int) -> str:
    repeated_words = " ".join("benchmark" for _ in range(max(1, target_prompt_tokens)))
    return (
        "Benchmark scaling probe input. "
        "Repeat and summarize this synthetic workload: "
        f"{repeated_words}"
    )


def _clear_backend_stats(runtime: LoadedRuntime) -> None:
    if isinstance(runtime.backend.stats, Stats):
        runtime.backend.stats.clear()


def _collect_native_runtime_profile(
    runtime: LoadedRuntime,
) -> NativeRuntimeProfile | None:
    if not isinstance(runtime.backend.stats, Stats):
        return None
    raw_summaries = runtime.backend.stats.collect_and_clear_ms()
    if not raw_summaries:
        return None
    storage_paths = tuple(
        sorted(
            {
                storage_path
                for event_name, storage_path in _STORAGE_PATH_BY_EVENT.items()
                if event_name in raw_summaries
            }
        )
    )
    event_summaries = {
        event_name: EventTimingSummary(
            count=int(summary["count"]),
            total_ms=float(summary["total_ms"]),
            min_ms=float(summary["min_ms"]),
            median_ms=float(summary["median_ms"]),
            p95_ms=float(summary["p95_ms"]),
            max_ms=float(summary["max_ms"]),
            mean_ms=float(summary["mean_ms"]),
        )
        for event_name, summary in raw_summaries.items()
    }
    return NativeRuntimeProfile(
        storage_paths=storage_paths,
        events=event_summaries,
    )


def _generate_outputs(
    executor: RuntimeExecutor,
    runtime: LoadedRuntime,
    request: PromptRequest,
    inputs: dict[str, object],
    generate_kwargs: dict[str, object],
    generation_config: object,
) -> tuple[object, float, dict[str, object], dict[str, object]]:
    generation_started = time.perf_counter()
    prepared_inputs, prepared_generate_kwargs = executor._prepare_generate_inputs(
        runtime,
        request,
        inputs,
        generate_kwargs,
    )
    try:
        with torch.inference_mode():
            with suppress_module_prints(runtime.backend.print_suppression_modules):
                with temporary_generation_config(runtime.model, generation_config):
                    return (
                        runtime.model.generate(
                            **prepared_inputs, **prepared_generate_kwargs
                        ),
                        generation_started,
                        prepared_inputs,
                        prepared_generate_kwargs,
                    )
    except TypeError as exc:
        if "streamer" not in str(exc):
            raise
        retry_kwargs = dict(prepared_generate_kwargs)
        retry_kwargs.pop("streamer", None)
        with torch.inference_mode():
            with suppress_module_prints(runtime.backend.print_suppression_modules):
                with temporary_generation_config(runtime.model, generation_config):
                    return (
                        runtime.model.generate(**prepared_inputs, **retry_kwargs),
                        generation_started,
                        prepared_inputs,
                        retry_kwargs,
                    )


def _count_prompt_tokens(inputs: dict[str, object]) -> int:
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise PromptExecutionError("Benchmark probe expected tensor-backed input_ids")
    return int(input_ids.shape[0] if input_ids.ndim == 1 else input_ids.shape[-1])


def _count_output_tokens(
    runtime: LoadedRuntime, inputs: dict[str, object], outputs: torch.Tensor
) -> int:
    input_ids = cast(torch.Tensor, inputs["input_ids"])
    if runtime.plan.generic_model_kind is GenericModelKind.SEQ2SEQ_LM:
        return int(outputs.shape[-1])
    return max(0, int(outputs.shape[-1] - input_ids.shape[-1]))


def _cache_mode(runtime: LoadedRuntime, request: PromptRequest) -> str:
    if not request.runtime_config.use_cache:
        return "none"
    if request.runtime_config.resolved_kv_cache_strategy() == "resident":
        return "resident-kv"
    if runtime.plan.supports_disk_cache:
        return "disk-kv"
    return "transformers-dynamic"


def _kv_cache_strategy(runtime: LoadedRuntime, request: PromptRequest) -> str | None:
    if not request.runtime_config.use_cache:
        return None
    resolved_strategy = request.runtime_config.resolved_kv_cache_strategy()
    if resolved_strategy == "resident":
        return resolved_strategy
    if not runtime.plan.supports_disk_cache:
        return None
    return resolved_strategy


def _inter_token_latencies(token_timestamps: tuple[float, ...]) -> tuple[float, ...]:
    if len(token_timestamps) < 2:
        return ()
    return tuple(
        round((right - left) * 1000.0, 6)
        for left, right in zip(token_timestamps, token_timestamps[1:])
    )


def _clip_text(text: str, *, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _extract_cache_state_snapshot(value: object) -> KVCacheStateSnapshot | None:
    snapshot_method = getattr(value, "cache_state_snapshot", None)
    if not callable(snapshot_method):
        return None
    snapshot = snapshot_method()
    if not isinstance(snapshot, KVCacheStateSnapshot):
        return None
    return snapshot


def _optional_int_detail(details: dict[str, str], key: str) -> int | None:
    raw_value = details.get(key)
    if raw_value is None or not raw_value:
        return None
    return int(raw_value)

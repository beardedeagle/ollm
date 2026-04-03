from dataclasses import dataclass
from typing import Protocol, cast

import torch
from transformers import GenerationConfig as TransformersGenerationConfig

from ollm.app.types import ContentKind, Message, PromptRequest, PromptResponse
from ollm.kv_cache.matrix import (
    build_kv_cache_adaptation_surface,
    resolve_kv_cache_base_dir,
    resolve_kv_cache_eviction_policy,
)
from ollm.kv_cache.state import KVCacheStateSnapshot
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.catalog import ModelModality
from ollm.runtime.errors import PromptExecutionError
from ollm.runtime.generation_config_support import (
    clear_sampling_fields,
    normalized_generation_config,
    temporary_generation_config,
)
from ollm.runtime.generation_support import (
    PLAN_METADATA_DETAIL_KEYS,
    extract_cache_state_snapshot,
    normalize_generate_inputs,
    prepare_text_inputs,
    render_plain_prompt,
    require_tensor,
)
from ollm.runtime.loaded_runtime import LoadedRuntime
from ollm.runtime.output_control import suppress_module_prints
from ollm.runtime.streaming import BufferedTextStreamer, NullStreamSink, StreamSink


class _StatsProtocol(Protocol):
    def print_and_clean(self) -> str: ...


DEFAULT_PREFILL_CHUNK_TOKENS = 512


def validate_runtime_request(runtime: LoadedRuntime, request: PromptRequest) -> None:
    if not request.messages:
        raise PromptExecutionError("At least one message is required")
    if runtime.backend.validate_request is not None:
        runtime.backend.validate_request(request)
        return

    contains_image = any(
        part.kind is ContentKind.IMAGE
        for message in request.messages
        for part in message.content
    )
    contains_audio = any(
        part.kind is ContentKind.AUDIO
        for message in request.messages
        for part in message.content
    )

    if contains_image and not runtime.capabilities.supports_modality(
        ModelModality.IMAGE
    ):
        raise PromptExecutionError(
            f"{runtime.config.model_reference} does not support image inputs"
        )
    if contains_audio and not runtime.capabilities.supports_modality(
        ModelModality.AUDIO
    ):
        raise PromptExecutionError(
            f"{runtime.config.model_reference} does not support audio inputs"
        )
    if (contains_image or contains_audio) and runtime.processor is None:
        if runtime.backend.allows_multimodal_without_processor:
            return
        raise PromptExecutionError(
            "Multimodal inputs require a processor-backed runtime. "
            "Enable --multimodal with a compatible model reference."
        )


def build_runtime_inputs(
    runtime: LoadedRuntime, messages: list[Message]
) -> dict[str, object]:
    transformers_messages = [
        message.as_transformers_message(
            structured_content=runtime.processor is not None
        )
        for message in messages
    ]
    if runtime.processor is not None:
        inputs = runtime.processor.apply_chat_template(
            transformers_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        contains_image = any(
            part.kind is ContentKind.IMAGE
            for message in messages
            for part in message.content
        )
        if contains_image:
            return inputs.to(runtime.device, dtype=torch.bfloat16)
        return inputs.to(runtime.device)

    if hasattr(runtime.tokenizer, "apply_chat_template"):
        try:
            inputs = runtime.tokenizer.apply_chat_template(
                transformers_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            return prepare_text_inputs(inputs, runtime.device)
        except (TypeError, ValueError, AttributeError):
            try:
                input_ids = runtime.tokenizer.apply_chat_template(
                    transformers_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=False,
                ).to(runtime.device)
                return prepare_text_inputs(input_ids, runtime.device)
            except (TypeError, ValueError, AttributeError):
                pass

    rendered_prompt = render_plain_prompt(messages)
    tokenized = runtime.tokenizer(rendered_prompt, return_tensors="pt")
    return {key: value.to(runtime.device) for key, value in tokenized.items()}


def build_runtime_generate_kwargs(
    runtime: LoadedRuntime,
    request: PromptRequest,
    streamer: BufferedTextStreamer | None,
) -> tuple[dict[str, object], TransformersGenerationConfig]:
    generation_config = normalized_generation_config(runtime, request)
    generate_kwargs: dict[str, object] = {}

    if request.runtime_config.use_cache:
        resolved_strategy = request.runtime_config.resolved_kv_cache_strategy()
        resolved_lifecycle = request.runtime_config.resolved_kv_cache_lifecycle()
        resolved_window_tokens = (
            request.runtime_config.resolved_kv_cache_window_tokens()
        )
        cache_base_dir = request.runtime_config.resolved_cache_dir()
        if resolved_strategy != "resident":
            cache_base_dir = resolve_kv_cache_base_dir(
                cache_dir=request.runtime_config.resolved_cache_dir(),
                lifecycle=resolved_lifecycle,
                model_reference=runtime.resolved_model.reference.raw,
                normalized_name=runtime.resolved_model.normalized_name,
                backend_id=runtime.plan.backend_id or "unknown",
                specialization_provider_id=runtime.plan.specialization_provider_id,
            )
        cache = runtime.get_or_create_kv_cache(
            cache_base_dir,
            resolved_strategy,
            resolved_lifecycle,
            resolved_window_tokens,
        )
        if cache is not None:
            generate_kwargs["past_key_values"] = cache

    if streamer is not None:
        generate_kwargs["streamer"] = streamer

    if runtime.processor is not None and any(
        part.kind is ContentKind.AUDIO
        for message in request.messages
        for part in message.content
    ):
        generation_config.do_sample = False
        clear_sampling_fields(generation_config)

    return generate_kwargs, generation_config


def prepare_runtime_generate_inputs(
    runtime: LoadedRuntime,
    request: PromptRequest,
    inputs: dict[str, object],
    generate_kwargs: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    input_ids_value = inputs.get("input_ids")
    if not isinstance(input_ids_value, torch.Tensor):
        return inputs, generate_kwargs
    if input_ids_value.ndim != 2 or input_ids_value.shape[0] != 1:
        return inputs, generate_kwargs
    if runtime.processor is not None:
        return inputs, generate_kwargs
    if runtime.plan.backend_id != "optimized-native":
        return inputs, generate_kwargs
    runtime_kind = (
        runtime.plan.generic_model_kind or runtime.resolved_model.generic_model_kind
    )
    if runtime_kind is not GenericModelKind.CAUSAL_LM:
        return inputs, generate_kwargs
    prefill_token_count = input_ids_value.shape[1] - 1
    if prefill_token_count <= DEFAULT_PREFILL_CHUNK_TOKENS:
        return inputs, generate_kwargs
    return _run_chunked_prefill(runtime, inputs, generate_kwargs)


def _run_chunked_prefill(
    runtime: LoadedRuntime,
    inputs: dict[str, object],
    generate_kwargs: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    forward_method = getattr(runtime.model, "forward", None)
    if not callable(forward_method):
        return inputs, generate_kwargs
    input_ids = require_tensor(inputs["input_ids"])
    attention_mask_value = inputs.get("attention_mask")
    attention_mask = (
        None if attention_mask_value is None else require_tensor(attention_mask_value)
    )
    prefill_cache = generate_kwargs.get("past_key_values")
    prefill_end = input_ids.shape[1] - 1
    with torch.inference_mode():
        with suppress_module_prints(runtime.backend.print_suppression_modules):
            for chunk_start in range(0, prefill_end, DEFAULT_PREFILL_CHUNK_TOKENS):
                chunk_end = min(prefill_end, chunk_start + DEFAULT_PREFILL_CHUNK_TOKENS)
                forward_inputs: dict[str, object] = {
                    "input_ids": input_ids[:, chunk_start:chunk_end],
                    "use_cache": True,
                    "cache_position": torch.arange(
                        chunk_start,
                        chunk_end,
                        device=input_ids.device,
                        dtype=torch.long,
                    ),
                }
                if attention_mask is not None:
                    forward_inputs["attention_mask"] = attention_mask[:, :chunk_end]
                if prefill_cache is not None:
                    forward_inputs["past_key_values"] = prefill_cache
                outputs = forward_method(**forward_inputs)
                prefill_cache = getattr(outputs, "past_key_values", None)
                if prefill_cache is None:
                    raise PromptExecutionError(
                        "Chunked prefill requires a causal runtime that returns past_key_values."
                    )
    updated_inputs = dict(inputs)
    updated_inputs["input_ids"] = input_ids[:, -1:]
    if attention_mask is not None:
        updated_inputs["attention_mask"] = attention_mask
    updated_generate_kwargs = dict(generate_kwargs)
    updated_generate_kwargs["past_key_values"] = prefill_cache
    return updated_inputs, updated_generate_kwargs


def decode_runtime_response(
    runtime: LoadedRuntime, inputs: dict[str, object], outputs: torch.Tensor
) -> str:
    if runtime.processor is not None:
        input_ids = require_tensor(inputs["input_ids"])
        decoded = runtime.processor.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        if not decoded:
            return ""
        return decoded[0]

    if runtime.plan.generic_model_kind is GenericModelKind.SEQ2SEQ_LM:
        return runtime.tokenizer.decode(outputs[0], skip_special_tokens=True)

    input_ids = require_tensor(inputs["input_ids"])
    return runtime.tokenizer.decode(
        outputs[0][input_ids.shape[-1] :], skip_special_tokens=True
    )


@dataclass(slots=True)
class RuntimeExecutor:
    def execute(
        self,
        runtime: LoadedRuntime,
        request: PromptRequest,
        sink: StreamSink | None = None,
    ) -> PromptResponse:
        stream_sink = sink or NullStreamSink()
        validate_runtime_request(runtime, request)
        stream_sink.on_status(self._status_message(runtime))

        if runtime.backend.execute_prompt is not None:
            response = runtime.backend.execute_prompt(request, stream_sink)
            return self._finalize_response(runtime, response)

        if request.generation_config.seed is not None:
            torch.manual_seed(request.generation_config.seed)

        inputs = build_runtime_inputs(runtime, request.messages)
        streamer = None
        if request.generation_config.stream:
            streamer = BufferedTextStreamer(
                runtime.tokenizer,
                stream_sink,
                skip_prompt=True,
                skip_special_tokens=True,
            )

        generate_kwargs, generation_config = build_runtime_generate_kwargs(
            runtime, request, streamer
        )
        filtered_inputs = normalize_generate_inputs(inputs)
        filtered_inputs, generate_kwargs = prepare_runtime_generate_inputs(
            runtime,
            request,
            filtered_inputs,
            generate_kwargs,
        )

        with torch.inference_mode():
            with suppress_module_prints(runtime.backend.print_suppression_modules):
                with temporary_generation_config(runtime.model, generation_config):
                    outputs = runtime.model.generate(
                        **filtered_inputs, **generate_kwargs
                    )

        if hasattr(outputs, "detach"):
            outputs = outputs.detach()
        outputs = outputs.cpu()
        response_text = decode_runtime_response(runtime, filtered_inputs, outputs)
        cache_state = extract_cache_state_snapshot(
            generate_kwargs.get("past_key_values")
        )
        if streamer is not None and not response_text.strip():
            response_text = streamer.text
        assistant_message = Message.assistant_text(response_text)
        metadata = self._plan_metadata(runtime, cache_state)
        return PromptResponse(
            text=response_text, assistant_message=assistant_message, metadata=metadata
        )

    def _finalize_response(
        self, runtime: LoadedRuntime, response: PromptResponse
    ) -> PromptResponse:
        metadata = dict(response.metadata)
        metadata.update(self._plan_metadata(runtime, None))
        return PromptResponse(
            text=response.text,
            assistant_message=response.assistant_message,
            metadata=metadata,
        )

    def _plan_metadata(
        self,
        runtime: LoadedRuntime,
        cache_state: KVCacheStateSnapshot | None,
    ) -> dict[str, str]:
        metadata = {
            "backend_id": runtime.plan.backend_id or "unknown",
            "specialization_state": runtime.plan.specialization_state.value,
            "specialization_applied": str(runtime.plan.specialization_applied).lower(),
            "specialization_provider_id": runtime.plan.specialization_provider_id or "",
            "specialization_pass_ids": ",".join(
                pass_id.value for pass_id in runtime.plan.specialization_pass_ids
            ),
            "applied_specialization_pass_ids": ",".join(
                pass_id.value
                for pass_id in runtime.plan.applied_specialization_pass_ids
            ),
            "fallback_reason": runtime.plan.fallback_reason or "",
            "kv_cache_strategy": (
                runtime.config.resolved_kv_cache_strategy()
                if runtime.config.use_cache
                and (
                    runtime.plan.supports_disk_cache
                    or runtime.config.resolved_kv_cache_strategy() == "resident"
                )
                else "none"
            ),
            "kv_cache_lifecycle": runtime.config.resolved_kv_cache_lifecycle(),
            "kv_cache_adaptation_mode": runtime.config.resolved_kv_cache_adaptation_mode(),
        }
        resolved_window_tokens = runtime.config.resolved_kv_cache_window_tokens()
        if resolved_window_tokens is not None:
            metadata["kv_cache_window_tokens"] = str(resolved_window_tokens)
        for detail_key in PLAN_METADATA_DETAIL_KEYS:
            detail_value = runtime.plan.details.get(detail_key)
            if detail_value is not None:
                metadata[detail_key] = detail_value
        if cache_state is not None:
            metadata.update(
                {
                    "kv_cache_policy_id": cache_state.policy_id,
                    "kv_cache_persistence_format": cache_state.persistence_format,
                    "kv_cache_residency_mode": cache_state.residency_mode,
                    "kv_cache_window_policy": cache_state.window_policy,
                    "kv_cache_cold_tier_encoding": cache_state.cold_tier_encoding,
                    "kv_cache_persisted_tokens": str(cache_state.persisted_tokens),
                    "kv_cache_persisted_artifacts": str(
                        cache_state.persisted_artifact_count
                    ),
                    "kv_cache_resident_layers": str(cache_state.resident_layer_count),
                    "kv_cache_resident_tokens": str(cache_state.resident_tokens),
                    "kv_cache_resident_bytes": str(cache_state.resident_bytes),
                    "kv_cache_hot_tokens": str(cache_state.hot_tokens),
                    "kv_cache_hot_layers": str(cache_state.hot_layer_count),
                    "kv_cache_compaction_count": str(cache_state.compaction_count),
                    "kv_cache_spill_count": str(cache_state.spill_count),
                    "kv_cache_spilled_tokens": str(cache_state.spilled_tokens),
                    "kv_cache_eviction_count": str(cache_state.eviction_count),
                    "kv_cache_evicted_tokens": str(cache_state.evicted_tokens),
                }
            )
            if cache_state.eviction_policy is not None:
                metadata["kv_cache_eviction_policy"] = cache_state.eviction_policy
            if cache_state.window_max_tokens is not None:
                metadata["kv_cache_window_max_tokens"] = str(
                    cache_state.window_max_tokens
                )
            if cache_state.cold_store_format is not None:
                metadata["kv_cache_cold_store_format"] = cache_state.cold_store_format
            if cache_state.cold_tier_representation is not None:
                metadata["kv_cache_cold_tier_representation"] = (
                    cache_state.cold_tier_representation
                )
        adaptation_surface = build_kv_cache_adaptation_surface(
            adaptation_mode=runtime.config.resolved_kv_cache_adaptation_mode(),
            current_strategy=runtime.config.resolved_kv_cache_strategy(),
            persisted_artifact_count=(
                None if cache_state is None else cache_state.persisted_artifact_count
            ),
            spill_count=None if cache_state is None else cache_state.spill_count,
            resident_bytes=None if cache_state is None else cache_state.resident_bytes,
            hot_bytes=None if cache_state is None else cache_state.hot_bytes,
        )
        metadata["kv_cache_adaptation_recommendation_available"] = str(
            adaptation_surface.recommendation_available
        ).lower()
        if adaptation_surface.recommended_strategy_id is not None:
            metadata["kv_cache_adaptation_recommended_strategy"] = (
                adaptation_surface.recommended_strategy_id
            )
        metadata["kv_cache_adaptation_reason"] = adaptation_surface.reason
        if (
            "kv_cache_eviction_policy" not in metadata
            and runtime.config.use_cache
            and (
                runtime.plan.supports_disk_cache
                or runtime.config.resolved_kv_cache_strategy() == "resident"
            )
        ):
            resolved_eviction_policy = resolve_kv_cache_eviction_policy(
                runtime.config.resolved_kv_cache_strategy()
            )
            if resolved_eviction_policy is not None:
                metadata["kv_cache_eviction_policy"] = resolved_eviction_policy
        stats = cast(_StatsProtocol | None, runtime.backend.stats)
        if stats is not None:
            metadata["stats"] = stats.print_and_clean()
        return metadata

    def _status_message(self, runtime: LoadedRuntime) -> str:
        return f"Running {runtime.config.model_reference} on {runtime.config.device} via {runtime.plan.backend_id}"

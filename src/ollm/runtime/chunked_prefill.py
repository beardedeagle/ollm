"""Chunked-prefill strategy resolution and execution."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from inspect import Parameter, signature
from typing import Self

import torch

from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.errors import PromptExecutionError
from ollm.runtime.generation_support import require_tensor
from ollm.runtime.loaded_runtime import LoadedRuntime
from ollm.runtime.output_control import suppress_module_prints


class ChunkedPrefillStrategyId(StrEnum):
    OPTIMIZED_NATIVE_TEXT = "optimized-native-text"
    OPTIMIZED_NATIVE_MULTIMODAL = "optimized-native-multimodal"
    TRANSFORMERS_GENERIC_TEXT = "transformers-generic-text"
    TRANSFORMERS_GENERIC_MULTIMODAL = "transformers-generic-multimodal"


class ChunkedPrefillGapId(StrEnum):
    PROMPT_TOKENIZATION_BEFORE_PREFILL = "prompt-tokenization-before-prefill"
    FULL_ATTENTION_MASK_BEFORE_PREFILL = "full-attention-mask-before-prefill"
    SEQ2SEQ_SOURCE_PREFILL = "seq2seq-source-prefill"


class ChunkedPrefillRecommendation(StrEnum):
    IMPLEMENT = "implement"
    DEFER = "defer"
    REJECT = "reject"


class ChunkedPrefillExecutionBoundary(StrEnum):
    POST_TOKENIZATION = "post-tokenization"


class ChunkedPrefillAttentionMaskMode(StrEnum):
    FULL_PREFIX_MATERIALIZED = "full-prefix-materialized"


@dataclass(frozen=True, slots=True)
class ChunkedPrefillGapDecision:
    gap_id: ChunkedPrefillGapId
    current_behavior: str
    recommendation: ChunkedPrefillRecommendation
    rationale: str

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        return {key: str(value) for key, value in payload.items()}


@dataclass(frozen=True, slots=True)
class ChunkedPrefillScopeSurface:
    strategy_id: ChunkedPrefillStrategyId | None
    runtime_eligible: bool
    applied: bool
    activation_reason: str
    execution_boundary: ChunkedPrefillExecutionBoundary
    attention_mask_mode: ChunkedPrefillAttentionMaskMode
    gap_inventory: tuple[ChunkedPrefillGapDecision, ...]

    def with_activation(self, *, applied: bool, activation_reason: str) -> Self:
        return replace(
            self,
            applied=applied,
            activation_reason=activation_reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_id": None if self.strategy_id is None else self.strategy_id.value,
            "runtime_eligible": self.runtime_eligible,
            "applied": self.applied,
            "activation_reason": self.activation_reason,
            "execution_boundary": self.execution_boundary.value,
            "attention_mask_mode": self.attention_mask_mode.value,
            "gap_inventory": [decision.to_dict() for decision in self.gap_inventory],
        }


@dataclass(frozen=True, slots=True)
class PreparedChunkedPrefill:
    inputs: dict[str, object]
    generate_kwargs: dict[str, object]
    scope: ChunkedPrefillScopeSurface


@dataclass(frozen=True, slots=True)
class ChunkedPrefillStrategy:
    strategy_id: ChunkedPrefillStrategyId
    matches: Callable[[LoadedRuntime, GenericModelKind | None], bool]
    prepare: Callable[
        [LoadedRuntime, dict[str, object], dict[str, object], int],
        tuple[dict[str, object], dict[str, object]],
    ]


_CHUNKED_PREFILL_GAP_INVENTORY = (
    ChunkedPrefillGapDecision(
        gap_id=ChunkedPrefillGapId.PROMPT_TOKENIZATION_BEFORE_PREFILL,
        current_behavior="Prompt tokenization completes before chunked prefill begins.",
        recommendation=ChunkedPrefillRecommendation.DEFER,
        rationale=(
            "Streaming prompt tokenization needs tokenizer-specific boundary "
            "preservation instead of the current whole-prompt tokenization path."
        ),
    ),
    ChunkedPrefillGapDecision(
        gap_id=ChunkedPrefillGapId.FULL_ATTENTION_MASK_BEFORE_PREFILL,
        current_behavior=(
            "A full prefix attention mask is materialized before chunked prefill "
            "hands off to the final generate step."
        ),
        recommendation=ChunkedPrefillRecommendation.DEFER,
        rationale=(
            "The current generation handoff still relies on full-prefix masks. "
            "A lazy mask contract needs backend proof before it can replace that "
            "shape safely."
        ),
    ),
    ChunkedPrefillGapDecision(
        gap_id=ChunkedPrefillGapId.SEQ2SEQ_SOURCE_PREFILL,
        current_behavior=(
            "Seq2seq source prompts do not use causal-cache chunked prefill."
        ),
        recommendation=ChunkedPrefillRecommendation.DEFER,
        rationale=(
            "Encoder-decoder source ingestion has no equivalent causal-cache "
            "prefill contract; it needs a separate encoder strategy."
        ),
    ),
)


def prepare_chunked_prefill(
    *,
    runtime: LoadedRuntime,
    inputs: dict[str, object],
    generate_kwargs: dict[str, object],
    chunk_tokens: int,
) -> PreparedChunkedPrefill:
    scope = build_chunked_prefill_scope_surface(
        runtime=runtime,
        inputs=inputs,
        chunk_tokens=chunk_tokens,
    )
    input_ids_value = inputs.get("input_ids")
    if not isinstance(input_ids_value, torch.Tensor):
        return PreparedChunkedPrefill(inputs, generate_kwargs, scope)
    if input_ids_value.ndim != 2 or input_ids_value.shape[0] != 1:
        return PreparedChunkedPrefill(inputs, generate_kwargs, scope)
    prefill_token_count = input_ids_value.shape[1] - 1
    if prefill_token_count <= chunk_tokens or not scope.runtime_eligible:
        return PreparedChunkedPrefill(inputs, generate_kwargs, scope)
    strategy = _require_strategy(scope.strategy_id)
    prepared_inputs, prepared_generate_kwargs = strategy.prepare(
        runtime,
        inputs,
        generate_kwargs,
        chunk_tokens,
    )
    return PreparedChunkedPrefill(
        inputs=prepared_inputs,
        generate_kwargs=prepared_generate_kwargs,
        scope=scope.with_activation(
            applied=True,
            activation_reason="Bounded chunked prefill ran before final decode.",
        ),
    )


def build_chunked_prefill_scope_surface(
    *,
    runtime: LoadedRuntime,
    inputs: dict[str, object],
    chunk_tokens: int,
) -> ChunkedPrefillScopeSurface:
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return _scope(
            strategy_id=None,
            runtime_eligible=False,
            activation_reason="Chunked prefill requires tensor-backed input_ids.",
        )
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        return _scope(
            strategy_id=None,
            runtime_eligible=False,
            activation_reason="Chunked prefill requires a single batch row.",
        )
    runtime_kind = (
        runtime.plan.generic_model_kind or runtime.resolved_model.generic_model_kind
    )
    if runtime_kind is GenericModelKind.SEQ2SEQ_LM:
        return _scope(
            strategy_id=None,
            runtime_eligible=False,
            activation_reason="Seq2seq source prompts cannot use causal-cache chunked prefill.",
        )
    strategy = _resolve_strategy(runtime, runtime_kind)
    if strategy is None:
        return _scope(
            strategy_id=None,
            runtime_eligible=False,
            activation_reason=(
                "Chunked prefill requires a supported causal runtime strategy."
            ),
        )
    prefill_token_count = int(input_ids.shape[1] - 1)
    if prefill_token_count <= chunk_tokens:
        return _scope(
            strategy_id=strategy.strategy_id,
            runtime_eligible=True,
            activation_reason=(
                "Prompt length does not exceed the chunked-prefill threshold."
            ),
        )
    return _scope(
        strategy_id=strategy.strategy_id,
        runtime_eligible=True,
        activation_reason="Runtime is eligible for bounded chunked prefill.",
    )


def chunked_prefill_gap_inventory() -> tuple[ChunkedPrefillGapDecision, ...]:
    return _CHUNKED_PREFILL_GAP_INVENTORY


def _resolve_strategy(
    runtime: LoadedRuntime,
    runtime_kind: GenericModelKind | None,
) -> ChunkedPrefillStrategy | None:
    for strategy in _CHUNKED_PREFILL_STRATEGIES:
        if strategy.matches(runtime, runtime_kind):
            return strategy
    return None


def _require_strategy(
    strategy_id: ChunkedPrefillStrategyId | None,
) -> ChunkedPrefillStrategy:
    if strategy_id is None:
        raise PromptExecutionError(
            "Chunked prefill strategy resolution was required but no strategy was selected."
        )
    for strategy in _CHUNKED_PREFILL_STRATEGIES:
        if strategy.strategy_id is strategy_id:
            return strategy
    raise PromptExecutionError(
        f"Unsupported chunked prefill strategy {strategy_id.value!r}."
    )


def _prepare_optimized_native_text_prefill(
    runtime: LoadedRuntime,
    inputs: dict[str, object],
    generate_kwargs: dict[str, object],
    chunk_tokens: int,
) -> tuple[dict[str, object], dict[str, object]]:
    return _run_causal_chunked_prefill(
        runtime=runtime,
        inputs=inputs,
        generate_kwargs=generate_kwargs,
        chunk_tokens=chunk_tokens,
        strategy_id=ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_TEXT,
    )


def _prepare_optimized_native_multimodal_prefill(
    runtime: LoadedRuntime,
    inputs: dict[str, object],
    generate_kwargs: dict[str, object],
    chunk_tokens: int,
) -> tuple[dict[str, object], dict[str, object]]:
    return _run_causal_chunked_prefill(
        runtime=runtime,
        inputs=inputs,
        generate_kwargs=generate_kwargs,
        chunk_tokens=chunk_tokens,
        strategy_id=ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_MULTIMODAL,
    )


def _prepare_transformers_generic_text_prefill(
    runtime: LoadedRuntime,
    inputs: dict[str, object],
    generate_kwargs: dict[str, object],
    chunk_tokens: int,
) -> tuple[dict[str, object], dict[str, object]]:
    return _run_causal_chunked_prefill(
        runtime=runtime,
        inputs=inputs,
        generate_kwargs=generate_kwargs,
        chunk_tokens=chunk_tokens,
        strategy_id=ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_TEXT,
    )


def _prepare_transformers_generic_multimodal_prefill(
    runtime: LoadedRuntime,
    inputs: dict[str, object],
    generate_kwargs: dict[str, object],
    chunk_tokens: int,
) -> tuple[dict[str, object], dict[str, object]]:
    return _run_causal_chunked_prefill(
        runtime=runtime,
        inputs=inputs,
        generate_kwargs=generate_kwargs,
        chunk_tokens=chunk_tokens,
        strategy_id=ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_MULTIMODAL,
    )


_CHUNKED_PREFILL_STRATEGIES = (
    ChunkedPrefillStrategy(
        strategy_id=ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_TEXT,
        matches=lambda runtime, runtime_kind: (
            runtime.plan.backend_id == "optimized-native"
            and runtime.processor is None
            and runtime_kind is GenericModelKind.CAUSAL_LM
        ),
        prepare=_prepare_optimized_native_text_prefill,
    ),
    ChunkedPrefillStrategy(
        strategy_id=ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_MULTIMODAL,
        matches=lambda runtime, runtime_kind: (
            runtime.plan.backend_id == "optimized-native"
            and runtime.processor is not None
            and runtime_kind is not GenericModelKind.SEQ2SEQ_LM
        ),
        prepare=_prepare_optimized_native_multimodal_prefill,
    ),
    ChunkedPrefillStrategy(
        strategy_id=ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_TEXT,
        matches=lambda runtime, runtime_kind: (
            runtime.plan.backend_id == "transformers-generic"
            and runtime.processor is None
            and runtime_kind is GenericModelKind.CAUSAL_LM
        ),
        prepare=_prepare_transformers_generic_text_prefill,
    ),
    ChunkedPrefillStrategy(
        strategy_id=ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_MULTIMODAL,
        matches=lambda runtime, runtime_kind: (
            runtime.plan.backend_id == "transformers-generic"
            and runtime.processor is not None
            and runtime_kind is not GenericModelKind.SEQ2SEQ_LM
        ),
        prepare=_prepare_transformers_generic_multimodal_prefill,
    ),
)


def _run_causal_chunked_prefill(
    *,
    runtime: LoadedRuntime,
    inputs: dict[str, object],
    generate_kwargs: dict[str, object],
    chunk_tokens: int,
    strategy_id: ChunkedPrefillStrategyId | None,
) -> tuple[dict[str, object], dict[str, object]]:
    forward_method = getattr(runtime.model, "forward", None)
    if not callable(forward_method):
        return inputs, generate_kwargs
    input_ids = require_tensor(inputs["input_ids"])
    attention_mask = _optional_tensor(inputs.get("attention_mask"))
    sequence_inputs = _collect_sequence_inputs(inputs, input_ids)
    static_inputs = _collect_static_inputs(inputs)
    prefill_cache = generate_kwargs.get("past_key_values")
    prefill_end = input_ids.shape[1] - 1
    with torch.inference_mode():
        with suppress_module_prints(runtime.backend.print_suppression_modules):
            for chunk_start in range(0, prefill_end, chunk_tokens):
                chunk_end = min(prefill_end, chunk_start + chunk_tokens)
                forward_inputs: dict[str, object] = dict(static_inputs)
                forward_inputs["input_ids"] = input_ids[:, chunk_start:chunk_end]
                if attention_mask is not None:
                    forward_inputs["attention_mask"] = attention_mask[:, :chunk_end]
                for key, value in sequence_inputs.items():
                    forward_inputs[key] = value[:, chunk_start:chunk_end]
                if prefill_cache is not None:
                    forward_inputs["past_key_values"] = prefill_cache
                forward_inputs["use_cache"] = True
                forward_inputs["cache_position"] = torch.arange(
                    chunk_start,
                    chunk_end,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                filtered_inputs = _filter_supported_forward_inputs(
                    forward_method,
                    forward_inputs,
                )
                outputs = forward_method(**filtered_inputs)
                prefill_cache = getattr(outputs, "past_key_values", None)
                if prefill_cache is None:
                    strategy_label = (
                        "unknown" if strategy_id is None else strategy_id.value
                    )
                    raise PromptExecutionError(
                        "Chunked prefill strategy "
                        f"{strategy_label!r} requires a runtime that returns "
                        "past_key_values."
                    )
    updated_inputs = dict(static_inputs)
    updated_inputs["input_ids"] = input_ids[:, -1:]
    if attention_mask is not None:
        updated_inputs["attention_mask"] = attention_mask
    for key, value in sequence_inputs.items():
        updated_inputs[key] = value[:, -1:]
    updated_generate_kwargs = dict(generate_kwargs)
    updated_generate_kwargs["past_key_values"] = prefill_cache
    return updated_inputs, updated_generate_kwargs


def _collect_sequence_inputs(
    inputs: dict[str, object],
    input_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sequence_inputs: dict[str, torch.Tensor] = {}
    sequence_length = input_ids.shape[1]
    for key, value in inputs.items():
        if key in {"input_ids", "attention_mask"}:
            continue
        if (
            isinstance(value, torch.Tensor)
            and value.ndim == 2
            and value.shape[1] == sequence_length
        ):
            sequence_inputs[key] = value
    return sequence_inputs


def _collect_static_inputs(inputs: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask"}
    }


def _filter_supported_forward_inputs(
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


def _optional_tensor(value: object) -> torch.Tensor | None:
    if value is None:
        return None
    return require_tensor(value)


def _scope(
    *,
    strategy_id: ChunkedPrefillStrategyId | None,
    runtime_eligible: bool,
    activation_reason: str,
) -> ChunkedPrefillScopeSurface:
    return ChunkedPrefillScopeSurface(
        strategy_id=strategy_id,
        runtime_eligible=runtime_eligible,
        applied=False,
        activation_reason=activation_reason,
        execution_boundary=ChunkedPrefillExecutionBoundary.POST_TOKENIZATION,
        attention_mask_mode=ChunkedPrefillAttentionMaskMode.FULL_PREFIX_MATERIALIZED,
        gap_inventory=_CHUNKED_PREFILL_GAP_INVENTORY,
    )

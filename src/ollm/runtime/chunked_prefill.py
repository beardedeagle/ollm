"""Chunked prompt-ingestion strategies for runtime generation."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Self

from ollm.app.types import Message
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.chunked_prefill_support import (
    build_forward_input_filter,
    ones_attention_mask,
    prepare_static_inputs,
    prompt_token_id_pieces,
    render_prompt_text,
    resolve_stream_tokenizer,
    run_causal_prefill_chunk,
    token_tensor,
)
from ollm.runtime.chunked_prefill_support import (
    prompt_token_count as count_prompt_tokens,
)
from ollm.runtime.errors import PromptExecutionError
from ollm.runtime.loaded_runtime import LoadedRuntime


class ChunkedPrefillStrategyId(StrEnum):
    OPTIMIZED_NATIVE_TEXT = "optimized-native-text"
    OPTIMIZED_NATIVE_MULTIMODAL = "optimized-native-multimodal"
    TRANSFORMERS_GENERIC_TEXT = "transformers-generic-text"
    TRANSFORMERS_GENERIC_MULTIMODAL = "transformers-generic-multimodal"
    TRANSFORMERS_GENERIC_SEQ2SEQ_SOURCE = "transformers-generic-seq2seq-source"


class ChunkedPrefillGapId(StrEnum):
    PROMPT_TOKENIZATION_BEFORE_PREFILL = "prompt-tokenization-before-prefill"
    FULL_ATTENTION_MASK_BEFORE_PREFILL = "full-attention-mask-before-prefill"
    SEQ2SEQ_SOURCE_PREFILL = "seq2seq-source-prefill"


class ChunkedPrefillRecommendation(StrEnum):
    IMPLEMENT = "implement"
    DEFER = "defer"
    REJECT = "reject"


class ChunkedPrefillExecutionBoundary(StrEnum):
    STREAMED_PROMPT_PREPARATION = "streamed-prompt-preparation"


class ChunkedPrefillAttentionMaskMode(StrEnum):
    LAZY_PREFIX_SYNTHESIS = "lazy-prefix-synthesis"


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
    prompt_token_count: int


@dataclass(frozen=True, slots=True)
class ChunkedPrefillStrategy:
    strategy_id: ChunkedPrefillStrategyId
    matches: Callable[[LoadedRuntime, GenericModelKind | None], bool]
    prepare: Callable[
        [LoadedRuntime, list[Message], dict[str, object], int],
        PreparedChunkedPrefill,
    ]


_CHUNKED_PREFILL_GAP_INVENTORY = (
    ChunkedPrefillGapDecision(
        gap_id=ChunkedPrefillGapId.PROMPT_TOKENIZATION_BEFORE_PREFILL,
        current_behavior=(
            "Supported strategies render the prompt template once, then tokenize "
            "prompt pieces incrementally during strategy execution."
        ),
        recommendation=ChunkedPrefillRecommendation.IMPLEMENT,
        rationale=(
            "Prompt tokenization no longer has to complete as one full prompt-wide "
            "step before the chunked strategy begins."
        ),
    ),
    ChunkedPrefillGapDecision(
        gap_id=ChunkedPrefillGapId.FULL_ATTENTION_MASK_BEFORE_PREFILL,
        current_behavior=(
            "Causal chunked strategies synthesize prefix attention masks per "
            "chunk and materialize the full mask only at the final generate "
            "handoff when the runtime still requires it."
        ),
        recommendation=ChunkedPrefillRecommendation.IMPLEMENT,
        rationale=(
            "Full prompt attention masks are no longer built before chunked "
            "prompt ingestion begins."
        ),
    ),
    ChunkedPrefillGapDecision(
        gap_id=ChunkedPrefillGapId.SEQ2SEQ_SOURCE_PREFILL,
        current_behavior=(
            "Seq2seq source prompts now use a dedicated streamed source-ingestion "
            "strategy instead of pretending they share the causal-cache prefill "
            "contract."
        ),
        recommendation=ChunkedPrefillRecommendation.IMPLEMENT,
        rationale=(
            "Seq2seq now has its own explicit strategy lane rather than being left "
            "unsupported."
        ),
    ),
)


def prepare_chunked_prefill(
    *,
    runtime: LoadedRuntime,
    messages: list[Message],
    generate_kwargs: dict[str, object],
    chunk_tokens: int,
    eager_input_builder: Callable[[LoadedRuntime, list[Message]], dict[str, object]],
) -> PreparedChunkedPrefill:
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be at least 1")
    runtime_kind = (
        runtime.plan.generic_model_kind or runtime.resolved_model.generic_model_kind
    )
    strategy = _resolve_strategy(runtime, runtime_kind)
    if strategy is None:
        inputs = eager_input_builder(runtime, messages)
        return PreparedChunkedPrefill(
            inputs=inputs,
            generate_kwargs=generate_kwargs,
            scope=_scope(
                strategy_id=None,
                runtime_eligible=False,
                activation_reason=(
                    "No chunked prompt-ingestion strategy matched the active runtime."
                ),
            ),
            prompt_token_count=count_prompt_tokens(inputs),
        )
    if (
        strategy.strategy_id
        is not ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_SEQ2SEQ_SOURCE
        and not callable(getattr(runtime.model, "forward", None))
    ):
        inputs = eager_input_builder(runtime, messages)
        return PreparedChunkedPrefill(
            inputs=inputs,
            generate_kwargs=generate_kwargs,
            scope=_scope(
                strategy_id=strategy.strategy_id,
                runtime_eligible=False,
                activation_reason=(
                    "Chunked prompt-ingestion strategy requires a callable forward method."
                ),
            ),
            prompt_token_count=count_prompt_tokens(inputs),
        )
    return strategy.prepare(runtime, messages, generate_kwargs, chunk_tokens)


def chunked_prefill_gap_inventory() -> tuple[ChunkedPrefillGapDecision, ...]:
    return _CHUNKED_PREFILL_GAP_INVENTORY


def _prepare_streamed_causal_strategy(
    runtime: LoadedRuntime,
    messages: list[Message],
    generate_kwargs: dict[str, object],
    chunk_tokens: int,
    *,
    strategy_id: ChunkedPrefillStrategyId,
) -> PreparedChunkedPrefill:
    rendered_prompt = render_prompt_text(runtime, messages)
    static_inputs = prepare_static_inputs(runtime, messages)
    total_prompt_token_count = 0
    deferred_tokens: list[int] = []
    prefilled_token_count = 0
    prefill_cache = generate_kwargs.get("past_key_values")
    forward_method = getattr(runtime.model, "forward", None)
    forward_input_filter = (
        None
        if not callable(forward_method)
        else build_forward_input_filter(forward_method)
    )

    for token_piece in prompt_token_id_pieces(
        resolve_stream_tokenizer(runtime),
        rendered_prompt,
    ):
        total_prompt_token_count += len(token_piece)
        deferred_tokens.extend(token_piece)
        while len(deferred_tokens) > chunk_tokens + 1:
            if not callable(forward_method):
                raise PromptExecutionError(
                    f"Chunked prompt-ingestion strategy {strategy_id.value!r} requires a callable forward method."
                )
            assert forward_input_filter is not None
            prefill_cache = run_causal_prefill_chunk(
                runtime=runtime,
                forward_method=forward_method,
                forward_input_filter=forward_input_filter,
                static_inputs=static_inputs,
                chunk_ids=deferred_tokens[:chunk_tokens],
                prefill_cache=prefill_cache,
                prefix_token_count=prefilled_token_count,
                strategy_label=strategy_id.value,
            )
            del deferred_tokens[:chunk_tokens]
            prefilled_token_count += chunk_tokens

    if total_prompt_token_count - 1 > chunk_tokens:
        while len(deferred_tokens) > 1:
            if not callable(forward_method):
                raise PromptExecutionError(
                    f"Chunked prompt-ingestion strategy {strategy_id.value!r} requires a callable forward method."
                )
            chunk_size = min(chunk_tokens, len(deferred_tokens) - 1)
            assert forward_input_filter is not None
            prefill_cache = run_causal_prefill_chunk(
                runtime=runtime,
                forward_method=forward_method,
                forward_input_filter=forward_input_filter,
                static_inputs=static_inputs,
                chunk_ids=deferred_tokens[:chunk_size],
                prefill_cache=prefill_cache,
                prefix_token_count=prefilled_token_count,
                strategy_label=strategy_id.value,
            )
            del deferred_tokens[:chunk_size]
            prefilled_token_count += chunk_size

    if total_prompt_token_count == 0:
        raise PromptExecutionError(
            "Chunked prompt ingestion produced no prompt tokens."
        )

    final_inputs = dict(static_inputs)
    final_generate_kwargs = dict(generate_kwargs)
    if prefilled_token_count > 0:
        final_inputs["input_ids"] = token_tensor(deferred_tokens, device=runtime.device)
        final_inputs["attention_mask"] = ones_attention_mask(
            token_count=total_prompt_token_count,
            device=runtime.device,
        )
        final_generate_kwargs["past_key_values"] = prefill_cache
        scope = _scope(
            strategy_id=strategy_id,
            runtime_eligible=True,
            activation_reason="Bounded chunked prefill ran before final decode.",
        ).with_activation(
            applied=True,
            activation_reason="Bounded chunked prefill ran before final decode.",
        )
    else:
        final_inputs["input_ids"] = token_tensor(deferred_tokens, device=runtime.device)
        final_inputs["attention_mask"] = ones_attention_mask(
            token_count=total_prompt_token_count,
            device=runtime.device,
        )
        scope = _scope(
            strategy_id=strategy_id,
            runtime_eligible=True,
            activation_reason=(
                "Prompt length does not exceed the chunked-prefill threshold."
            ),
        )
    return PreparedChunkedPrefill(
        inputs=final_inputs,
        generate_kwargs=final_generate_kwargs,
        scope=scope,
        prompt_token_count=total_prompt_token_count,
    )


def _prepare_seq2seq_source_strategy(
    runtime: LoadedRuntime,
    messages: list[Message],
    generate_kwargs: dict[str, object],
    chunk_tokens: int,
) -> PreparedChunkedPrefill:
    rendered_prompt = render_prompt_text(runtime, messages)
    prompt_tokens = [
        token_id
        for token_piece in prompt_token_id_pieces(
            resolve_stream_tokenizer(runtime),
            rendered_prompt,
        )
        for token_id in token_piece
    ]
    if not prompt_tokens:
        raise PromptExecutionError(
            "Seq2seq source ingestion produced no prompt tokens."
        )
    strategy_id = ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_SEQ2SEQ_SOURCE
    applied = len(prompt_tokens) > chunk_tokens
    activation_reason = (
        "Streamed seq2seq source tokens were built incrementally before encoder generation."
        if applied
        else "Prompt length does not exceed the streamed seq2seq source threshold."
    )
    return PreparedChunkedPrefill(
        inputs={
            "input_ids": token_tensor(prompt_tokens, device=runtime.device),
            "attention_mask": ones_attention_mask(
                token_count=len(prompt_tokens),
                device=runtime.device,
            ),
        },
        generate_kwargs=generate_kwargs,
        scope=_scope(
            strategy_id=strategy_id,
            runtime_eligible=True,
            activation_reason=activation_reason,
        ).with_activation(applied=applied, activation_reason=activation_reason),
        prompt_token_count=len(prompt_tokens),
    )


_CHUNKED_PREFILL_STRATEGIES = (
    ChunkedPrefillStrategy(
        strategy_id=ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_TEXT,
        matches=lambda runtime, runtime_kind: (
            runtime.plan.backend_id == "optimized-native"
            and runtime.processor is None
            and runtime_kind is GenericModelKind.CAUSAL_LM
        ),
        prepare=lambda runtime, messages, generate_kwargs, chunk_tokens: (
            _prepare_streamed_causal_strategy(
                runtime,
                messages,
                generate_kwargs,
                chunk_tokens,
                strategy_id=ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_TEXT,
            )
        ),
    ),
    ChunkedPrefillStrategy(
        strategy_id=ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_MULTIMODAL,
        matches=lambda runtime, runtime_kind: (
            runtime.plan.backend_id == "optimized-native"
            and runtime.processor is not None
            and runtime_kind is not GenericModelKind.SEQ2SEQ_LM
        ),
        prepare=lambda runtime, messages, generate_kwargs, chunk_tokens: (
            _prepare_streamed_causal_strategy(
                runtime,
                messages,
                generate_kwargs,
                chunk_tokens,
                strategy_id=ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_MULTIMODAL,
            )
        ),
    ),
    ChunkedPrefillStrategy(
        strategy_id=ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_TEXT,
        matches=lambda runtime, runtime_kind: (
            runtime.plan.backend_id == "transformers-generic"
            and runtime.processor is None
            and runtime_kind is GenericModelKind.CAUSAL_LM
        ),
        prepare=lambda runtime, messages, generate_kwargs, chunk_tokens: (
            _prepare_streamed_causal_strategy(
                runtime,
                messages,
                generate_kwargs,
                chunk_tokens,
                strategy_id=ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_TEXT,
            )
        ),
    ),
    ChunkedPrefillStrategy(
        strategy_id=ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_MULTIMODAL,
        matches=lambda runtime, runtime_kind: (
            runtime.plan.backend_id == "transformers-generic"
            and runtime.processor is not None
            and runtime_kind is not GenericModelKind.SEQ2SEQ_LM
        ),
        prepare=lambda runtime, messages, generate_kwargs, chunk_tokens: (
            _prepare_streamed_causal_strategy(
                runtime,
                messages,
                generate_kwargs,
                chunk_tokens,
                strategy_id=ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_MULTIMODAL,
            )
        ),
    ),
    ChunkedPrefillStrategy(
        strategy_id=ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_SEQ2SEQ_SOURCE,
        matches=lambda runtime, runtime_kind: (
            runtime.plan.backend_id == "transformers-generic"
            and runtime_kind is GenericModelKind.SEQ2SEQ_LM
        ),
        prepare=_prepare_seq2seq_source_strategy,
    ),
)


def _resolve_strategy(
    runtime: LoadedRuntime,
    runtime_kind: GenericModelKind | None,
) -> ChunkedPrefillStrategy | None:
    for strategy in _CHUNKED_PREFILL_STRATEGIES:
        if strategy.matches(runtime, runtime_kind):
            return strategy
    return None


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
        execution_boundary=ChunkedPrefillExecutionBoundary.STREAMED_PROMPT_PREPARATION,
        attention_mask_mode=ChunkedPrefillAttentionMaskMode.LAZY_PREFIX_SYNTHESIS,
        gap_inventory=_CHUNKED_PREFILL_GAP_INVENTORY,
    )

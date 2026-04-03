"""Typed scope facts for bounded chunked prefill."""

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Self

import torch

from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.loaded_runtime import LoadedRuntime


class ChunkedPrefillGapId(StrEnum):
    PROMPT_CONSTRUCTION_BEFORE_PREFILL = "prompt-construction-before-prefill"
    NON_CAUSAL_RUNTIME_EXPANSION = "non-causal-runtime-expansion"


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
    runtime_eligible: bool
    applied: bool
    activation_reason: str
    supported_backend_id: str
    supported_model_kind: str
    supported_prompt_kind: str
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
            "runtime_eligible": self.runtime_eligible,
            "applied": self.applied,
            "activation_reason": self.activation_reason,
            "supported_backend_id": self.supported_backend_id,
            "supported_model_kind": self.supported_model_kind,
            "supported_prompt_kind": self.supported_prompt_kind,
            "execution_boundary": self.execution_boundary.value,
            "attention_mask_mode": self.attention_mask_mode.value,
            "gap_inventory": [decision.to_dict() for decision in self.gap_inventory],
        }


_CHUNKED_PREFILL_GAP_INVENTORY = (
    ChunkedPrefillGapDecision(
        gap_id=ChunkedPrefillGapId.PROMPT_CONSTRUCTION_BEFORE_PREFILL,
        current_behavior=(
            "Prompt tokenization and full-prefix attention-mask materialization "
            "complete before chunked prefill begins."
        ),
        recommendation=ChunkedPrefillRecommendation.REJECT,
        rationale=(
            "Bounded chunked prefill is an execution-stage memory control after "
            "prompt construction. Streaming prompt construction would be a "
            "different feature with different tokenizer and processor "
            "contracts."
        ),
    ),
    ChunkedPrefillGapDecision(
        gap_id=ChunkedPrefillGapId.NON_CAUSAL_RUNTIME_EXPANSION,
        current_behavior=(
            "Chunked prefill is limited to optimized-native causal text "
            "runtimes and does not cover seq2seq, multimodal, or "
            "transformers-generic paths."
        ),
        recommendation=ChunkedPrefillRecommendation.REJECT,
        rationale=(
            "Those runtimes have different encoder, cache, and prompt-shape "
            "contracts. If pursued, they should ship as separate features "
            "with their own execution and benchmark semantics."
        ),
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
        return _surface(
            runtime_eligible=False,
            activation_reason="Chunked prefill requires tensor-backed input_ids.",
        )
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        return _surface(
            runtime_eligible=False,
            activation_reason="Chunked prefill requires a single batch row.",
        )
    if runtime.processor is not None:
        return _surface(
            runtime_eligible=False,
            activation_reason=(
                "Chunked prefill is limited to text prompts without a processor."
            ),
        )
    if runtime.plan.backend_id != "optimized-native":
        return _surface(
            runtime_eligible=False,
            activation_reason=(
                "Chunked prefill is limited to the optimized-native backend."
            ),
        )
    runtime_kind = (
        runtime.plan.generic_model_kind or runtime.resolved_model.generic_model_kind
    )
    if runtime_kind is not GenericModelKind.CAUSAL_LM:
        return _surface(
            runtime_eligible=False,
            activation_reason=(
                "Chunked prefill is limited to causal decoder-only text runtimes."
            ),
        )
    prefill_token_count = int(input_ids.shape[1] - 1)
    if prefill_token_count <= chunk_tokens:
        return _surface(
            runtime_eligible=True,
            activation_reason=(
                "Prompt length does not exceed the chunked-prefill threshold."
            ),
        )
    return _surface(
        runtime_eligible=True,
        activation_reason="Runtime is eligible for bounded chunked prefill.",
    )


def chunked_prefill_gap_inventory() -> tuple[ChunkedPrefillGapDecision, ...]:
    return _CHUNKED_PREFILL_GAP_INVENTORY


def _surface(
    *, runtime_eligible: bool, activation_reason: str
) -> ChunkedPrefillScopeSurface:
    return ChunkedPrefillScopeSurface(
        runtime_eligible=runtime_eligible,
        applied=False,
        activation_reason=activation_reason,
        supported_backend_id="optimized-native",
        supported_model_kind=GenericModelKind.CAUSAL_LM.value,
        supported_prompt_kind="text-only",
        execution_boundary=ChunkedPrefillExecutionBoundary.POST_TOKENIZATION,
        attention_mask_mode=ChunkedPrefillAttentionMaskMode.FULL_PREFIX_MATERIALIZED,
        gap_inventory=_CHUNKED_PREFILL_GAP_INVENTORY,
    )

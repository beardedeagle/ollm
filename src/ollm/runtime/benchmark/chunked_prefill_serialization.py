"""Chunked-prefill benchmark JSON parsing helpers."""

from collections.abc import Mapping
from typing import cast

from ollm.runtime.chunked_prefill import (
    ChunkedPrefillAttentionMaskMode,
    ChunkedPrefillExecutionBoundary,
    ChunkedPrefillGapDecision,
    ChunkedPrefillGapId,
    ChunkedPrefillRecommendation,
    ChunkedPrefillScopeSurface,
    ChunkedPrefillStrategyId,
)


def parse_chunked_prefill(
    value: object,
    *,
    require_bool,
    require_object_mapping,
    require_sequence,
    require_string,
) -> ChunkedPrefillScopeSurface:
    if not isinstance(value, Mapping):
        raise ValueError("chunked_prefill must be an object")
    payload = cast(Mapping[str, object], value)
    gap_items = require_sequence(payload, "gap_inventory")
    return ChunkedPrefillScopeSurface(
        strategy_id=_optional_strategy_id(payload, require_string=require_string),
        runtime_eligible=require_bool(payload, "runtime_eligible"),
        applied=require_bool(payload, "applied"),
        activation_reason=require_string(payload, "activation_reason"),
        execution_boundary=ChunkedPrefillExecutionBoundary(
            require_string(payload, "execution_boundary")
        ),
        attention_mask_mode=ChunkedPrefillAttentionMaskMode(
            require_string(payload, "attention_mask_mode")
        ),
        gap_inventory=tuple(
            parse_chunked_prefill_gap(
                require_object_mapping(item, f"gap_inventory[{index}]"),
                require_string=require_string,
            )
            for index, item in enumerate(gap_items)
        ),
    )


def parse_chunked_prefill_gap(
    payload: Mapping[str, object],
    *,
    require_string,
) -> ChunkedPrefillGapDecision:
    return ChunkedPrefillGapDecision(
        gap_id=ChunkedPrefillGapId(require_string(payload, "gap_id")),
        current_behavior=require_string(payload, "current_behavior"),
        recommendation=ChunkedPrefillRecommendation(
            require_string(payload, "recommendation")
        ),
        rationale=require_string(payload, "rationale"),
    )


def _optional_strategy_id(
    payload: Mapping[str, object],
    *,
    require_string,
) -> ChunkedPrefillStrategyId | None:
    value = payload.get("strategy_id")
    if value is None:
        return None
    return ChunkedPrefillStrategyId(require_string(payload, "strategy_id"))

from dataclasses import replace

import pytest
import torch
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.chunked_prefill import (
    ChunkedPrefillGapId,
    ChunkedPrefillRecommendation,
    ChunkedPrefillStrategyId,
)
from ollm.runtime.generation import (
    build_runtime_generate_kwargs,
    build_runtime_inputs,
    prepare_runtime_generate_inputs,
)
from tests.test_runtime_executor import (
    build_request,
    build_runtime_with_model,
)
from tests.test_runtime_executor_prefill import (
    ChunkedPrefillModel,
    LongMappingTokenizer,
)


def test_prepare_runtime_generate_inputs_surfaces_chunked_prefill_scope(
    monkeypatch,
) -> None:
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=LongMappingTokenizer(),
        model=ChunkedPrefillModel(),
    )
    runtime.plan = replace(
        runtime.plan,
        backend_id="optimized-native",
        generic_model_kind=GenericModelKind.CAUSAL_LM,
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("long prompt")]),
    )
    monkeypatch.setattr("ollm.runtime.generation.DEFAULT_PREFILL_CHUNK_TOKENS", 2)

    inputs = build_runtime_inputs(runtime, request.messages)
    generate_kwargs, _generation_config = build_runtime_generate_kwargs(
        runtime,
        request,
        streamer=None,
    )
    _prepared_inputs, _prepared_generate_kwargs, chunked_prefill = (
        prepare_runtime_generate_inputs(runtime, inputs, generate_kwargs)
    )

    assert chunked_prefill.runtime_eligible is True
    assert chunked_prefill.applied is True
    assert chunked_prefill.strategy_id is ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_TEXT
    assert (
        chunked_prefill.activation_reason
        == "Bounded chunked prefill ran before final decode."
    )
    gap_inventory = {
        decision.gap_id: decision.recommendation
        for decision in chunked_prefill.gap_inventory
    }
    assert gap_inventory[ChunkedPrefillGapId.PROMPT_TOKENIZATION_BEFORE_PREFILL] is (
        ChunkedPrefillRecommendation.DEFER
    )
    assert (
        gap_inventory[ChunkedPrefillGapId.FULL_ATTENTION_MASK_BEFORE_PREFILL]
        is ChunkedPrefillRecommendation.DEFER
    )
    assert gap_inventory[ChunkedPrefillGapId.SEQ2SEQ_SOURCE_PREFILL] is (
        ChunkedPrefillRecommendation.DEFER
    )


def test_prepare_runtime_generate_inputs_defers_seq2seq_source_prefill(
    monkeypatch,
) -> None:
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=LongMappingTokenizer(),
        model=ChunkedPrefillModel(),
    )
    runtime.plan = replace(
        runtime.plan,
        backend_id="optimized-native",
        generic_model_kind=GenericModelKind.SEQ2SEQ_LM,
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("long prompt")]),
    )
    monkeypatch.setattr("ollm.runtime.generation.DEFAULT_PREFILL_CHUNK_TOKENS", 2)

    inputs = build_runtime_inputs(runtime, request.messages)
    generate_kwargs, _generation_config = build_runtime_generate_kwargs(
        runtime,
        request,
        streamer=None,
    )
    _prepared_inputs, _prepared_generate_kwargs, chunked_prefill = (
        prepare_runtime_generate_inputs(runtime, inputs, generate_kwargs)
    )

    assert chunked_prefill.runtime_eligible is False
    assert chunked_prefill.applied is False
    assert chunked_prefill.strategy_id is None
    assert (
        chunked_prefill.activation_reason
        == "Seq2seq source prompts cannot use causal-cache chunked prefill."
    )


def test_t5_encoder_does_not_expose_cacheable_source_prefill() -> None:
    model = T5ForConditionalGeneration(
        T5Config(
            vocab_size=64,
            d_model=32,
            d_kv=8,
            d_ff=64,
            num_layers=2,
            num_decoder_layers=2,
            num_heads=4,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )
    )
    model.eval()
    input_ids = torch.tensor([[5, 6, 7]])
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode(), pytest.raises(ValueError, match="used as a decoder"):
        model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

from dataclasses import replace

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.chunked_prefill import (
    ChunkedPrefillGapId,
    ChunkedPrefillRecommendation,
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
    prepared = prepare_runtime_generate_inputs(runtime, inputs, generate_kwargs)

    assert prepared.chunked_prefill.runtime_eligible is True
    assert prepared.chunked_prefill.applied is True
    assert (
        prepared.chunked_prefill.activation_reason
        == "Bounded chunked prefill ran before final decode."
    )
    gap_inventory = {
        decision.gap_id: decision.recommendation
        for decision in prepared.chunked_prefill.gap_inventory
    }
    assert gap_inventory == {
        ChunkedPrefillGapId.PROMPT_CONSTRUCTION_BEFORE_PREFILL: (
            ChunkedPrefillRecommendation.REJECT
        ),
        ChunkedPrefillGapId.NON_CAUSAL_RUNTIME_EXPANSION: (
            ChunkedPrefillRecommendation.REJECT
        ),
    }


def test_prepare_runtime_generate_inputs_rejects_seq2seq_scope_extension(
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
    prepared = prepare_runtime_generate_inputs(runtime, inputs, generate_kwargs)

    assert prepared.chunked_prefill.runtime_eligible is False
    assert prepared.chunked_prefill.applied is False
    assert (
        prepared.chunked_prefill.activation_reason
        == "Chunked prefill is limited to causal decoder-only text runtimes."
    )

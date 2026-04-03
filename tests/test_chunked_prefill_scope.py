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
    prepare_chunked_prefill,
)
from ollm.runtime.chunked_prefill_support import (
    StreamedTokenBuffer,
    build_forward_input_filter,
    call_processor_for_static_inputs,
    prompt_token_id_pieces,
    render_prompt_text,
    stream_tokenizer_piece_batch_limit,
    tokenize_prompt_piece,
)
from ollm.runtime.generation import (
    build_runtime_generate_kwargs,
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


class _CountingPreTokenizer:
    def pre_tokenize_str(self, rendered_prompt: str):
        return [(piece, (0, 0)) for piece in rendered_prompt.split("|") if piece]


class _CountingBackendTokenizer:
    def __init__(self) -> None:
        self.pre_tokenizer = _CountingPreTokenizer()


class _CountingBatchTokenizer:
    def __init__(self) -> None:
        self.backend_tokenizer = _CountingBackendTokenizer()
        self.batch_calls = 0
        self.single_calls = 0

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        return_tensors=None,
        return_dict=False,
    ):
        del messages, add_generation_prompt, return_tensors, return_dict
        if not tokenize:
            return "|".join(["aa"] * 16)
        ids = [1] * 32
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        del add_special_tokens, return_attention_mask
        if isinstance(text, list):
            self.batch_calls += 1
            return {"input_ids": [[1, 2] for _ in text]}
        self.single_calls += 1
        return {"input_ids": [1, 2]}

    def decode(self, tensor, skip_special_tokens=False):
        del tensor, skip_special_tokens
        return "decoded"


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

    generate_kwargs, _generation_config = build_runtime_generate_kwargs(
        runtime,
        request,
        streamer=None,
    )
    prepared_result = prepare_runtime_generate_inputs(runtime, request, generate_kwargs)
    chunked_prefill = prepared_result.scope

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
        ChunkedPrefillRecommendation.IMPLEMENT
    )
    assert (
        gap_inventory[ChunkedPrefillGapId.FULL_ATTENTION_MASK_BEFORE_PREFILL]
        is ChunkedPrefillRecommendation.IMPLEMENT
    )
    assert gap_inventory[ChunkedPrefillGapId.SEQ2SEQ_SOURCE_PREFILL] is (
        ChunkedPrefillRecommendation.IMPLEMENT
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
        backend_id="transformers-generic",
        generic_model_kind=GenericModelKind.SEQ2SEQ_LM,
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("long prompt")]),
    )
    monkeypatch.setattr("ollm.runtime.generation.DEFAULT_PREFILL_CHUNK_TOKENS", 2)

    generate_kwargs, _generation_config = build_runtime_generate_kwargs(
        runtime,
        request,
        streamer=None,
    )
    prepared_result = prepare_runtime_generate_inputs(runtime, request, generate_kwargs)
    chunked_prefill = prepared_result.scope

    assert chunked_prefill.runtime_eligible is True
    assert chunked_prefill.applied is True
    assert (
        chunked_prefill.strategy_id
        is ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_SEQ2SEQ_SOURCE
    )
    assert (
        chunked_prefill.activation_reason
        == "Streamed seq2seq source tokens were built incrementally before encoder generation."
    )


def test_prepare_runtime_generate_inputs_batches_seq2seq_source_tokenization(
    monkeypatch,
) -> None:
    tokenizer = _CountingBatchTokenizer()
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=tokenizer,
        model=ChunkedPrefillModel(),
    )
    runtime.plan = replace(
        runtime.plan,
        backend_id="transformers-generic",
        generic_model_kind=GenericModelKind.SEQ2SEQ_LM,
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("long prompt")]),
    )
    monkeypatch.setattr("ollm.runtime.generation.DEFAULT_PREFILL_CHUNK_TOKENS", 8)

    generate_kwargs, _generation_config = build_runtime_generate_kwargs(
        runtime,
        request,
        streamer=None,
    )
    prepared_result = prepare_runtime_generate_inputs(runtime, request, generate_kwargs)

    assert prepared_result.scope.strategy_id is (
        ChunkedPrefillStrategyId.TRANSFORMERS_GENERIC_SEQ2SEQ_SOURCE
    )
    assert tokenizer.batch_calls > 0
    assert tokenizer.single_calls == 0


def test_prepare_runtime_generate_inputs_batches_causal_fallback_tokenization(
    monkeypatch,
) -> None:
    tokenizer = _CountingBatchTokenizer()
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=tokenizer,
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
    monkeypatch.setattr("ollm.runtime.generation.DEFAULT_PREFILL_CHUNK_TOKENS", 8)

    generate_kwargs, _generation_config = build_runtime_generate_kwargs(
        runtime,
        request,
        streamer=None,
    )
    prepared_result = prepare_runtime_generate_inputs(runtime, request, generate_kwargs)

    assert prepared_result.scope.strategy_id is (
        ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_TEXT
    )
    assert tokenizer.batch_calls > 0
    assert tokenizer.single_calls == 0


def test_prepare_runtime_generate_inputs_leaves_boundary_blank_without_strategy(
    monkeypatch,
) -> None:
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=LongMappingTokenizer(),
        model=ChunkedPrefillModel(),
    )
    runtime.plan = replace(
        runtime.plan,
        backend_id="custom-backend",
        generic_model_kind=GenericModelKind.CAUSAL_LM,
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("long prompt")]),
    )
    monkeypatch.setattr("ollm.runtime.generation.DEFAULT_PREFILL_CHUNK_TOKENS", 2)

    generate_kwargs, _generation_config = build_runtime_generate_kwargs(
        runtime,
        request,
        streamer=None,
    )
    prepared_result = prepare_runtime_generate_inputs(runtime, request, generate_kwargs)

    assert prepared_result.scope.strategy_id is None
    assert prepared_result.scope.runtime_eligible is False


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


def test_build_forward_input_filter_falls_back_for_uninspectable_callable() -> None:
    class UninspectableForward:
        @property
        def __signature__(self):
            raise ValueError("no signature")

        def __call__(self, **kwargs):
            return kwargs

    forward_filter = build_forward_input_filter(UninspectableForward())
    inputs: dict[str, object] = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "unexpected": "kept",
    }

    assert forward_filter(inputs) is inputs


def test_build_forward_input_filter_inspects_signature_once() -> None:
    class CountingForward:
        signature_reads = 0

        @property
        def __signature__(self):
            type(self).signature_reads += 1
            return None

        def __call__(self, *, input_ids, attention_mask):
            return input_ids, attention_mask

    forward = CountingForward()
    forward_filter = build_forward_input_filter(forward)
    first = forward_filter(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "cache_position": torch.tensor([0, 1]),
        }
    )
    second = forward_filter(
        {
            "input_ids": torch.tensor([[3, 4]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "cache_position": torch.tensor([2, 3]),
        }
    )

    assert CountingForward.signature_reads == 1
    assert set(first) == {"input_ids", "attention_mask"}
    assert set(second) == {"input_ids", "attention_mask"}


def test_prepare_chunked_prefill_rejects_non_positive_chunk_budget() -> None:
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=LongMappingTokenizer(),
        model=ChunkedPrefillModel(),
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("long prompt")]),
    )

    with pytest.raises(ValueError, match="chunk_tokens must be at least 1"):
        prepare_chunked_prefill(
            runtime=runtime,
            messages=request.messages,
            generate_kwargs={},
            chunk_tokens=0,
            eager_input_builder=lambda runtime, messages: {
                "input_ids": torch.tensor([[1]])
            },
        )


def test_tokenize_prompt_piece_disables_special_tokens_in_fallback() -> None:
    class EncodeOnlyTokenizer:
        def encode(self, piece_text: str, *, add_special_tokens: bool = True):
            assert piece_text == "piece"
            assert add_special_tokens is False
            return [7, 8]

        def __call__(self, piece_text: str):
            raise TypeError(piece_text)

    assert tokenize_prompt_piece(EncodeOnlyTokenizer(), "piece") == [7, 8]


def test_prompt_token_id_pieces_batches_fallback_tokenizer_calls() -> None:
    tokenizer = _CountingBatchTokenizer()

    pieces = tuple(
        prompt_token_id_pieces(
            tokenizer,
            "|".join(["aa"] * 16),
            piece_batch_limit=4,
        )
    )

    assert len(pieces) == 16
    assert tokenizer.batch_calls == 4
    assert tokenizer.single_calls == 0


def test_stream_tokenizer_piece_batch_limit_scales_with_chunk_budget() -> None:
    assert stream_tokenizer_piece_batch_limit(1) == 1
    assert stream_tokenizer_piece_batch_limit(8) == 2
    assert stream_tokenizer_piece_batch_limit(128) == 32


def test_streamed_token_buffer_avoids_front_delete_semantics() -> None:
    buffer = StreamedTokenBuffer()
    buffer.append_piece([1, 2, 3])
    buffer.append_piece([4, 5, 6])

    assert buffer.pop_chunk(2) == [1, 2]
    assert buffer.buffered_token_count == 4
    assert buffer.pop_chunk(3) == [3, 4, 5]
    assert buffer.remaining_tokens() == [6]
    assert buffer.total_token_count == 6


def test_render_prompt_text_falls_back_when_processor_signature_differs() -> None:
    class FragileProcessor:
        def apply_chat_template(self, messages, tokenize):
            del messages, tokenize
            raise TypeError("different signature")

    class FallbackTokenizer:
        def apply_chat_template(
            self,
            messages,
            tokenize,
            add_generation_prompt,
            return_tensors=None,
            return_dict=False,
        ):
            del messages, add_generation_prompt, return_dict, return_tensors
            if not tokenize:
                return "tokenizer-rendered"
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=FallbackTokenizer(),
        model=ChunkedPrefillModel(),
    )
    runtime.backend = replace(
        runtime.backend,
        processor=FragileProcessor(),
        tokenizer=FallbackTokenizer(),
    )

    rendered = render_prompt_text(
        runtime,
        [Message(role=MessageRole.USER, content=[ContentPart.text("hello")])],
    )

    assert rendered == "tokenizer-rendered"


def test_call_processor_for_static_inputs_omits_return_tensors_when_unsupported() -> (
    None
):
    class ReturnTensorsRejectingProcessor:
        def __call__(self, *, images):
            assert images == ["image.png"]
            return {"pixel_values": torch.tensor([[[1.0]]])}

    prepared = call_processor_for_static_inputs(
        processor=ReturnTensorsRejectingProcessor(),
        image_values=["image.png"],
        audio_values=[],
        device=torch.device("cpu"),
    )

    pixel_values = prepared["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert torch.equal(pixel_values, torch.tensor([[[1.0]]]))

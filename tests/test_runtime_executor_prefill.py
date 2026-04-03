import types
from dataclasses import replace
from typing import cast

import torch

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.generation import RuntimeExecutor
from tests.test_runtime_executor import (
    FakeModel,
    build_request,
    build_runtime_with_model,
)


class LongMappingTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        return_tensors,
        return_dict,
    ):
        del (
            messages,
            tokenize,
            add_generation_prompt,
            return_tensors,
        )
        if not return_dict:
            raise TypeError("return_dict=True required")
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

    def decode(self, tensor, skip_special_tokens=False):
        del tensor, skip_special_tokens
        return "long-decoded"


class ChunkedPrefillModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls: list[dict[str, object]] = []
        self.prefill_cache = object()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        cache_position=None,
    ):
        self.forward_calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": None
                if attention_mask is None
                else attention_mask.clone(),
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "cache_position": None
                if cache_position is None
                else cache_position.clone(),
            }
        )
        return types.SimpleNamespace(past_key_values=self.prefill_cache)


def test_runtime_executor_prefills_long_causal_prompts_in_chunks(monkeypatch) -> None:
    model = ChunkedPrefillModel()
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=LongMappingTokenizer(),
        model=model,
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

    response = RuntimeExecutor().execute(runtime, request)

    assert response.text == "long-decoded"
    assert len(model.forward_calls) == 2
    first_call = model.forward_calls[0]
    second_call = model.forward_calls[1]
    first_input_ids = cast(torch.Tensor, first_call["input_ids"])
    first_attention_mask = cast(torch.Tensor, first_call["attention_mask"])
    first_cache_position = cast(torch.Tensor, first_call["cache_position"])
    second_input_ids = cast(torch.Tensor, second_call["input_ids"])
    second_attention_mask = cast(torch.Tensor, second_call["attention_mask"])
    second_cache_position = cast(torch.Tensor, second_call["cache_position"])
    assert torch.equal(first_input_ids, torch.tensor([[1, 2]]))
    assert torch.equal(first_attention_mask, torch.tensor([[1, 1]]))
    assert torch.equal(first_cache_position, torch.tensor([0, 1]))
    assert first_call["past_key_values"] is None
    assert first_call["use_cache"] is True
    assert torch.equal(second_input_ids, torch.tensor([[3, 4]]))
    assert torch.equal(second_attention_mask, torch.tensor([[1, 1, 1, 1]]))
    assert torch.equal(second_cache_position, torch.tensor([2, 3]))
    assert second_call["past_key_values"] is model.prefill_cache
    generate_input_ids = model.generate_kwargs["input_ids"]
    assert isinstance(generate_input_ids, torch.Tensor)
    assert torch.equal(generate_input_ids, torch.tensor([[5]]))
    generate_attention_mask = model.generate_kwargs["attention_mask"]
    assert isinstance(generate_attention_mask, torch.Tensor)
    assert torch.equal(
        generate_attention_mask,
        torch.tensor([[1, 1, 1, 1, 1]]),
    )
    assert model.generate_kwargs["past_key_values"] is model.prefill_cache


def test_runtime_executor_skips_chunked_prefill_for_seq2seq_runtime(
    monkeypatch,
) -> None:
    model = ChunkedPrefillModel()
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=LongMappingTokenizer(),
        model=model,
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

    RuntimeExecutor().execute(runtime, request)

    assert model.forward_calls == []
    generate_input_ids = model.generate_kwargs["input_ids"]
    assert isinstance(generate_input_ids, torch.Tensor)
    assert torch.equal(generate_input_ids, torch.tensor([[1, 2, 3, 4, 5]]))

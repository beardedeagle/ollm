from transformers import GenerationConfig as TransformersGenerationConfig

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.config import GenerationConfig
from ollm.runtime.generation_config_support import normalized_generation_config
from tests.test_runtime_executor import (
    FakeModel,
    PlainTokenizer,
    build_request,
    build_runtime_with_model,
)


class GenerationConfigModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.generation_config = TransformersGenerationConfig.from_dict(
            {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": 50,
                "max_length": 131072,
                "eos_token_id": 7,
                "pad_token_id": 3,
                "bos_token_id": 1,
            }
        )


def test_runtime_executor_normalizes_deterministic_generation_config() -> None:
    model = GenerationConfigModel()
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=PlainTokenizer(),
        model=model,
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )
    request.generation_config = GenerationConfig(
        stream=False,
        max_new_tokens=64,
        temperature=0.0,
    )

    normalized = normalized_generation_config(runtime, request)
    assert isinstance(normalized, TransformersGenerationConfig)
    assert normalized is not model.generation_config
    assert normalized.max_new_tokens == 64
    assert normalized.max_length is None
    assert normalized.do_sample is False
    assert normalized.temperature == 1.0
    assert normalized.top_p == 1.0
    assert normalized.top_k == 50
    assert normalized.eos_token_id == 7
    assert normalized.pad_token_id == 3
    assert normalized.bos_token_id == 1


def test_runtime_executor_normalizes_sampling_generation_config() -> None:
    model = GenerationConfigModel()
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=PlainTokenizer(),
        model=model,
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )
    request.generation_config = GenerationConfig(
        stream=False,
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )

    normalized = normalized_generation_config(runtime, request)
    assert isinstance(normalized, TransformersGenerationConfig)
    assert normalized.max_new_tokens == 32
    assert normalized.max_length is None
    assert normalized.do_sample is True
    assert normalized.temperature == 0.7
    assert normalized.top_p == 0.8
    assert normalized.top_k == 20
    assert normalized.eos_token_id == 7
    assert normalized.pad_token_id == 3

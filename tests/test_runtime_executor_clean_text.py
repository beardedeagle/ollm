from ollm.app.types import ContentPart, Message, MessageRole, PromptRequest
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.catalog import ModelModality
from ollm.runtime.config import GenerationConfig
from ollm.runtime.generation import RuntimeExecutor
from tests.test_runtime_executor import (
    PlainTokenizer,
    RecordingProcessor,
    build_request,
    build_runtime,
    build_runtime_with_processor,
)


class SpecialTokenAwareTokenizer(PlainTokenizer):
    def __init__(self) -> None:
        self.decode_calls: list[bool] = []

    def decode(self, tensor, skip_special_tokens=False):
        del tensor
        self.decode_calls.append(skip_special_tokens)
        if skip_special_tokens:
            return "clean-decoded"
        return "clean-decoded<|im_end|>"


class SpecialTokenAwareProcessor(RecordingProcessor):
    def __init__(self):
        super().__init__()
        self.decode_calls: list[bool] = []

    def batch_decode(self, outputs, skip_special_tokens=False):
        del outputs
        self.decode_calls.append(skip_special_tokens)
        if skip_special_tokens:
            return ["plain-decoded"]
        return ["plain-decoded<|im_end|>"]


class CapturingStreamer:
    def __init__(
        self,
        tokenizer,
        sink,
        skip_prompt=True,
        skip_special_tokens=False,
    ):
        del tokenizer, sink, skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.text = ""


def test_runtime_executor_strips_special_tokens_from_text_response() -> None:
    runtime = build_runtime(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=SpecialTokenAwareTokenizer(),
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )

    response = RuntimeExecutor().execute(runtime, request)

    tokenizer = runtime.tokenizer
    assert isinstance(tokenizer, SpecialTokenAwareTokenizer)
    assert response.text == "clean-decoded"
    assert response.assistant_message.text_content() == "clean-decoded"
    assert tokenizer.decode_calls == [True]


def test_runtime_executor_strips_special_tokens_from_processor_response() -> None:
    processor = SpecialTokenAwareProcessor()
    capabilities = CapabilityProfile(
        support_level=SupportLevel.OPTIMIZED,
        modalities=(ModelModality.TEXT, ModelModality.IMAGE),
        requires_processor=True,
    )
    runtime = build_runtime_with_processor(capabilities, processor)
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )

    response = RuntimeExecutor().execute(runtime, request)

    assert response.text == "plain-decoded"
    assert processor.decode_calls == [True]


def test_runtime_executor_streamer_skips_special_tokens(monkeypatch) -> None:
    captured: list[bool] = []

    def build_streamer(
        tokenizer,
        sink,
        skip_prompt=True,
        skip_special_tokens=False,
    ):
        del tokenizer, sink, skip_prompt
        captured.append(skip_special_tokens)
        return CapturingStreamer(
            None,
            None,
            skip_prompt=True,
            skip_special_tokens=skip_special_tokens,
        )

    monkeypatch.setattr("ollm.runtime.generation.BufferedTextStreamer", build_streamer)
    runtime = build_runtime(CapabilityProfile(support_level=SupportLevel.GENERIC))
    request = PromptRequest(
        runtime_config=runtime.config,
        generation_config=GenerationConfig(stream=True),
        messages=[Message(role=MessageRole.USER, content=[ContentPart.text("hello")])],
    )

    RuntimeExecutor().execute(runtime, request)

    assert captured == [True]

import pytest
import torch

from ollm.app.types import ContentPart, Message, MessageRole, PromptRequest
from ollm.runtime.catalog import ModelCatalogEntry, ModelModality
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.generation import PromptExecutionError, RuntimeExecutor
from ollm.runtime.loader import LoadedRuntime


class FakeTokenizer:
    def apply_chat_template(self, messages, reasoning_effort, tokenize, add_generation_prompt, return_tensors, return_dict):
        del messages, reasoning_effort, tokenize, add_generation_prompt, return_tensors, return_dict
        return torch.tensor([[1, 2, 3]])

    def decode(self, tensor, skip_special_tokens=False):
        del tensor, skip_special_tokens
        return "decoded-response"


class FakeModel:
    def generate(self, **kwargs):
        del kwargs
        return torch.tensor([[1, 2, 3, 4, 5]])


class FakeBackend:
    def __init__(self):
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.device = torch.device("cpu")
        self.stats = None

    def DiskCache(self, cache_dir="./kvcache"):
        del cache_dir
        return None


def build_runtime(entry: ModelCatalogEntry, multimodal: bool = False) -> LoadedRuntime:
    config = RuntimeConfig(model_id=entry.model_id, device="cpu", multimodal=multimodal, use_cache=False)
    return LoadedRuntime(
        entry=entry,
        config=config,
        backend=FakeBackend(),
        model_path=config.model_path(),
    )


def build_request(runtime_config: RuntimeConfig, message: Message) -> PromptRequest:
    return PromptRequest(
        runtime_config=runtime_config,
        generation_config=GenerationConfig(stream=False),
        messages=[message],
    )


def test_runtime_executor_executes_text_request() -> None:
    entry = ModelCatalogEntry(
        model_id="llama3-1B-chat",
        summary="text",
        repo_id="repo",
        modalities=(ModelModality.TEXT,),
    )
    runtime = build_runtime(entry)
    request = build_request(runtime.config, Message(role=MessageRole.USER, content=[ContentPart.text("hello")]))
    response = RuntimeExecutor().execute(runtime, request)
    assert response.text == "decoded-response"
    assert response.assistant_message.text_content() == "decoded-response"


def test_runtime_executor_rejects_unsupported_image_input() -> None:
    entry = ModelCatalogEntry(
        model_id="llama3-1B-chat",
        summary="text",
        repo_id="repo",
        modalities=(ModelModality.TEXT,),
    )
    runtime = build_runtime(entry)
    request = build_request(runtime.config, Message(role=MessageRole.USER, content=[ContentPart.image("image.png")]))
    with pytest.raises(PromptExecutionError):
        RuntimeExecutor().execute(runtime, request)

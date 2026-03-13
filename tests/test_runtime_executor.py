import pytest
import torch

from ollm.app.types import ContentPart, Message, MessageRole, PromptRequest
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.catalog import ModelModality
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.generation import PromptExecutionError, RuntimeExecutor
from ollm.runtime.loader import LoadedRuntime
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel


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


def build_runtime(capabilities: CapabilityProfile) -> LoadedRuntime:
    config = RuntimeConfig(model_reference="llama3-1B-chat", device="cpu", multimodal=False, use_cache=False)
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("llama3-1B-chat"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="llama3-1B-chat",
        model_path=config.resolved_models_dir() / "llama3-1B-chat",
        repo_id="repo",
        revision=None,
        provider_name=None,
        catalog_entry=None,
        capabilities=capabilities,
        native_family=None,
        resolution_message="built-in alias",
    )
    return LoadedRuntime(
        resolved_model=resolved_model,
        capabilities=capabilities,
        config=config,
        backend=FakeBackend(),
        model_path=resolved_model.model_path,
    )


def build_request(runtime_config: RuntimeConfig, message: Message) -> PromptRequest:
    return PromptRequest(
        runtime_config=runtime_config,
        generation_config=GenerationConfig(stream=False),
        messages=[message],
    )


def test_runtime_executor_executes_text_request() -> None:
    capabilities = CapabilityProfile(support_level=SupportLevel.GENERIC)
    runtime = build_runtime(capabilities)
    request = build_request(runtime.config, Message(role=MessageRole.USER, content=[ContentPart.text("hello")]))
    response = RuntimeExecutor().execute(runtime, request)
    assert response.text == "decoded-response"
    assert response.assistant_message.text_content() == "decoded-response"


def test_runtime_executor_rejects_unsupported_image_input() -> None:
    capabilities = CapabilityProfile(
        support_level=SupportLevel.GENERIC,
        modalities=(ModelModality.TEXT,),
    )
    runtime = build_runtime(capabilities)
    request = build_request(runtime.config, Message(role=MessageRole.USER, content=[ContentPart.image("image.png")]))
    with pytest.raises(PromptExecutionError):
        RuntimeExecutor().execute(runtime, request)

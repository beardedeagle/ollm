import types

import pytest
import torch

from ollm.app.types import ContentPart, Message, MessageRole, PromptRequest
from ollm.runtime.backends.base import BackendRuntime
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.catalog import ModelModality
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.generation import PromptExecutionError, RuntimeExecutor
from ollm.runtime.loader import LoadedRuntime
from ollm.runtime.plan import RuntimePlan, SpecializationState
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        reasoning_effort,
        tokenize,
        add_generation_prompt,
        return_tensors,
        return_dict,
    ):
        del (
            messages,
            reasoning_effort,
            tokenize,
            add_generation_prompt,
            return_tensors,
            return_dict,
        )
        return torch.tensor([[1, 2, 3]])

    def decode(self, tensor, skip_special_tokens=False):
        del tensor, skip_special_tokens
        return "decoded-response"


class FakeModel:
    def __init__(self):
        self.generate_kwargs: dict[str, object] = {}

    def forward(self, input_ids, attention_mask=None):
        del input_ids, attention_mask
        return None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4, 5]])


class PlainTokenizer:
    def __call__(self, text, return_tensors="pt"):
        del text, return_tensors
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, tensor, skip_special_tokens=False):
        del tensor, skip_special_tokens
        return "plain-decoded"


class InspectingTokenizer(PlainTokenizer):
    def decode(self, tensor, skip_special_tokens=False):
        del skip_special_tokens
        return f"decoded:{tensor.tolist()}"


class MappingTokenizer:
    def apply_chat_template(
        self,
        messages,
        reasoning_effort,
        tokenize,
        add_generation_prompt,
        return_tensors,
        return_dict,
    ):
        del (
            messages,
            reasoning_effort,
            tokenize,
            add_generation_prompt,
            return_tensors,
        )
        if not return_dict:
            raise TypeError("return_dict=True required")
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, tensor, skip_special_tokens=False):
        del tensor, skip_special_tokens
        return "mapping-decoded"


class TensorOnlyChatTemplateTokenizer:
    def apply_chat_template(
        self,
        messages,
        reasoning_effort,
        tokenize,
        add_generation_prompt,
        return_tensors,
        return_dict,
    ):
        del (
            messages,
            reasoning_effort,
            tokenize,
            add_generation_prompt,
            return_tensors,
            return_dict,
        )
        return torch.tensor([[1, 2, 3]])

    def decode(self, tensor, skip_special_tokens=False):
        del tensor, skip_special_tokens
        return "tensor-decoded"


class FakeProcessorInputs(dict):
    def __init__(self):
        super().__init__(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "token_type_ids": torch.tensor([[0, 0, 0]]),
            }
        )
        self.to_calls: list[tuple[torch.device, torch.dtype | None]] = []

    def to(self, device, dtype=None):
        self.to_calls.append((device, dtype))
        return self


class RecordingProcessor:
    def __init__(self):
        self.messages = None
        self.inputs = FakeProcessorInputs()

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt,
        tokenize,
        return_dict,
        return_tensors,
    ):
        del add_generation_prompt, tokenize, return_dict, return_tensors
        self.messages = messages
        return self.inputs

    def batch_decode(self, outputs, skip_special_tokens=False):
        del outputs, skip_special_tokens
        return ["plain-decoded"]


class Seq2SeqModel(FakeModel):
    def generate(self, **kwargs):
        del kwargs
        return torch.tensor([[9, 8]])


class PrintingModel(FakeModel):
    def __init__(self, module):
        super().__init__()
        self._module = module

    def generate(self, **kwargs):
        del kwargs
        self._module.print("noisy-generate")
        return torch.tensor([[1, 2, 3, 4]])


def build_runtime(capabilities: CapabilityProfile, tokenizer=None) -> LoadedRuntime:
    config = RuntimeConfig(
        model_reference="llama3-1B-chat",
        device="cpu",
        multimodal=False,
        use_cache=False,
    )
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("llama3-1B-chat"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="llama3-1B-chat",
        model_path=config.resolved_models_dir() / "llama3-1B-chat",
        repo_id="repo",
        revision=None,
        catalog_entry=None,
        capabilities=capabilities,
        native_family=None,
        resolution_message="built-in alias",
        architecture="LlamaForCausalLM",
        model_type="llama",
        generic_model_kind=None,
    )
    plan = RuntimePlan(
        resolved_model=resolved_model,
        backend_id="test-backend",
        model_path=resolved_model.model_path,
        support_level=capabilities.support_level,
        generic_model_kind=None,
        supports_disk_cache=False,
        supports_cpu_offload=False,
        supports_gpu_offload=False,
        specialization_enabled=False,
        specialization_applied=False,
        specialization_provider_id=None,
        specialization_state=SpecializationState.NOT_PLANNED,
        reason="test plan",
    )
    backend = BackendRuntime(
        backend_id="test-backend",
        model=FakeModel(),
        tokenizer=FakeTokenizer() if tokenizer is None else tokenizer,
        processor=None,
        device=torch.device("cpu"),
        stats=None,
        print_suppression_modules=(),
        create_cache=lambda cache_dir, cache_strategy=None: None,
        apply_offload=lambda runtime_config: None,
    )
    return LoadedRuntime(
        resolved_model=resolved_model,
        config=config,
        plan=plan,
        backend=backend,
        model_path=resolved_model.model_path,
    )


def build_runtime_with_processor(
    capabilities: CapabilityProfile, processor: RecordingProcessor
) -> LoadedRuntime:
    runtime = build_runtime(capabilities, tokenizer=PlainTokenizer())
    runtime.backend = BackendRuntime(
        backend_id=runtime.backend.backend_id,
        model=FakeModel(),
        tokenizer=PlainTokenizer(),
        processor=processor,
        device=torch.device("cpu"),
        stats=None,
        print_suppression_modules=(),
        create_cache=lambda cache_dir, cache_strategy=None: None,
        apply_offload=lambda runtime_config: None,
    )
    return runtime


def build_runtime_with_model(
    capabilities: CapabilityProfile, tokenizer, model: FakeModel
) -> LoadedRuntime:
    runtime = build_runtime(capabilities, tokenizer=tokenizer)
    runtime.backend = BackendRuntime(
        backend_id=runtime.backend.backend_id,
        model=model,
        tokenizer=tokenizer,
        processor=None,
        device=torch.device("cpu"),
        stats=None,
        print_suppression_modules=(),
        create_cache=lambda cache_dir, cache_strategy=None: None,
        apply_offload=lambda runtime_config: None,
    )
    return runtime


def build_runtime_with_printing_module(
    capabilities: CapabilityProfile,
) -> LoadedRuntime:
    runtime = build_runtime(capabilities)
    module = types.ModuleType("fake_runtime_module")
    runtime.backend = BackendRuntime(
        backend_id="test-backend",
        model=PrintingModel(module),
        tokenizer=PlainTokenizer(),
        processor=None,
        device=torch.device("cpu"),
        stats=None,
        print_suppression_modules=(module,),
        create_cache=lambda cache_dir, cache_strategy=None: None,
        apply_offload=lambda runtime_config: None,
    )
    return runtime


def build_seq2seq_runtime() -> LoadedRuntime:
    config = RuntimeConfig(
        model_reference="t5-small", device="cpu", multimodal=False, use_cache=False
    )
    capabilities = CapabilityProfile(support_level=SupportLevel.GENERIC)
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("t5-small"),
        source_kind=ModelSourceKind.LOCAL_PATH,
        normalized_name="t5-small",
        model_path=config.resolved_models_dir() / "t5-small",
        repo_id=None,
        revision=None,
        catalog_entry=None,
        capabilities=capabilities,
        native_family=None,
        resolution_message="seq2seq",
        architecture="T5ForConditionalGeneration",
        model_type="t5",
        generic_model_kind=GenericModelKind.SEQ2SEQ_LM,
    )
    plan = RuntimePlan(
        resolved_model=resolved_model,
        backend_id="transformers-generic",
        model_path=resolved_model.model_path,
        support_level=SupportLevel.GENERIC,
        generic_model_kind=GenericModelKind.SEQ2SEQ_LM,
        supports_disk_cache=False,
        supports_cpu_offload=False,
        supports_gpu_offload=False,
        specialization_enabled=False,
        specialization_applied=False,
        specialization_provider_id=None,
        specialization_state=SpecializationState.NOT_PLANNED,
        reason="seq2seq plan",
    )
    backend = BackendRuntime(
        backend_id="transformers-generic",
        model=Seq2SeqModel(),
        tokenizer=InspectingTokenizer(),
        processor=None,
        device=torch.device("cpu"),
        stats=None,
        print_suppression_modules=(),
        create_cache=lambda cache_dir, cache_strategy=None: None,
        apply_offload=lambda runtime_config: None,
    )
    return LoadedRuntime(
        resolved_model=resolved_model,
        config=config,
        plan=plan,
        backend=backend,
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
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )
    response = RuntimeExecutor().execute(runtime, request)
    assert response.text == "decoded-response"
    assert response.assistant_message.text_content() == "decoded-response"
    assert response.metadata["specialization_state"] == "not-planned"


def test_runtime_executor_uses_structured_content_for_processor_text_requests() -> None:
    processor = RecordingProcessor()
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
    assert processor.messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        }
    ]
    assert processor.inputs.to_calls == [(torch.device("cpu"), None)]
    assert "token_type_ids" not in runtime.model.generate_kwargs


def test_runtime_executor_rejects_unsupported_image_input() -> None:
    capabilities = CapabilityProfile(
        support_level=SupportLevel.GENERIC,
        modalities=(ModelModality.TEXT,),
    )
    runtime = build_runtime(capabilities)
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.image("image.png")]),
    )
    with pytest.raises(PromptExecutionError):
        RuntimeExecutor().execute(runtime, request)


def test_runtime_executor_falls_back_when_chat_template_is_unavailable() -> None:
    capabilities = CapabilityProfile(support_level=SupportLevel.GENERIC)
    runtime = build_runtime(capabilities, tokenizer=PlainTokenizer())
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )
    response = RuntimeExecutor().execute(runtime, request)
    assert response.text == "plain-decoded"


def test_runtime_executor_decodes_seq2seq_outputs_without_prompt_slicing() -> None:
    runtime = build_seq2seq_runtime()
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )
    response = RuntimeExecutor().execute(runtime, request)
    assert response.text == "decoded:[9, 8]"


def test_runtime_executor_suppresses_module_prints_during_generate(capfd) -> None:
    runtime = build_runtime_with_printing_module(
        CapabilityProfile(support_level=SupportLevel.GENERIC)
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )

    response = RuntimeExecutor().execute(runtime, request)

    captured = capfd.readouterr()
    assert response.text == "plain-decoded"
    assert captured.out == ""


def test_runtime_executor_preserves_attention_mask_from_chat_template_mapping() -> None:
    model = FakeModel()
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=MappingTokenizer(),
        model=model,
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )

    response = RuntimeExecutor().execute(runtime, request)

    assert response.text == "mapping-decoded"
    attention_mask = model.generate_kwargs["attention_mask"]
    assert isinstance(attention_mask, torch.Tensor)
    assert torch.equal(
        attention_mask,
        torch.tensor([[1, 1, 1]]),
    )


def test_runtime_executor_synthesizes_attention_mask_for_tensor_chat_template() -> None:
    model = FakeModel()
    runtime = build_runtime_with_model(
        CapabilityProfile(support_level=SupportLevel.GENERIC),
        tokenizer=TensorOnlyChatTemplateTokenizer(),
        model=model,
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )

    response = RuntimeExecutor().execute(runtime, request)

    assert response.text == "tensor-decoded"
    attention_mask = model.generate_kwargs["attention_mask"]
    assert isinstance(attention_mask, torch.Tensor)
    assert torch.equal(
        attention_mask,
        torch.tensor([[1, 1, 1]]),
    )

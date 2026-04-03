from pathlib import Path
from typing import cast

import torch

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.runtime.backends.base import BackendRuntime
from ollm.runtime.benchmark.probe_execution import (
    build_prompt_request,
    execute_request_probe,
)
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.catalog import ModelModality
from ollm.runtime.chunked_prefill import ChunkedPrefillStrategyId
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.execution_trace import execute_request_with_trace
from ollm.runtime.loaded_runtime import LoadedRuntime
from ollm.runtime.plan import RuntimePlan, SpecializationState
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel
from tests.test_runtime_executor_prefill import (
    ChunkedPrefillModel,
    LongMappingTokenizer,
)


class BenchmarkProcessorInputs(dict):
    def __init__(self):
        super().__init__(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "token_type_ids": torch.tensor([[0, 0, 0]]),
            }
        )

    def to(self, device, dtype=None):
        del device, dtype
        return self


class BenchmarkProcessor:
    def __init__(self):
        self.inputs = BenchmarkProcessorInputs()

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt,
        tokenize,
        return_dict,
        return_tensors,
    ):
        del messages, add_generation_prompt, tokenize, return_dict, return_tensors
        return self.inputs

    def batch_decode(self, outputs, skip_special_tokens=False):
        del outputs, skip_special_tokens
        return ["decoded-benchmark"]


class BenchmarkModel:
    def __init__(self):
        self.generate_kwargs: dict[str, object] = {}

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4]])


class BenchmarkTokenizer:
    def decode(self, tensor, skip_special_tokens=False):
        del tensor, skip_special_tokens
        return "decoded-benchmark"


def _build_processor_runtime() -> LoadedRuntime:
    config = RuntimeConfig(
        model_reference="gemma3-12B",
        models_dir=Path("models"),
        device="cpu",
        backend="optimized-native",
        stats=False,
    )
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("gemma3-12B"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="gemma3-12B",
        model_path=config.resolved_models_dir() / "gemma3-12B",
        repo_id="google/gemma-3-12b-it",
        revision=None,
        catalog_entry=None,
        capabilities=CapabilityProfile(
            support_level=SupportLevel.OPTIMIZED,
            modalities=(ModelModality.TEXT, ModelModality.IMAGE),
            requires_processor=True,
            supports_disk_cache=True,
        ),
        native_family=None,
        resolution_message="gemma test",
        architecture="Gemma3ForConditionalGeneration",
        model_type="gemma3",
        generic_model_kind=None,
    )
    plan = RuntimePlan(
        resolved_model=resolved_model,
        backend_id="optimized-native",
        model_path=resolved_model.model_path,
        support_level=SupportLevel.OPTIMIZED,
        generic_model_kind=None,
        supports_disk_cache=True,
        supports_cpu_offload=True,
        supports_gpu_offload=False,
        specialization_enabled=True,
        specialization_applied=False,
        specialization_provider_id="gemma3-native",
        specialization_state=SpecializationState.PLANNED,
        reason="benchmark test",
    )
    return LoadedRuntime(
        resolved_model=resolved_model,
        config=config,
        plan=plan,
        backend=BackendRuntime(
            backend_id="optimized-native",
            model=BenchmarkModel(),
            tokenizer=BenchmarkTokenizer(),
            processor=BenchmarkProcessor(),
            device=torch.device("cpu"),
            stats=None,
            print_suppression_modules=(),
            create_cache=lambda cache_dir, cache_strategy=None, cache_lifecycle=None, cache_window_tokens=None: (
                None
            ),
            apply_offload=lambda runtime_config: None,
        ),
        model_path=resolved_model.model_path,
    )


def test_execute_request_probe_strips_processor_token_type_ids() -> None:
    runtime = _build_processor_runtime()
    request = build_prompt_request(
        runtime_config=runtime.config,
        generation_config=GenerationConfig(stream=False, max_new_tokens=1),
        messages=[
            Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
        ],
    )

    execution = execute_request_probe(runtime=runtime, request=request)

    assert execution.response_text == "decoded-benchmark"
    assert "token_type_ids" not in runtime.model.generate_kwargs
    assert execution.metrics.chunked_prefill.strategy_id is (
        ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_MULTIMODAL
    )
    assert execution.metrics.chunked_prefill.runtime_eligible is True
    assert execution.metrics.chunked_prefill.applied is False


def test_execute_request_probe_uses_chunked_prefill_for_long_causal_prompts(
    monkeypatch,
) -> None:
    runtime = _build_processor_runtime()
    runtime.backend = BackendRuntime(
        backend_id="optimized-native",
        model=ChunkedPrefillModel(),
        tokenizer=LongMappingTokenizer(),
        processor=None,
        device=torch.device("cpu"),
        stats=None,
        print_suppression_modules=(),
        create_cache=lambda cache_dir, cache_strategy=None, cache_lifecycle=None, cache_window_tokens=None: (
            None
        ),
        apply_offload=lambda runtime_config: None,
    )
    runtime.plan = RuntimePlan(
        resolved_model=runtime.plan.resolved_model,
        backend_id="optimized-native",
        model_path=runtime.plan.model_path,
        support_level=SupportLevel.OPTIMIZED,
        generic_model_kind=GenericModelKind.CAUSAL_LM,
        supports_disk_cache=True,
        supports_cpu_offload=True,
        supports_gpu_offload=False,
        specialization_enabled=True,
        specialization_applied=False,
        specialization_provider_id="llama-native",
        specialization_state=SpecializationState.PLANNED,
        reason="benchmark test",
    )
    request = build_prompt_request(
        runtime_config=runtime.config,
        generation_config=GenerationConfig(stream=False, max_new_tokens=1),
        messages=[
            Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
        ],
    )
    monkeypatch.setattr("ollm.runtime.generation.DEFAULT_PREFILL_CHUNK_TOKENS", 2)
    model = cast(ChunkedPrefillModel, runtime.model)

    execution = execute_request_probe(runtime=runtime, request=request)

    assert execution.response_text == "long-decoded"
    assert execution.metrics.chunked_prefill.strategy_id is (
        ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_TEXT
    )
    assert len(model.forward_calls) == 2
    generate_input_ids = model.generate_kwargs["input_ids"]
    assert isinstance(generate_input_ids, torch.Tensor)
    assert torch.equal(generate_input_ids, torch.tensor([[5]]))


def test_execute_request_with_trace_reports_processor_counts() -> None:
    runtime = _build_processor_runtime()
    request = build_prompt_request(
        runtime_config=runtime.config,
        generation_config=GenerationConfig(stream=False, max_new_tokens=1),
        messages=[
            Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
        ],
    )

    trace = execute_request_with_trace(runtime=runtime, request=request)

    assert trace.response_text == "decoded-benchmark"
    assert trace.prompt_token_count == 3
    assert trace.decode_prefix_token_count == 3
    assert trace.output_token_count == 1
    assert trace.cache_state is None
    assert trace.chunked_prefill.strategy_id is (
        ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_MULTIMODAL
    )
    assert trace.chunked_prefill.runtime_eligible is True
    assert trace.chunked_prefill.applied is False


def test_execute_request_with_trace_tracks_chunked_prefill_prefix_length(
    monkeypatch,
) -> None:
    runtime = _build_processor_runtime()
    runtime.backend = BackendRuntime(
        backend_id="optimized-native",
        model=ChunkedPrefillModel(),
        tokenizer=LongMappingTokenizer(),
        processor=None,
        device=torch.device("cpu"),
        stats=None,
        print_suppression_modules=(),
        create_cache=lambda cache_dir, cache_strategy=None, cache_lifecycle=None, cache_window_tokens=None: (
            None
        ),
        apply_offload=lambda runtime_config: None,
    )
    runtime.plan = RuntimePlan(
        resolved_model=runtime.plan.resolved_model,
        backend_id="optimized-native",
        model_path=runtime.plan.model_path,
        support_level=SupportLevel.OPTIMIZED,
        generic_model_kind=GenericModelKind.CAUSAL_LM,
        supports_disk_cache=True,
        supports_cpu_offload=True,
        supports_gpu_offload=False,
        specialization_enabled=True,
        specialization_applied=False,
        specialization_provider_id="llama-native",
        specialization_state=SpecializationState.PLANNED,
        reason="benchmark test",
    )
    request = build_prompt_request(
        runtime_config=runtime.config,
        generation_config=GenerationConfig(stream=False, max_new_tokens=1),
        messages=[
            Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
        ],
    )
    monkeypatch.setattr("ollm.runtime.generation.DEFAULT_PREFILL_CHUNK_TOKENS", 2)

    trace = execute_request_with_trace(runtime=runtime, request=request)
    model = cast(ChunkedPrefillModel, runtime.model)

    assert trace.response_text == "long-decoded"
    assert trace.prompt_token_count == 5
    assert trace.decode_prefix_token_count == 1
    assert trace.output_token_count == 4
    assert trace.chunked_prefill.strategy_id is (
        ChunkedPrefillStrategyId.OPTIMIZED_NATIVE_TEXT
    )
    assert trace.chunked_prefill.runtime_eligible is True
    assert trace.chunked_prefill.applied is True
    assert len(model.forward_calls) == 2


def test_execute_request_with_trace_starts_timing_before_prefill(
    monkeypatch,
) -> None:
    runtime = _build_processor_runtime()
    runtime.backend = BackendRuntime(
        backend_id="optimized-native",
        model=ChunkedPrefillModel(),
        tokenizer=LongMappingTokenizer(),
        processor=None,
        device=torch.device("cpu"),
        stats=None,
        print_suppression_modules=(),
        create_cache=lambda cache_dir, cache_strategy=None, cache_lifecycle=None, cache_window_tokens=None: (
            None
        ),
        apply_offload=lambda runtime_config: None,
    )
    runtime.plan = RuntimePlan(
        resolved_model=runtime.plan.resolved_model,
        backend_id="optimized-native",
        model_path=runtime.plan.model_path,
        support_level=SupportLevel.OPTIMIZED,
        generic_model_kind=GenericModelKind.CAUSAL_LM,
        supports_disk_cache=True,
        supports_cpu_offload=True,
        supports_gpu_offload=False,
        specialization_enabled=True,
        specialization_applied=False,
        specialization_provider_id="llama-native",
        specialization_state=SpecializationState.PLANNED,
        reason="benchmark test",
    )
    request = build_prompt_request(
        runtime_config=runtime.config,
        generation_config=GenerationConfig(stream=False, max_new_tokens=1),
        messages=[
            Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
        ],
    )
    monkeypatch.setattr("ollm.runtime.generation.DEFAULT_PREFILL_CHUNK_TOKENS", 2)
    order: list[str] = []

    def wrapped_perf_counter() -> float:
        order.append("time")
        return 123.0

    original_prepare = execute_request_with_trace.__globals__[
        "prepare_runtime_generate_inputs"
    ]

    def wrapped_prepare(runtime, inputs, generate_kwargs):
        order.append("prepare")
        return original_prepare(runtime, inputs, generate_kwargs)

    monkeypatch.setattr(
        "ollm.runtime.execution_trace.time.perf_counter", wrapped_perf_counter
    )
    monkeypatch.setattr(
        "ollm.runtime.execution_trace.prepare_runtime_generate_inputs",
        wrapped_prepare,
    )

    trace = execute_request_with_trace(runtime=runtime, request=request)

    assert trace.generation_started_at == 123.0
    assert order.index("time") < order.index("prepare")

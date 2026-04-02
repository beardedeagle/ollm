import torch

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.runtime.backends.base import BackendRuntime
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loaded_runtime import LoadedRuntime
from ollm.runtime.plan import RuntimePlan, SpecializationState
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel
from tests.test_runtime_executor import (
    InspectingTokenizer,
    Seq2SeqModel,
    build_request,
)


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
        create_cache=lambda cache_dir, cache_strategy=None, cache_lifecycle=None, cache_window_tokens=None: (
            None
        ),
        apply_offload=lambda runtime_config: None,
    )
    return LoadedRuntime(
        resolved_model=resolved_model,
        config=config,
        plan=plan,
        backend=backend,
        model_path=resolved_model.model_path,
    )


def test_runtime_executor_decodes_seq2seq_outputs_without_prompt_slicing() -> None:
    runtime = build_seq2seq_runtime()
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )

    response = RuntimeExecutor().execute(runtime, request)

    assert response.text == "decoded:[9, 8]"

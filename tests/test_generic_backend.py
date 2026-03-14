from pathlib import Path

import pytest
import torch

from ollm.runtime.backends.transformers_generic import TransformersGenericBackend
from ollm.runtime.capabilities import CapabilityProfile
from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel


class FakeModel:
    def __init__(self):
        self.device = None
        self.eval_calls = 0
        self.generation_config = type("GenerationConfig", (), {"pad_token_id": None})()

    def eval(self):
        self.eval_calls += 1

    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    pad_token = None
    eos_token = "</s>"
    pad_token_id = 7


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()


def build_plan(kind: GenericModelKind, model_path: Path) -> RuntimePlan:
    resolved_model = ResolvedModel(
        reference=ModelReference.parse(str(model_path)),
        source_kind=ModelSourceKind.LOCAL_PATH,
        normalized_name="model",
        model_path=model_path,
        repo_id=None,
        revision=None,
        provider_name=None,
        catalog_entry=None,
        capabilities=CapabilityProfile(support_level=SupportLevel.GENERIC),
        native_family=None,
        resolution_message="generic",
        architecture="Qwen2ForCausalLM",
        model_type="qwen2",
        generic_model_kind=kind,
    )
    return RuntimePlan(
        resolved_model=resolved_model,
        backend_id="transformers-generic",
        model_path=model_path,
        support_level=SupportLevel.GENERIC,
        generic_model_kind=kind,
        supports_disk_cache=False,
        supports_cpu_offload=False,
        supports_gpu_offload=False,
        specialization_enabled=False,
        specialization_provider_id=None,
        reason="generic",
    )


def _create_safe_model_dir(tmp_path: Path, name: str) -> Path:
    model_path = tmp_path / name
    model_path.mkdir()
    (model_path / "model.safetensors").write_text("safe", encoding="utf-8")
    return model_path


def test_transformers_generic_backend_loads_causal_models_via_injected_callables(tmp_path: Path) -> None:
    fake_model = FakeModel()
    model_path = _create_safe_model_dir(tmp_path, "causal")
    backend = TransformersGenericBackend(
        causal_loader=lambda **kwargs: fake_model,
        tokenizer_loader=lambda path, trust_remote_code=False: FakeTokenizer(),
    )
    runtime = backend.load(build_plan(GenericModelKind.CAUSAL_LM, model_path), RuntimeConfig(device="cpu"))
    assert runtime.backend_id == "transformers-generic"
    assert runtime.model is fake_model
    assert runtime.device == torch.device("cpu")
    assert runtime.tokenizer.pad_token == "</s>"
    assert runtime.model.generation_config.pad_token_id == 7


def test_transformers_generic_backend_uses_processor_tokenizer_for_image_text_models(tmp_path: Path) -> None:
    fake_model = FakeModel()
    model_path = _create_safe_model_dir(tmp_path, "vision")
    backend = TransformersGenericBackend(
        image_text_loader=lambda **kwargs: fake_model,
        tokenizer_loader=lambda path, trust_remote_code=False: None,
        processor_loader=lambda path, trust_remote_code=False: FakeProcessor(),
    )
    runtime = backend.load(
        build_plan(GenericModelKind.IMAGE_TEXT_TO_TEXT, model_path),
        RuntimeConfig(device="cpu", multimodal=True),
    )
    assert isinstance(runtime.processor, FakeProcessor)
    assert isinstance(runtime.tokenizer, FakeTokenizer)


def test_transformers_generic_backend_rejects_custom_offload_controls(tmp_path: Path) -> None:
    fake_model = FakeModel()
    model_path = _create_safe_model_dir(tmp_path, "offload")
    backend = TransformersGenericBackend(
        causal_loader=lambda **kwargs: fake_model,
        tokenizer_loader=lambda path, trust_remote_code=False: FakeTokenizer(),
    )
    runtime = backend.load(build_plan(GenericModelKind.CAUSAL_LM, model_path), RuntimeConfig(device="cpu"))
    with pytest.raises(ValueError):
        runtime.apply_offload(RuntimeConfig(device="cpu", offload_cpu_layers=1))


def test_transformers_generic_backend_rejects_unsafe_weight_artifacts(tmp_path: Path) -> None:
    model_path = tmp_path / "unsafe"
    model_path.mkdir()
    (model_path / "pytorch_model.bin").write_text("unsafe", encoding="utf-8")
    backend = TransformersGenericBackend(
        causal_loader=lambda **kwargs: FakeModel(),
        tokenizer_loader=lambda path, trust_remote_code=False: FakeTokenizer(),
    )
    with pytest.raises(ValueError, match="safetensors"):
        backend.load(build_plan(GenericModelKind.CAUSAL_LM, model_path), RuntimeConfig(device="cpu"))

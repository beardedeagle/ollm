import json
from pathlib import Path
from types import SimpleNamespace

import ollm.runtime.specialization.providers as providers_module
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, NativeFamily, ResolvedModel
from ollm.runtime.specialization.passes.base import SpecializationPassId
from ollm.runtime.specialization.providers import (
    Gemma3SpecializationProvider,
    LlamaSpecializationProvider,
    VoxtralSpecializationProvider,
)
from ollm.utils import Stats


class FakeLoadedModel:
    def eval(self):
        return self

    def to(self, device):
        del device
        return self


def test_llama_specialization_provider_allows_placeholder_weight_mismatches(
    tmp_path: Path, monkeypatch
) -> None:
    captured_kwargs: dict[str, object] = {}
    model_dir = tmp_path / "smollm2"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("safe", encoding="utf-8")

    class FakeLlamaModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del cls, args
            captured_kwargs.update(kwargs)
            return FakeLoadedModel()

    fake_module = SimpleNamespace(
        loader=None,
        stats=None,
        MyLlamaForCausalLM=FakeLlamaModel,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.import_module",
        lambda module_name: fake_module,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.get_attention_implementation",
        lambda: None,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.SingleDenseWeightsLoader",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: object(),
    )

    provider = LlamaSpecializationProvider()
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("HuggingFaceTB/SmolLM2-1.7B-Instruct"),
        source_kind=ModelSourceKind.HUGGING_FACE,
        normalized_name="HuggingFaceTB--SmolLM2-1.7B-Instruct",
        model_path=model_dir,
        repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        revision=None,
        catalog_entry=None,
        capabilities=CapabilityProfile(support_level=SupportLevel.OPTIMIZED),
        native_family=NativeFamily.LLAMA,
        resolution_message="optimized",
        architecture="LlamaForCausalLM",
        model_type="llama",
        generic_model_kind=GenericModelKind.CAUSAL_LM,
    )

    artifacts = provider.load(
        resolved_model,
        RuntimeConfig(
            model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            models_dir=tmp_path,
            device="cpu",
            use_cache=False,
        ),
        stats=None,
    )

    assert artifacts.model is not None
    assert captured_kwargs["ignore_mismatched_sizes"] is True
    assert captured_kwargs["use_safetensors"] is True
    assert captured_kwargs["device_map"] == "cpu"
    assert captured_kwargs["trust_remote_code"] is False
    assert captured_kwargs["low_cpu_mem_usage"] is True
    assert "attn_implementation" not in captured_kwargs


def test_llama_specialization_provider_wires_stats_into_gds_loader(
    tmp_path: Path, monkeypatch
) -> None:
    fake_stats = Stats()
    model_dir = tmp_path / "smollm2"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("safe", encoding="utf-8")

    class FakeLlamaModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del cls, args, kwargs
            return FakeLoadedModel()

    fake_module = SimpleNamespace(
        loader=None,
        stats=None,
        MyLlamaForCausalLM=FakeLlamaModel,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.import_module",
        lambda module_name: fake_module,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.get_attention_implementation",
        lambda: None,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.SingleDenseWeightsLoader",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: object(),
    )

    provider = LlamaSpecializationProvider()
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("HuggingFaceTB/SmolLM2-1.7B-Instruct"),
        source_kind=ModelSourceKind.HUGGING_FACE,
        normalized_name="HuggingFaceTB--SmolLM2-1.7B-Instruct",
        model_path=model_dir,
        repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        revision=None,
        catalog_entry=None,
        capabilities=CapabilityProfile(support_level=SupportLevel.OPTIMIZED),
        native_family=NativeFamily.LLAMA,
        resolution_message="optimized",
        architecture="LlamaForCausalLM",
        model_type="llama",
        generic_model_kind=GenericModelKind.CAUSAL_LM,
    )

    provider.load(
        resolved_model,
        RuntimeConfig(
            model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            models_dir=tmp_path,
            device="cpu",
            use_cache=False,
        ),
        stats=fake_stats,
    )

    assert fake_module.stats is fake_stats
    assert providers_module.gds_loader_module.stats is fake_stats


def test_gemma3_specialization_provider_skips_processor_for_text_only_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "gemma3"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma3",
                "architectures": ["Gemma3ForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("safe", encoding="utf-8")

    class FakeGemmaTextModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del cls, args, kwargs
            return FakeLoadedModel()

    class FakeGemmaMultimodalModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del cls, args, kwargs
            return FakeLoadedModel()

    fake_module = SimpleNamespace(
        loader=None,
        stats=None,
        MyGemma3ForCausalLM=FakeGemmaTextModel,
        MyGemma3ForConditionalGeneration=FakeGemmaMultimodalModel,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.import_module",
        lambda module_name: fake_module,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.get_attention_implementation",
        lambda: None,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.DenseWeightsLoader",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.AutoProcessor.from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("AutoProcessor should not load for text-only Gemma3")
        ),
    )

    provider = Gemma3SpecializationProvider()
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("gemma3-12B"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="gemma3-12b",
        model_path=model_dir,
        repo_id="google/gemma-3-12b-it",
        revision=None,
        catalog_entry=None,
        capabilities=CapabilityProfile(support_level=SupportLevel.OPTIMIZED),
        native_family=NativeFamily.GEMMA3,
        resolution_message="optimized",
        architecture="Gemma3ForCausalLM",
        model_type="gemma3",
        generic_model_kind=GenericModelKind.CAUSAL_LM,
    )

    artifacts = provider.load(
        resolved_model,
        RuntimeConfig(
            model_reference="gemma3-12B",
            models_dir=tmp_path,
            device="cpu",
            multimodal=False,
            use_cache=False,
        ),
        stats=None,
    )

    assert artifacts.processor is None


def test_voxtral_specialization_provider_declares_mlp_chunking(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "voxtral"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "voxtral",
                "architectures": ["VoxtralForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("safe", encoding="utf-8")

    class FakeVoxtralModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del cls, args, kwargs
            return FakeLoadedModel()

    fake_module = SimpleNamespace(
        loader=None,
        stats=None,
        dense_projection_chunk_rows=None,
        MyVoxtralForConditionalGeneration=FakeVoxtralModel,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.import_module",
        lambda module_name: fake_module,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.get_attention_implementation",
        lambda: None,
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.DenseWeightsLoader",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "ollm.runtime.specialization.providers.AutoProcessor.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=object()),
    )

    provider = VoxtralSpecializationProvider()
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("voxtral-small-24B"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="voxtral-small-24b",
        model_path=model_dir,
        repo_id="mistralai/Voxtral-Small-24B-2507",
        revision=None,
        catalog_entry=None,
        capabilities=CapabilityProfile(support_level=SupportLevel.OPTIMIZED),
        native_family=NativeFamily.VOXTRAL,
        resolution_message="optimized",
        architecture="VoxtralForConditionalGeneration",
        model_type="voxtral",
        generic_model_kind=GenericModelKind.SEQ2SEQ_LM,
    )

    artifacts = provider.load(
        resolved_model,
        RuntimeConfig(
            model_reference="voxtral-small-24B",
            models_dir=tmp_path,
            device="cpu",
            multimodal=True,
            use_cache=False,
            dense_projection_chunk_rows=2048,
        ),
        stats=None,
    )

    assert artifacts.provided_pass_ids == (SpecializationPassId.MLP_CHUNKING,)
    assert fake_module.dense_projection_chunk_rows == 2048

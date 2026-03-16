import json
from pathlib import Path
from types import SimpleNamespace

from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, NativeFamily, ResolvedModel
from ollm.runtime.specialization.providers import LlamaSpecializationProvider


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
        "ollm.runtime.specialization.providers._get_attention_implementation",
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

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ollm.inference import AutoInference, Inference
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.resolver import ModelSourceKind, NativeFamily, ResolvedModel
from ollm.runtime.specialization import SpecializationRegistry
from ollm.runtime.specialization.base import (
    OptimizedModelArtifacts,
    SpecializationMatch,
    SpecializationProvider,
    SpecializationTraits,
)


class FakeProvider(SpecializationProvider):
    provider_id = "fake-llama"
    native_family = NativeFamily.LLAMA

    def __init__(self):
        self.load_calls: list[str] = []

    def match(self, resolved_model: ResolvedModel, config: RuntimeConfig) -> SpecializationMatch | None:
        del config
        if resolved_model.native_family is not NativeFamily.LLAMA:
            return None
        return SpecializationMatch(
            provider_id=self.provider_id,
            native_family=self.native_family,
            reason="matched fake llama specialization",
            traits=SpecializationTraits(
                supports_disk_cache=True,
                supports_cpu_offload=True,
                supports_gpu_offload=False,
            ),
        )

    def load(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        stats,
    ) -> OptimizedModelArtifacts:
        del config
        self.load_calls.append(resolved_model.reference.raw)
        return OptimizedModelArtifacts(
            model=object(),
            tokenizer=object(),
            processor=None,
            device=torch.device("cpu"),
            stats=stats,
            supports_disk_cache=True,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            print_suppression_modules=(),
            create_cache=lambda cache_dir: str(cache_dir),
            apply_cpu_offload=lambda layers_num: None,
            apply_gpu_offload=None,
        )


def test_auto_inference_rejects_missing_local_model_dir(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-model"
    with pytest.raises(ValueError, match="Local model directory does not exist"):
        AutoInference(str(missing_path), device="cpu", logging=False)


class StubAutoInference(AutoInference):
    def _load_optimized_model(
        self,
        model_path: Path,
        source_kind: ModelSourceKind,
        raw_reference: str,
        catalog_entry=None,
    ) -> None:
        del model_path, source_kind, raw_reference, catalog_entry
        self.model = object()


def test_auto_inference_rejects_unsafe_adapter_artifacts(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.bin").write_text("unsafe", encoding="utf-8")

    with pytest.raises(ValueError, match="safetensors"):
        StubAutoInference(str(model_dir), adapter_dir=str(adapter_dir), device="cpu", logging=False)


def test_inference_load_model_delegates_to_specialization_registry(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama3-1B-chat"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    provider = FakeProvider()
    registry = SpecializationRegistry((provider,))

    inference = Inference(
        "llama3-1B-chat",
        device="cpu",
        logging=False,
        specialization_registry=registry,
    )
    inference.load_model(str(model_dir))

    assert provider.load_calls == ["llama3-1B-chat"]
    assert inference.model is not None
    assert inference.tokenizer is not None
    assert inference.loaded_resolved_model is not None
    assert inference.loaded_resolved_model.reference.raw == "llama3-1B-chat"
    assert inference.loaded_specialization_provider_id == "fake-llama"
    assert inference.loaded_applied_specialization_pass_ids == ()


def test_auto_inference_preserves_local_path_reference_for_optimized_loads(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama-local"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    provider = FakeProvider()
    registry = SpecializationRegistry((provider,))

    auto_inference = AutoInference(
        str(model_dir),
        device="cpu",
        logging=False,
        specialization_registry=registry,
    )

    assert provider.load_calls == [str(model_dir)]
    assert auto_inference.loaded_resolved_model is not None
    assert auto_inference.loaded_resolved_model.reference.raw == str(model_dir)
    assert auto_inference.model_reference == str(model_dir)
    assert auto_inference.model_id == str(model_dir)
    assert auto_inference.optimized_model_id == "llama3-1B-chat"


def test_auto_inference_does_not_claim_sharded_local_llama_is_8b(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama-local"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    auto_inference = StubAutoInference(str(model_dir), device="cpu", logging=False)

    assert auto_inference.model_id == str(model_dir)
    assert auto_inference.optimized_model_id == "llama3-1B-chat"


def test_importing_ollm_does_not_require_peft() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockPeft(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'peft' or fullname.startswith('peft.'):
            raise ModuleNotFoundError('blocked peft import')
        return None

sys.meta_path.insert(0, BlockPeft())
sys.path.insert(0, r'REPO_SRC')
import ollm
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script.replace("REPO_SRC", str(repo_root / "src"))],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_get_attn_implementation_does_not_write_to_stdout(capfd) -> None:
    logging.getLogger("ollm.inference").handlers = []

    result = Inference("llama3-1B-chat", device="cpu", logging=False)
    del result
    from ollm.inference import get_attn_implementation

    assert get_attn_implementation() is None
    captured = capfd.readouterr()
    assert captured.out == ""

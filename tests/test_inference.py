import json
import logging
import sys
from pathlib import Path

import pytest
import torch

from ollm.async_io import subprocess_run_process
from ollm.inference import (
    HF_RUNTIME_ARTIFACT_PATTERNS,
    AutoInference,
    Inference,
    ManagedModelDownloadError,
    download_hf_snapshot,
    hf_runtime_artifacts_complete,
)
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
        self.cache_calls: list[tuple[Path, str | None, str | None]] = []
        self.cpu_offload_calls: list[tuple[int, ...]] = []

    def match(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> SpecializationMatch | None:
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
        cache_calls = self.cache_calls
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
            create_cache=lambda cache_dir, cache_strategy=None, cache_lifecycle=None, cache_window_tokens=None: (
                cache_calls.append((cache_dir, cache_strategy, cache_lifecycle))
                or str(cache_dir)
            ),
            apply_cpu_offload=lambda layer_indices: self.cpu_offload_calls.append(
                layer_indices
            ),
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
        StubAutoInference(
            str(model_dir), adapter_dir=str(adapter_dir), device="cpu", logging=False
        )


def test_inference_load_model_delegates_to_specialization_registry(
    tmp_path: Path,
) -> None:
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


def test_inference_disk_cache_forwards_explicit_strategy(tmp_path: Path) -> None:
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

    cache_root = tmp_path / "cache-root"
    cache_value = inference.DiskCache(
        cache_dir=str(cache_root),
        cache_strategy="streamed-segmented",
    )

    assert cache_value == str(cache_root.resolve())
    assert provider.cache_calls == [(cache_root.resolve(), "streamed-segmented", None)]


def test_inference_offload_layers_to_cpu_applies_policy_selected_indices(
    tmp_path: Path,
) -> None:
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
        device="mps",
        logging=False,
        specialization_registry=registry,
    )
    inference.load_model(str(model_dir))
    inference.model = type("FakeModel", (), {"num_hidden_layers": 8})()

    inference.offload_layers_to_cpu(2, policy="suffix")

    assert provider.cpu_offload_calls == [(6, 7)]


def test_inference_load_model_does_not_prune_caller_owned_local_files(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "llama3-1B-chat"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (model_dir / "training_args.bin").write_text("leave me alone", encoding="utf-8")
    provider = FakeProvider()
    registry = SpecializationRegistry((provider,))

    inference = Inference(
        "llama3-1B-chat",
        device="cpu",
        logging=False,
        specialization_registry=registry,
    )
    inference.load_model(str(model_dir))

    assert (model_dir / "training_args.bin").exists()


def test_auto_inference_preserves_local_path_reference_for_optimized_loads(
    tmp_path: Path,
) -> None:
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


def test_auto_inference_does_not_claim_sharded_local_llama_is_8b(
    tmp_path: Path,
) -> None:
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
    result = subprocess_run_process(
        (sys.executable, "-c", script.replace("REPO_SRC", str(repo_root / "src"))),
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


def test_download_hf_snapshot_uses_runtime_allowlist_and_repairs_target_dir(
    tmp_path: Path, monkeypatch
) -> None:
    captured_call: dict[str, object] = {}
    target_dir = tmp_path / "model"
    target_dir.mkdir()
    (target_dir / "training_args.bin").write_text("unsafe", encoding="utf-8")

    def fake_snapshot_download(**kwargs) -> None:
        captured_call.update(kwargs)
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (local_dir / "model.safetensors").write_text("safe", encoding="utf-8")
        (local_dir / "README.md").write_text("docs", encoding="utf-8")

    monkeypatch.setattr(
        "ollm.runtime.materialization.snapshot_download",
        fake_snapshot_download,
    )

    download_hf_snapshot("repo/model", str(target_dir), revision="main")

    assert captured_call["repo_id"] == "repo/model"
    assert captured_call["revision"] == "main"
    assert captured_call["local_dir"] == str(target_dir.resolve())
    allow_patterns = captured_call["allow_patterns"]
    assert isinstance(allow_patterns, list)
    assert tuple(allow_patterns) == HF_RUNTIME_ARTIFACT_PATTERNS
    assert (target_dir / "config.json").exists()
    assert (target_dir / "tokenizer.json").exists()
    assert (target_dir / "model.safetensors").exists()
    assert not (target_dir / "training_args.bin").exists()
    assert not (target_dir / "README.md").exists()


def test_hf_runtime_artifacts_complete_requires_all_indexed_shards(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "model"
    target_dir.mkdir()
    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    (target_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (target_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 2},
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00002.safetensors",
                    "model.layers.1.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (target_dir / "model-00001-of-00002.safetensors").write_text(
        "safe",
        encoding="utf-8",
    )

    assert hf_runtime_artifacts_complete(target_dir) is False

    (target_dir / "model-00002-of-00002.safetensors").write_text(
        "safe",
        encoding="utf-8",
    )

    assert hf_runtime_artifacts_complete(target_dir) is True


def test_hf_runtime_artifacts_complete_rejects_out_of_root_shard_reference(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "model"
    target_dir.mkdir()
    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    (target_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    outside_shard = tmp_path / "outside.safetensors"
    outside_shard.write_text("safe", encoding="utf-8")
    (target_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {"model.layers.0.weight": "../outside.safetensors"},
            }
        ),
        encoding="utf-8",
    )

    assert hf_runtime_artifacts_complete(target_dir) is False


def test_download_hf_snapshot_raises_clear_error_after_partial_sharded_failure(
    tmp_path: Path, monkeypatch
) -> None:
    target_dir = tmp_path / "model"

    def fake_snapshot_download(**kwargs) -> None:
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (local_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 2},
                    "weight_map": {
                        "model.layers.0.weight": "model-00001-of-00002.safetensors",
                        "model.layers.1.weight": "model-00002-of-00002.safetensors",
                    },
                }
            ),
            encoding="utf-8",
        )
        (local_dir / "model-00001-of-00002.safetensors").write_text(
            "safe",
            encoding="utf-8",
        )
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(
        "ollm.runtime.materialization.snapshot_download",
        fake_snapshot_download,
    )

    with pytest.raises(ManagedModelDownloadError, match="missing shard referenced"):
        download_hf_snapshot("repo/model", str(target_dir))

    assert (target_dir / "config.json").exists()
    assert (target_dir / "tokenizer.json").exists()
    assert (target_dir / "model-00001-of-00002.safetensors").exists()

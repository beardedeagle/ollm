import json
from pathlib import Path

import pytest
import torch

from ollm.runtime.backends.base import BackendRuntime, ExecutionBackend
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.loader import RuntimeLoader
from ollm.runtime.specialization.base import SpecializationApplicationError


class FakeModel:
    def eval(self):
        return self

    def to(self, device):
        del device
        return self


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        del text, return_tensors
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, tensor, skip_special_tokens=False):
        del tensor, skip_special_tokens
        return "ok"


class FailingOptimizedBackend(ExecutionBackend):
    backend_id = "optimized-native"

    def load(self, plan, config) -> BackendRuntime:
        del config
        raise SpecializationApplicationError(
            "Optimized specialization 'llama-native' could not apply 'disk-cache': missing cache support",
            provider_id="llama-native",
            planned_pass_ids=plan.specialization_pass_ids,
            applied_pass_ids=(),
            failed_pass_id=plan.specialization_pass_ids[0],
            details={"reason": "missing cache support"},
        )


class FakeGenericBackend(ExecutionBackend):
    backend_id = "transformers-generic"

    def __init__(self):
        self.load_count = 0

    def load(self, plan, config) -> BackendRuntime:
        del config
        self.load_count += 1
        return BackendRuntime(
            backend_id=self.backend_id,
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            processor=None,
            device=torch.device("cpu"),
            stats=None,
            print_suppression_modules=(),
            create_cache=lambda cache_dir: None,
            apply_offload=lambda runtime_config: None,
        )


def _snapshot_downloader(
    repo_id: str, model_dir: str, force_download: bool, revision: str | None
) -> None:
    del repo_id, force_download, revision
    target = Path(model_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (target / "tokenizer.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_text("safe", encoding="utf-8")


def test_runtime_loader_falls_back_to_generic_when_optimized_application_fails(
    tmp_path: Path,
) -> None:
    generic_backend = FakeGenericBackend()
    loader = RuntimeLoader(
        backends=(FailingOptimizedBackend(), generic_backend),
        snapshot_downloader=_snapshot_downloader,
    )

    runtime = loader.load(
        RuntimeConfig(
            model_reference="llama3-1B-chat",
            models_dir=tmp_path / "models",
            device="cpu",
            use_cache=False,
        )
    )

    assert generic_backend.load_count == 1
    assert runtime.backend.backend_id == "transformers-generic"
    assert runtime.plan.backend_id == "transformers-generic"
    assert runtime.plan.specialization_enabled is True
    assert runtime.plan.specialization_applied is False
    assert runtime.plan.specialization_provider_id == "llama-native"
    assert runtime.plan.applied_specialization_pass_ids == ()
    assert runtime.plan.fallback_reason is not None
    assert "disk-cache" in runtime.plan.fallback_reason
    assert runtime.capabilities.support_level is runtime.plan.support_level
    assert runtime.capabilities.supports_disk_cache is runtime.plan.supports_disk_cache


def test_runtime_loader_raises_when_no_safe_generic_fallback_exists(
    tmp_path: Path,
) -> None:
    loader = RuntimeLoader(
        backends=(FailingOptimizedBackend(),),
        snapshot_downloader=_snapshot_downloader,
    )

    with pytest.raises(ValueError, match="could not apply 'disk-cache'"):
        loader.load(
            RuntimeConfig(
                model_reference="llama3-1B-chat",
                models_dir=tmp_path / "models",
                device="cpu",
                use_cache=False,
            )
        )

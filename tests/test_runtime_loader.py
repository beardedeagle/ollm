import json
from pathlib import Path

import torch

from ollm.runtime.backends.base import BackendRuntime, ExecutionBackend
from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.loader import RuntimeLoader


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


class FakeGenericBackend(ExecutionBackend):
    backend_id = "transformers-generic"

    def __init__(self):
        self.loaded_backend_ids: list[str] = []

    def load(self, plan, config) -> BackendRuntime:
        del config
        self.loaded_backend_ids.append(plan.backend_id or "missing")
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


def test_runtime_loader_materializes_non_catalog_hf_reference_and_routes_to_generic_backend(tmp_path: Path) -> None:
    calls: list[tuple[str, str, bool, str | None]] = []

    def snapshot_downloader(repo_id: str, model_dir: str, force_download: bool, revision: str | None) -> None:
        calls.append((repo_id, model_dir, force_download, revision))
        target = Path(model_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
            encoding="utf-8",
        )

    fake_backend = FakeGenericBackend()
    loader = RuntimeLoader(
        backends=(fake_backend,),
        snapshot_downloader=snapshot_downloader,
    )
    config = RuntimeConfig(
        model_reference="hf://Qwen/Qwen2.5-7B-Instruct@main",
        models_dir=tmp_path / "models",
        device="cpu",
        use_cache=False,
    )

    runtime = loader.load(config)

    assert calls == [
        (
            "Qwen/Qwen2.5-7B-Instruct",
            str((tmp_path / "models" / "Qwen--Qwen2.5-7B-Instruct--main").resolve()),
            False,
            "main",
        )
    ]
    assert runtime.plan.backend_id == "transformers-generic"
    assert runtime.resolved_model.generic_model_kind is not None
    assert runtime.resolved_model.reference.raw == "hf://Qwen/Qwen2.5-7B-Instruct@main"
    assert fake_backend.loaded_backend_ids == ["transformers-generic"]


def test_runtime_loader_routes_built_in_alias_with_adapter_to_generic_backend(tmp_path: Path) -> None:
    def snapshot_downloader(repo_id: str, model_dir: str, force_download: bool, revision: str | None) -> None:
        del repo_id, force_download, revision
        target = Path(model_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
            encoding="utf-8",
        )

    fake_backend = FakeGenericBackend()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    loader = RuntimeLoader(
        backends=(fake_backend,),
        snapshot_downloader=snapshot_downloader,
    )
    config = RuntimeConfig(
        model_reference="llama3-1B-chat",
        models_dir=tmp_path / "models",
        adapter_dir=adapter_dir,
        device="cpu",
        use_cache=False,
    )

    runtime = loader.load(config)

    assert runtime.plan.backend_id == "transformers-generic"
    assert runtime.resolved_model.architecture == "LlamaForCausalLM"


def test_runtime_loader_routes_built_in_alias_to_generic_when_specialization_is_disabled(tmp_path: Path) -> None:
    def snapshot_downloader(repo_id: str, model_dir: str, force_download: bool, revision: str | None) -> None:
        del repo_id, force_download, revision
        target = Path(model_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
            encoding="utf-8",
        )

    fake_backend = FakeGenericBackend()
    loader = RuntimeLoader(
        backends=(fake_backend,),
        snapshot_downloader=snapshot_downloader,
    )
    config = RuntimeConfig(
        model_reference="llama3-1B-chat",
        models_dir=tmp_path / "models",
        device="cpu",
        use_specialization=False,
    )

    runtime = loader.load(config)

    assert runtime.plan.backend_id == "transformers-generic"
    assert runtime.plan.specialization_enabled is False


def test_runtime_loader_plan_predicts_generic_backend_when_specialization_is_disabled(tmp_path: Path) -> None:
    loader = RuntimeLoader()
    runtime_plan = loader.plan(
        RuntimeConfig(
            model_reference="llama3-1B-chat",
            models_dir=tmp_path / "models",
            device="cpu",
            use_specialization=False,
        )
    )

    assert runtime_plan.backend_id == "transformers-generic"
    assert runtime_plan.support_level is SupportLevel.GENERIC


def test_runtime_loader_plan_does_not_materialize_missing_model_references(tmp_path: Path) -> None:
    calls: list[tuple[str, str, bool, str | None]] = []

    def snapshot_downloader(repo_id: str, model_dir: str, force_download: bool, revision: str | None) -> None:
        calls.append((repo_id, model_dir, force_download, revision))

    loader = RuntimeLoader(snapshot_downloader=snapshot_downloader)
    runtime_plan = loader.plan(
        RuntimeConfig(
            model_reference="hf://Qwen/Qwen2.5-7B-Instruct@main",
            models_dir=tmp_path / "models",
            device="cpu",
            use_cache=False,
        )
    )

    assert calls == []
    assert runtime_plan.backend_id == "transformers-generic"
    assert runtime_plan.support_level is SupportLevel.GENERIC

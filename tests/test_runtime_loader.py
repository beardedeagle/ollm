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
            create_cache=lambda cache_dir, cache_strategy=None, cache_lifecycle=None, cache_window_tokens=None: (
                None
            ),
            apply_offload=lambda runtime_config: None,
        )


def test_runtime_loader_materializes_non_catalog_hf_reference_and_routes_to_generic_backend(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, bool, str | None]] = []

    def snapshot_downloader(
        repo_id: str, model_dir: str, force_download: bool, revision: str | None
    ) -> None:
        calls.append((repo_id, model_dir, force_download, revision))
        target = Path(model_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
            encoding="utf-8",
        )
        (target / "tokenizer.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_text("safe", encoding="utf-8")

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


def test_runtime_loader_materializes_catalog_matching_hf_reference_into_alias_path(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, bool, str | None]] = []

    def snapshot_downloader(
        repo_id: str, model_dir: str, force_download: bool, revision: str | None
    ) -> None:
        calls.append((repo_id, model_dir, force_download, revision))
        target = Path(model_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma3",
                    "architectures": ["Gemma3ForConditionalGeneration"],
                    "vision_config": {},
                }
            ),
            encoding="utf-8",
        )
        (target / "tokenizer.json").write_text("{}", encoding="utf-8")
        (target / "processor_config.json").write_text("{}", encoding="utf-8")
        (target / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_text("safe", encoding="utf-8")

    fake_backend = FakeGenericBackend()
    loader = RuntimeLoader(
        backends=(fake_backend,),
        snapshot_downloader=snapshot_downloader,
    )
    runtime = loader.load(
        RuntimeConfig(
            model_reference="hf://google/gemma-3-12b-it",
            models_dir=tmp_path / "models",
            device="cpu",
            use_cache=False,
            use_specialization=False,
            multimodal=True,
        )
    )

    assert calls == [
        (
            "google/gemma-3-12b-it",
            str((tmp_path / "models" / "gemma3-12B").resolve()),
            False,
            None,
        )
    ]
    assert runtime.plan.backend_id == "transformers-generic"
    assert runtime.resolved_model.reference.raw == "hf://google/gemma-3-12b-it"
    assert fake_backend.loaded_backend_ids == ["transformers-generic"]


def test_runtime_loader_routes_built_in_alias_with_adapter_to_generic_backend(
    tmp_path: Path,
) -> None:
    def snapshot_downloader(
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


def test_runtime_loader_routes_built_in_alias_to_generic_when_specialization_is_disabled(
    tmp_path: Path,
) -> None:
    def snapshot_downloader(
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
    assert runtime.config.kv_cache_strategy == "resident"
    assert runtime.plan.details["strategy_selector_rule_id"] == "no-disk-cache-support"


def test_runtime_loader_coerces_resident_fallback_to_runtime_scoped_lifecycle(
    tmp_path: Path,
) -> None:
    def snapshot_downloader(
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

    fake_backend = FakeGenericBackend()
    loader = RuntimeLoader(
        backends=(fake_backend,),
        snapshot_downloader=snapshot_downloader,
    )
    runtime = loader.load(
        RuntimeConfig(
            model_reference="llama3-1B-chat",
            models_dir=tmp_path / "models",
            device="cpu",
            use_specialization=False,
            kv_cache_lifecycle="persistent",
        )
    )

    assert runtime.config.kv_cache_strategy == "resident"
    assert runtime.config.kv_cache_lifecycle == "runtime-scoped"
    assert (
        runtime.plan.details["strategy_selector_applied_kv_cache_lifecycle"]
        == "runtime-scoped"
    )
    assert (
        runtime.plan.details["strategy_selector_lifecycle_reason"]
        == "Resident fallback requires runtime-scoped lifecycle."
    )


def test_runtime_loader_plan_predicts_generic_backend_when_specialization_is_disabled(
    tmp_path: Path,
) -> None:
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


def test_runtime_loader_plan_does_not_materialize_missing_model_references(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, bool, str | None]] = []

    def snapshot_downloader(
        repo_id: str, model_dir: str, force_download: bool, revision: str | None
    ) -> None:
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


def test_runtime_loader_repairs_existing_managed_model_directory(
    tmp_path: Path,
) -> None:
    managed_model_dir = (
        tmp_path / "models" / "HuggingFaceTB--SmolLM2-1.7B-Instruct"
    ).resolve()
    managed_model_dir.mkdir(parents=True)
    (managed_model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (managed_model_dir / "model.safetensors").write_text("safe", encoding="utf-8")
    (managed_model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (managed_model_dir / "training_args.bin").write_text("unsafe", encoding="utf-8")
    (managed_model_dir / "README.md").write_text("docs", encoding="utf-8")
    stale_docs_dir = managed_model_dir / "notes"
    stale_docs_dir.mkdir()
    (stale_docs_dir / "usage.txt").write_text("ignore", encoding="utf-8")

    fake_backend = FakeGenericBackend()
    loader = RuntimeLoader(backends=(fake_backend,))

    runtime = loader.load(
        RuntimeConfig(
            model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            models_dir=tmp_path / "models",
            device="cpu",
            use_cache=False,
            use_specialization=False,
        )
    )

    assert runtime.plan.backend_id == "transformers-generic"
    assert fake_backend.loaded_backend_ids == ["transformers-generic"]
    assert (managed_model_dir / "config.json").exists()
    assert (managed_model_dir / "model.safetensors").exists()
    assert not (managed_model_dir / "training_args.bin").exists()
    assert not (managed_model_dir / "README.md").exists()
    assert not stale_docs_dir.exists()


def test_runtime_loader_redownloads_incomplete_managed_model_directory(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, bool, str | None]] = []
    managed_model_dir = (
        tmp_path / "models" / "HuggingFaceTB--SmolLM2-1.7B-Instruct"
    ).resolve()
    managed_model_dir.mkdir(parents=True)
    (managed_model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (managed_model_dir / "model.safetensors").write_text("safe", encoding="utf-8")

    def snapshot_downloader(
        repo_id: str, model_dir: str, force_download: bool, revision: str | None
    ) -> None:
        calls.append((repo_id, model_dir, force_download, revision))
        target = Path(model_dir)
        (target / "tokenizer.json").write_text("{}", encoding="utf-8")

    fake_backend = FakeGenericBackend()
    loader = RuntimeLoader(
        backends=(fake_backend,),
        snapshot_downloader=snapshot_downloader,
    )

    runtime = loader.load(
        RuntimeConfig(
            model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            models_dir=tmp_path / "models",
            device="cpu",
            use_cache=False,
            use_specialization=False,
        )
    )

    assert runtime.plan.backend_id == "transformers-generic"
    assert calls == [
        (
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            str(managed_model_dir),
            False,
            None,
        )
    ]
    assert (managed_model_dir / "tokenizer.json").exists()


def test_runtime_loader_redownloads_partially_sharded_managed_model_directory(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, bool, str | None]] = []
    managed_model_dir = (
        tmp_path / "models" / "HuggingFaceTB--SmolLM2-1.7B-Instruct"
    ).resolve()
    managed_model_dir.mkdir(parents=True)
    (managed_model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (managed_model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (managed_model_dir / "model.safetensors.index.json").write_text(
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
    (managed_model_dir / "model-00001-of-00002.safetensors").write_text(
        "safe",
        encoding="utf-8",
    )

    def snapshot_downloader(
        repo_id: str, model_dir: str, force_download: bool, revision: str | None
    ) -> None:
        calls.append((repo_id, model_dir, force_download, revision))
        target = Path(model_dir)
        (target / "model-00002-of-00002.safetensors").write_text(
            "safe",
            encoding="utf-8",
        )

    fake_backend = FakeGenericBackend()
    loader = RuntimeLoader(
        backends=(fake_backend,),
        snapshot_downloader=snapshot_downloader,
    )

    runtime = loader.load(
        RuntimeConfig(
            model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            models_dir=tmp_path / "models",
            device="cpu",
            use_cache=False,
            use_specialization=False,
        )
    )

    assert runtime.plan.backend_id == "transformers-generic"
    assert calls == [
        (
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            str(managed_model_dir),
            False,
            None,
        )
    ]
    assert (managed_model_dir / "model-00002-of-00002.safetensors").exists()

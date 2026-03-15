"""Low-level optimized-native inference helpers and snapshot utilities."""

import importlib.util
import logging
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from ollm.runtime.catalog import ModelCatalogEntry, find_model_catalog_entry
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelResolver, ModelSourceKind
from ollm.runtime.safety import validate_safe_adapter_artifacts
from ollm.runtime.specialization import (
    SpecializationPipeline,
    SpecializationRegistry,
    apply_specialization,
    build_default_specialization_registry,
)
from ollm.utils import Stats

LOGGER = logging.getLogger(__name__)


def get_attn_implementation() -> str | None:
    """Return the preferred attention implementation identifier when available."""
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    LOGGER.warning("flash_attention_2 is not imported. The context length will be limited.")
    return None


def download_hf_snapshot(
    repo_id: str,
    model_dir: str,
    force_download: bool = False,
    revision: str | None = None,
) -> None:
    """Download a Hugging Face snapshot into a local model directory."""
    LOGGER.info("Downloading model snapshot for %s.", repo_id)
    snapshot_download_fn = cast(Any, snapshot_download)
    snapshot_download_fn(
        repo_id=repo_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        force_download=force_download,
        revision=revision,
    )


class Inference:
    """Direct optimized-native helper for built-in aliases and matching local native families."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        logging: bool = True,
        multimodality: bool = False,
        specialization_registry: SpecializationRegistry | None = None,
        resolver: ModelResolver | None = None,
    ):
        self.model_id = model_id
        self.model_reference = model_id
        self.optimized_model_id = model_id
        self.device = torch.device(device)
        self.multimodality = multimodality
        self.stats = Stats() if logging else None
        self._specialization_registry = (
            build_default_specialization_registry()
            if specialization_registry is None
            else specialization_registry
        )
        self._resolver = resolver or ModelResolver()
        self._specialization_pipeline = SpecializationPipeline()
        self._cache_factory = None
        self._apply_cpu_offload = None
        self._apply_gpu_offload = None
        self.loaded_resolved_model = None
        self.loaded_specialization_provider_id = None
        self.loaded_applied_specialization_pass_ids = ()

    def hf_download(self, model_dir: str, force_download: bool = False) -> None:
        """Download the built-in optimized alias into a local directory."""
        entry = find_model_catalog_entry(self.optimized_model_id)
        if entry is None:
            raise ValueError(
                f"Inference only supports built-in optimized aliases. Received {self.optimized_model_id!r}."
            )
        download_hf_snapshot(entry.repo_id, model_dir, force_download=force_download)

    def ini_model(self, models_dir: str = "./models/", force_download: bool = False) -> None:
        """Download if needed and then load the optimized-native runtime."""
        entry = find_model_catalog_entry(self.optimized_model_id)
        if entry is None:
            raise ValueError(
                f"Inference only supports built-in optimized aliases. Received {self.optimized_model_id!r}."
            )

        model_dir = Path(models_dir).expanduser().resolve() / entry.model_id
        if force_download or not model_dir.exists():
            self.hf_download(str(model_dir), force_download=force_download)

        self.load_model(str(model_dir))

    def load_model(self, model_dir: str) -> None:
        """Load an optimized-native runtime from a local directory."""
        model_path = Path(model_dir).expanduser().resolve()
        if not model_path.exists() or not model_path.is_dir():
            raise ValueError(f"Model directory does not exist: {model_path}")
        entry = find_model_catalog_entry(self.optimized_model_id)
        if entry is None:
            raise ValueError(
                f"Inference only supports built-in optimized aliases. Received {self.optimized_model_id!r}."
            )
        self._load_optimized_model(
            model_path,
            ModelSourceKind.BUILTIN,
            self.model_reference,
            catalog_entry=entry,
        )

    def _load_optimized_model(
        self,
        model_path: Path,
        source_kind: ModelSourceKind,
        raw_reference: str,
        catalog_entry: ModelCatalogEntry | None = None,
    ) -> None:
        entry = catalog_entry
        if source_kind is ModelSourceKind.BUILTIN:
            entry = find_model_catalog_entry(self.optimized_model_id)
            if entry is None:
                raise ValueError(
                    f"Inference only supports built-in optimized aliases. Received {self.optimized_model_id!r}."
                )

        LOGGER.info("Loading optimized model from %s.", model_path)
        runtime_config = RuntimeConfig(
            model_reference=raw_reference,
            device=str(self.device),
            multimodal=self.multimodality,
            stats=self.stats is not None,
        )
        resolved_model = self._resolver.inspect_materialized_model(
            ModelReference.parse(raw_reference),
            model_path,
            source_kind=source_kind,
            repo_id=None if entry is None else entry.repo_id,
            revision=None,
            provider_name=None,
            catalog_entry=entry,
        )
        specialization_match = self._specialization_registry.select(resolved_model, runtime_config)
        if specialization_match is None:
            raise ValueError(
                f"No optimized specialization provider is available for {self.model_id!r} at {model_path}"
            )
        planned_specialization = self._specialization_pipeline.plan(
            resolved_model,
            runtime_config,
            specialization_match.provider_id,
        )
        artifacts = self._specialization_registry.load(
            specialization_match.provider_id,
            resolved_model,
            runtime_config,
            self.stats,
        )
        applied_specialization = apply_specialization(
            planned_specialization,
            artifacts,
            runtime_config,
        )
        self.loaded_resolved_model = resolved_model
        self.loaded_specialization_provider_id = specialization_match.provider_id
        self.loaded_applied_specialization_pass_ids = applied_specialization.applied_pass_ids
        self.model = artifacts.model
        self.tokenizer = artifacts.tokenizer
        if artifacts.processor is None:
            if hasattr(self, "processor"):
                delattr(self, "processor")
        else:
            self.processor = artifacts.processor
        self._cache_factory = artifacts.create_cache
        self._apply_cpu_offload = artifacts.apply_cpu_offload
        self._apply_gpu_offload = artifacts.apply_gpu_offload

    def offload_layers_to_cpu(self, layers_num: int) -> None:
        """Apply CPU layer offload through the selected specialization when supported."""
        if self._apply_cpu_offload is None:
            raise ValueError(f"{self.model_id} does not support CPU layer offload")
        self._apply_cpu_offload(layers_num)

    def offload_layers_to_gpu_cpu(self, gpu_layers_num: int = 0, cpu_layers_num: int = 0) -> None:
        """Apply mixed GPU/CPU layer placement when the specialization exposes it."""
        if gpu_layers_num == 0 and cpu_layers_num == 0:
            return
        if self._apply_gpu_offload is None:
            raise ValueError(f"{self.model_id} does not support GPU layer offload")
        self._apply_gpu_offload(gpu_layers_num, cpu_layers_num)

    def DiskCache(self, cache_dir: str = "./kvcache"):
        """Create the specialization-backed disk KV cache when supported."""
        if self._cache_factory is None:
            return None
        return self._cache_factory(Path(cache_dir).expanduser().resolve())


class AutoInference(Inference):
    """Optimized-native helper that infers the matching local native family from a model directory."""

    def __init__(
        self,
        model_dir: str,
        adapter_dir: str | None = None,
        device: str = "cuda:0",
        logging: bool = True,
        multimodality: bool = False,
        specialization_registry: SpecializationRegistry | None = None,
        resolver: ModelResolver | None = None,
    ):
        model_path = Path(model_dir).expanduser().resolve()
        if not model_path.exists() or not model_path.is_dir():
            raise ValueError(f"Local model directory does not exist: {model_path}")

        config = AutoConfig.from_pretrained(model_path)
        architectures = getattr(config, "architectures", None) or ()
        architecture = architectures[0] if architectures else None
        if architecture == "LlamaForCausalLM":
            optimized_model_id = "llama3-1B-chat"
        elif architecture in {"Gemma3ForConditionalGeneration", "Gemma3ForCausalLM"}:
            optimized_model_id = "gemma3-12B"
        else:
            raise ValueError(
                f"The current optimized path cannot run architecture {architecture!r}. "
                "Use a built-in optimized alias or a compatible local Llama/Gemma3 model directory."
            )

        super().__init__(
            str(model_path),
            device=device,
            logging=logging,
            multimodality=multimodality,
            specialization_registry=specialization_registry,
            resolver=resolver,
        )
        self.optimized_model_id = optimized_model_id
        self.model_reference = str(model_path)
        self._load_optimized_model(
            model_path,
            ModelSourceKind.LOCAL_PATH,
            self.model_reference,
        )
        if adapter_dir:
            from peft import LoraConfig, get_peft_model

            adapter_path = Path(adapter_dir).expanduser().resolve()
            if not adapter_path.exists() or not adapter_path.is_dir():
                raise ValueError(f"Adapter directory does not exist: {adapter_path}")
            validate_safe_adapter_artifacts(adapter_path)
            peft_config = LoraConfig.from_pretrained(str(adapter_path))
            self.model = get_peft_model(self.model, peft_config)
            self.model.load_adapter(str(adapter_path), adapter_name="default", use_safetensors=True)

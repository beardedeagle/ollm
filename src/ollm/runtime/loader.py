from dataclasses import dataclass
from pathlib import Path

from ollm.inference import AutoInference, Inference, download_hf_snapshot
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.resolver import ModelResolver, ModelSourceKind, ResolvedModel


@dataclass(slots=True)
class LoadedRuntime:
    resolved_model: ResolvedModel
    capabilities: CapabilityProfile
    config: RuntimeConfig
    backend: Inference | AutoInference
    model_path: Path

    @property
    def model(self):
        return self.backend.model

    @property
    def tokenizer(self):
        return self.backend.tokenizer

    @property
    def processor(self):
        return getattr(self.backend, "processor", None)

    @property
    def device(self):
        return self.backend.device


class RuntimeLoader:
    def __init__(self, resolver: ModelResolver | None = None):
        self._resolver = resolver or ModelResolver()

    def resolve(self, model_reference: str, models_dir: Path) -> ResolvedModel:
        return self._resolver.resolve(model_reference, models_dir)

    def discover_local_models(self, models_dir: Path) -> tuple[ResolvedModel, ...]:
        return self._resolver.discover_local_models(models_dir)

    def download(self, model_reference: str, models_dir: Path, force_download: bool = False) -> Path:
        resolved_model = self.resolve(model_reference, models_dir)
        if resolved_model.source_kind is ModelSourceKind.LOCAL_PATH:
            if resolved_model.model_path is None or not resolved_model.model_path.exists():
                raise ValueError(f"Local model path does not exist: {model_reference}")
            return resolved_model.model_path
        if not resolved_model.is_downloadable() or resolved_model.repo_id is None or resolved_model.model_path is None:
            raise ValueError(f"Model reference '{model_reference}' does not support snapshot download")
        if resolved_model.model_path.exists() and not force_download:
            return resolved_model.model_path
        resolved_model.model_path.parent.mkdir(parents=True, exist_ok=True)
        download_hf_snapshot(
            resolved_model.repo_id,
            str(resolved_model.model_path),
            force_download=force_download,
            revision=resolved_model.revision,
        )
        return resolved_model.model_path

    def load(self, config: RuntimeConfig) -> LoadedRuntime:
        config.validate()
        resolved_model = self.resolve(config.model_reference, config.resolved_models_dir())
        if resolved_model.capabilities.support_level is SupportLevel.UNSUPPORTED:
            reason = resolved_model.capabilities.details.get("reason", resolved_model.resolution_message)
            raise ValueError(reason)
        if resolved_model.source_kind is ModelSourceKind.PROVIDER:
            raise ValueError(
                f"Provider-backed model references are not executable yet: {resolved_model.reference.raw}"
            )

        model_path = self._ensure_local_model(resolved_model, config.force_download)
        adapter_dir = config.resolved_adapter_dir()
        logging_enabled = config.stats or config.verbose
        use_optimized_runtime = (
            resolved_model.catalog_entry is not None
            and resolved_model.source_kind in {ModelSourceKind.BUILTIN, ModelSourceKind.HUGGING_FACE}
            and adapter_dir is None
        )

        if use_optimized_runtime:
            backend = Inference(
                resolved_model.catalog_entry.model_id,
                device=config.device,
                logging=logging_enabled,
                multimodality=config.multimodal,
            )
            backend.load_model(str(model_path))
        else:
            backend = AutoInference(
                str(model_path),
                adapter_dir=None if adapter_dir is None else str(adapter_dir),
                device=config.device,
                logging=logging_enabled,
                multimodality=config.multimodal,
            )

        self._apply_offload(backend, config)
        return LoadedRuntime(
            resolved_model=resolved_model,
            capabilities=resolved_model.capabilities,
            config=config,
            backend=backend,
            model_path=model_path,
        )

    def _apply_offload(self, backend: Inference | AutoInference, config: RuntimeConfig) -> None:
        if config.offload_gpu_layers > 0:
            backend.offload_layers_to_gpu_cpu(
                gpu_layers_num=config.offload_gpu_layers,
                cpu_layers_num=config.offload_cpu_layers,
            )
            return
        if config.offload_cpu_layers > 0:
            backend.offload_layers_to_cpu(layers_num=config.offload_cpu_layers)

    def _ensure_local_model(self, resolved_model: ResolvedModel, force_download: bool) -> Path:
        if resolved_model.model_path is None:
            raise ValueError(f"Model reference '{resolved_model.reference.raw}' does not resolve to a local model path")
        if resolved_model.model_path.exists() and not force_download:
            return resolved_model.model_path
        if resolved_model.repo_id is None:
            raise ValueError(f"Local model path does not exist: {resolved_model.model_path}")
        resolved_model.model_path.parent.mkdir(parents=True, exist_ok=True)
        download_hf_snapshot(
            resolved_model.repo_id,
            str(resolved_model.model_path),
            force_download=force_download,
            revision=resolved_model.revision,
        )
        return resolved_model.model_path

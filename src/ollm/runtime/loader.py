from dataclasses import dataclass
from pathlib import Path

from ollm.inference import AutoInference, Inference, download_model_snapshot
from ollm.runtime.catalog import ModelCatalogEntry, get_model_catalog_entry
from ollm.runtime.config import RuntimeConfig


@dataclass(slots=True)
class LoadedRuntime:
    entry: ModelCatalogEntry
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
    def download(self, model_id: str, models_dir: Path, force_download: bool = False) -> Path:
        entry = get_model_catalog_entry(model_id)
        model_path = models_dir.expanduser().resolve() / entry.model_id
        if model_path.exists() and not force_download:
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        download_model_snapshot(entry.model_id, str(model_path), force_download=force_download)
        return model_path

    def load(self, config: RuntimeConfig) -> LoadedRuntime:
        config.validate()
        entry = get_model_catalog_entry(config.model_id)
        model_path = config.model_path()
        adapter_dir = config.resolved_adapter_dir()
        logging_enabled = config.stats or config.verbose

        if adapter_dir is not None:
            if not model_path.exists() or config.force_download:
                self.download(config.model_id, config.resolved_models_dir(), force_download=config.force_download)
            backend = AutoInference(
                str(model_path),
                adapter_dir=str(adapter_dir),
                device=config.device,
                logging=logging_enabled,
                multimodality=config.multimodal,
            )
        else:
            backend = Inference(
                config.model_id,
                device=config.device,
                logging=logging_enabled,
                multimodality=config.multimodal,
            )
            backend.ini_model(models_dir=str(config.resolved_models_dir()), force_download=config.force_download)

        self._apply_offload(backend, config)
        return LoadedRuntime(entry=entry, config=config, backend=backend, model_path=model_path)

    def _apply_offload(self, backend: Inference | AutoInference, config: RuntimeConfig) -> None:
        if config.offload_gpu_layers > 0:
            backend.offload_layers_to_gpu_cpu(
                gpu_layers_num=config.offload_gpu_layers,
                cpu_layers_num=config.offload_cpu_layers,
            )
            return
        if config.offload_cpu_layers > 0:
            backend.offload_layers_to_cpu(layers_num=config.offload_cpu_layers)

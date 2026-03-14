from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ollm.inference import download_hf_snapshot
from ollm.runtime.backend_selector import BackendSelector
from ollm.runtime.backends.base import BackendRuntime, ExecutionBackend
from ollm.runtime.backends.native_optimized import NativeOptimizedBackend
from ollm.runtime.backends.transformers_generic import TransformersGenericBackend
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.resolver import ModelResolver, ModelSourceKind, ResolvedModel
from ollm.runtime.specialization import SpecializationRegistry, build_default_specialization_registry


@dataclass(slots=True)
class LoadedRuntime:
    resolved_model: ResolvedModel
    config: RuntimeConfig
    backend: BackendRuntime
    model_path: Path
    plan: RuntimePlan

    @property
    def capabilities(self):
        return self.resolved_model.capabilities

    @property
    def model(self):
        return self.backend.model

    @property
    def tokenizer(self):
        return self.backend.tokenizer

    @property
    def processor(self):
        return self.backend.processor

    @property
    def device(self):
        return self.backend.device


class RuntimeLoader:
    def __init__(
        self,
        resolver: ModelResolver | None = None,
        selector: BackendSelector | None = None,
        backends: tuple[ExecutionBackend, ...] | None = None,
        snapshot_downloader: Callable[[str, str, bool, str | None], None] | None = None,
        specialization_registry: SpecializationRegistry | None = None,
    ):
        self._resolver = resolver or ModelResolver()
        self._specialization_registry = (
            build_default_specialization_registry()
            if specialization_registry is None
            else specialization_registry
        )
        self._selector = selector or BackendSelector(specialization_registry=self._specialization_registry)
        backend_list = backends or (
            NativeOptimizedBackend(specialization_registry=self._specialization_registry),
            TransformersGenericBackend(),
        )
        self._backends = {backend.backend_id: backend for backend in backend_list}
        self._snapshot_downloader = download_hf_snapshot if snapshot_downloader is None else snapshot_downloader

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
        self._snapshot_downloader(
            resolved_model.repo_id,
            str(resolved_model.model_path),
            force_download,
            resolved_model.revision,
        )
        return resolved_model.model_path

    def load(self, config: RuntimeConfig) -> LoadedRuntime:
        config.validate()
        resolved_model = self.resolve(config.model_reference, config.resolved_models_dir())
        if resolved_model.source_kind is ModelSourceKind.PROVIDER:
            raise ValueError(f"Provider-backed model references are not executable yet: {resolved_model.reference.raw}")

        model_path = self._ensure_local_model(resolved_model, config.force_download)
        execution_model = self._refresh_materialized_model(resolved_model, model_path)
        runtime_plan = self._selector.select(execution_model, config)
        if not runtime_plan.is_executable():
            raise ValueError(runtime_plan.reason)
        self._validate_runtime_plan(runtime_plan, config)
        backend_impl = self._backends.get(runtime_plan.backend_id)
        if backend_impl is None:
            raise ValueError(f"No runtime backend is registered for '{runtime_plan.backend_id}'")
        backend_runtime = backend_impl.load(runtime_plan, config)
        backend_runtime.apply_offload(config)
        return LoadedRuntime(
            resolved_model=execution_model,
            config=config,
            backend=backend_runtime,
            model_path=model_path,
            plan=runtime_plan,
        )

    def plan(self, config: RuntimeConfig) -> RuntimePlan:
        config.validate()
        resolved_model = self.resolve(config.model_reference, config.resolved_models_dir())
        if resolved_model.source_kind is ModelSourceKind.PROVIDER:
            return self._selector.select(resolved_model, config)
        if resolved_model.model_path is None or not resolved_model.model_path.exists():
            return self._selector.select(resolved_model, config)
        execution_model = self._refresh_materialized_model(resolved_model, resolved_model.model_path)
        return self._selector.select(execution_model, config)

    def _refresh_materialized_model(self, resolved_model: ResolvedModel, model_path: Path) -> ResolvedModel:
        return self._resolver.inspect_materialized_model(
            resolved_model.reference,
            model_path,
            source_kind=resolved_model.source_kind,
            repo_id=resolved_model.repo_id,
            revision=resolved_model.revision,
            provider_name=resolved_model.provider_name,
            catalog_entry=resolved_model.catalog_entry,
        )

    def _ensure_local_model(self, resolved_model: ResolvedModel, force_download: bool) -> Path:
        if resolved_model.model_path is None:
            raise ValueError(f"Model reference '{resolved_model.reference.raw}' does not resolve to a local model path")
        if resolved_model.model_path.exists() and not force_download:
            return resolved_model.model_path
        if resolved_model.repo_id is None:
            raise ValueError(f"Local model path does not exist: {resolved_model.model_path}")
        resolved_model.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshot_downloader(
            resolved_model.repo_id,
            str(resolved_model.model_path),
            force_download,
            resolved_model.revision,
        )
        return resolved_model.model_path

    def _validate_runtime_plan(self, runtime_plan: RuntimePlan, config: RuntimeConfig) -> None:
        if config.offload_gpu_layers > 0 and not runtime_plan.supports_gpu_offload:
            raise ValueError(
                f"Selected backend '{runtime_plan.backend_id}' does not support GPU layer offload controls"
            )
        if config.offload_cpu_layers > 0 and not runtime_plan.supports_cpu_offload:
            raise ValueError(
                f"Selected backend '{runtime_plan.backend_id}' does not support CPU layer offload controls"
            )

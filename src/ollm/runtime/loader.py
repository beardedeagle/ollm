"""Runtime loading, planning, and safe fallback orchestration."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from ollm.inference import (
    download_hf_snapshot,
    hf_runtime_artifacts_complete,
    prune_hf_runtime_artifacts,
)
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.backend_selector import BackendSelector
from ollm.runtime.backends.base import BackendRuntime, ExecutionBackend
from ollm.runtime.backends.native_optimized import NativeOptimizedBackend
from ollm.runtime.backends.transformers_generic import TransformersGenericBackend
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan, SpecializationState
from ollm.runtime.resolver import ModelResolver, ModelSourceKind, ResolvedModel
from ollm.runtime.specialization import (
    SpecializationApplicationError,
    SpecializationLoadError,
    SpecializationRegistry,
    build_default_specialization_registry,
)


@dataclass(slots=True)
class LoadedRuntime:
    """Loaded runtime bundle containing the finalized backend and plan metadata."""

    resolved_model: ResolvedModel
    config: RuntimeConfig
    backend: BackendRuntime
    model_path: Path | None
    plan: RuntimePlan

    @property
    def capabilities(self) -> CapabilityProfile:
        """Return capability information aligned with the finalized runtime plan."""
        resolved_capabilities = self.resolved_model.capabilities
        if (
            resolved_capabilities.support_level is self.plan.support_level
            and resolved_capabilities.supports_disk_cache
            is self.plan.supports_disk_cache
        ):
            return resolved_capabilities
        details = dict(resolved_capabilities.details)
        details["runtime_plan_support_level"] = self.plan.support_level.value
        return CapabilityProfile(
            support_level=self.plan.support_level,
            modalities=resolved_capabilities.modalities,
            requires_processor=resolved_capabilities.requires_processor,
            supports_disk_cache=self.plan.supports_disk_cache,
            supports_local_materialization=resolved_capabilities.supports_local_materialization,
            supports_specialization=resolved_capabilities.supports_specialization,
            details=details,
        )

    @property
    def model(self):
        """Expose the backend-owned model object when one exists."""
        return self.backend.model

    @property
    def tokenizer(self):
        """Expose the backend-owned tokenizer when one exists."""
        return self.backend.tokenizer

    @property
    def processor(self):
        """Expose the backend-owned processor when one exists."""
        return self.backend.processor

    @property
    def device(self):
        """Expose the backend runtime device."""
        return self.backend.device


class RuntimeLoader:
    """Resolve, plan, materialize, and load runtimes for model references."""

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
        self._selector = selector or BackendSelector(
            specialization_registry=self._specialization_registry
        )
        backend_list = backends or (
            NativeOptimizedBackend(
                specialization_registry=self._specialization_registry
            ),
            TransformersGenericBackend(),
        )
        self._backends = {backend.backend_id: backend for backend in backend_list}
        self._snapshot_downloader = (
            download_hf_snapshot if snapshot_downloader is None else snapshot_downloader
        )

    def resolve(self, model_reference: str, models_dir: Path) -> ResolvedModel:
        """Resolve a model reference without planning or loading."""
        return self._resolver.resolve(model_reference, models_dir)

    def discover_local_models(self, models_dir: Path) -> tuple[ResolvedModel, ...]:
        """Discover local materialized models under a models directory."""
        return self._resolver.discover_local_models(models_dir)

    def download(
        self, model_reference: str, models_dir: Path, force_download: bool = False
    ) -> Path:
        """Materialize a downloadable model reference locally and return its path."""
        resolved_model = self.resolve(model_reference, models_dir)
        if resolved_model.source_kind is ModelSourceKind.LOCAL_PATH:
            if (
                resolved_model.model_path is None
                or not resolved_model.model_path.exists()
            ):
                raise ValueError(f"Local model path does not exist: {model_reference}")
            return resolved_model.model_path
        if (
            not resolved_model.is_downloadable()
            or resolved_model.repo_id is None
            or resolved_model.model_path is None
        ):
            raise ValueError(
                f"Model reference '{model_reference}' does not support snapshot download"
            )
        if resolved_model.model_path.exists() and not force_download:
            self._repair_managed_model_dir(resolved_model)
            if self._managed_model_dir_is_complete(resolved_model):
                return resolved_model.model_path
        resolved_model.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshot_downloader(
            resolved_model.repo_id,
            str(resolved_model.model_path),
            force_download,
            resolved_model.revision,
        )
        self._repair_managed_model_dir(resolved_model)
        self._validate_managed_model_dir(resolved_model)
        return resolved_model.model_path

    def load(self, config: RuntimeConfig) -> LoadedRuntime:
        """Validate, plan, and load a runtime backend for execution."""
        config.validate()
        resolved_model = self.resolve(
            config.model_reference, config.resolved_models_dir()
        )
        if (
            resolved_model.model_path is None
            and resolved_model.capabilities.support_level is SupportLevel.UNSUPPORTED
        ):
            raise ValueError(resolved_model.resolution_message)
        model_path = self._ensure_local_model(resolved_model, config.force_download)
        execution_model = self._refresh_materialized_model(resolved_model, model_path)

        runtime_plan = self._refine_runtime_plan(
            self._selector.select(execution_model, config), config
        )
        if not runtime_plan.is_executable():
            raise ValueError(runtime_plan.reason)
        self._validate_runtime_plan(runtime_plan, config)
        try:
            backend_runtime = self._load_backend_runtime(runtime_plan, config)
            runtime_plan = self._finalize_runtime_plan(runtime_plan, backend_runtime)
        except (SpecializationApplicationError, SpecializationLoadError) as exc:
            fallback_plan = self._build_generic_fallback_plan(
                execution_model=execution_model,
                runtime_plan=runtime_plan,
                config=config,
                error=exc,
            )
            if fallback_plan is None:
                raise ValueError(str(exc)) from exc
            backend_runtime = self._load_backend_runtime(fallback_plan, config)
            runtime_plan = self._finalize_runtime_plan(fallback_plan, backend_runtime)
        return LoadedRuntime(
            resolved_model=runtime_plan.resolved_model,
            config=config,
            backend=backend_runtime,
            model_path=runtime_plan.model_path,
            plan=runtime_plan,
        )

    def plan(self, config: RuntimeConfig) -> RuntimePlan:
        """Build a runtime plan without loading a backend."""
        config.validate()
        resolved_model = self.resolve(
            config.model_reference, config.resolved_models_dir()
        )
        if resolved_model.model_path is None or not resolved_model.model_path.exists():
            return self._refine_runtime_plan(
                self._selector.select(resolved_model, config), config
            )
        execution_model = self._refresh_materialized_model(
            resolved_model, resolved_model.model_path
        )
        return self._refine_runtime_plan(
            self._selector.select(execution_model, config), config
        )

    def _refresh_materialized_model(
        self, resolved_model: ResolvedModel, model_path: Path
    ) -> ResolvedModel:
        return self._resolver.inspect_materialized_model(
            resolved_model.reference,
            model_path,
            source_kind=resolved_model.source_kind,
            repo_id=resolved_model.repo_id,
            revision=resolved_model.revision,
            catalog_entry=resolved_model.catalog_entry,
        )

    def _ensure_local_model(
        self, resolved_model: ResolvedModel, force_download: bool
    ) -> Path:
        if resolved_model.model_path is None:
            raise ValueError(
                f"Model reference '{resolved_model.reference.raw}' does not resolve to a local model path"
            )
        if resolved_model.model_path.exists() and not force_download:
            self._repair_managed_model_dir(resolved_model)
            if self._managed_model_dir_is_complete(resolved_model):
                return resolved_model.model_path
        if resolved_model.repo_id is None:
            raise ValueError(
                f"Local model path does not exist: {resolved_model.model_path}"
            )
        resolved_model.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshot_downloader(
            resolved_model.repo_id,
            str(resolved_model.model_path),
            force_download,
            resolved_model.revision,
        )
        self._repair_managed_model_dir(resolved_model)
        self._validate_managed_model_dir(resolved_model)
        return resolved_model.model_path

    def _repair_managed_model_dir(self, resolved_model: ResolvedModel) -> None:
        model_path = resolved_model.model_path
        if (
            resolved_model.repo_id is None
            or model_path is None
            or not model_path.exists()
        ):
            return
        prune_hf_runtime_artifacts(model_path)

    def _managed_model_dir_is_complete(self, resolved_model: ResolvedModel) -> bool:
        model_path = resolved_model.model_path
        if resolved_model.repo_id is None or model_path is None:
            return True
        return hf_runtime_artifacts_complete(model_path)

    def _validate_managed_model_dir(self, resolved_model: ResolvedModel) -> None:
        if self._managed_model_dir_is_complete(resolved_model):
            return
        raise ValueError(
            f"Managed model directory is missing required runtime artifacts: {resolved_model.model_path}"
        )

    def _validate_runtime_plan(
        self, runtime_plan: RuntimePlan, config: RuntimeConfig
    ) -> None:
        if config.offload_gpu_layers > 0 and not runtime_plan.supports_gpu_offload:
            raise ValueError(
                f"Selected backend '{runtime_plan.backend_id}' does not support GPU layer offload controls"
            )
        if config.offload_cpu_layers > 0 and not runtime_plan.supports_cpu_offload:
            raise ValueError(
                f"Selected backend '{runtime_plan.backend_id}' does not support CPU layer offload controls"
            )

    def _load_backend_runtime(
        self, runtime_plan: RuntimePlan, config: RuntimeConfig
    ) -> BackendRuntime:
        backend_id = runtime_plan.backend_id
        if backend_id is None:
            raise ValueError("Runtime plan did not select a backend")
        backend_impl = self._backends.get(backend_id)
        if backend_impl is None:
            raise ValueError(f"No runtime backend is registered for '{backend_id}'")
        backend_runtime = backend_impl.load(runtime_plan, config)
        backend_runtime.apply_offload(config)
        return backend_runtime

    def _refine_runtime_plan(
        self, runtime_plan: RuntimePlan, config: RuntimeConfig
    ) -> RuntimePlan:
        if runtime_plan.backend_id is None:
            return runtime_plan
        backend_impl = self._backends.get(runtime_plan.backend_id)
        if backend_impl is None:
            return runtime_plan
        return backend_impl.refine_plan(runtime_plan, config)

    def _finalize_runtime_plan(
        self, runtime_plan: RuntimePlan, backend_runtime: BackendRuntime
    ) -> RuntimePlan:
        if backend_runtime.applied_specialization is None:
            return runtime_plan
        applied_specialization = backend_runtime.applied_specialization
        details = dict(runtime_plan.details)
        details["applied_specialization_pass_ids"] = ",".join(
            pass_id.value for pass_id in applied_specialization.applied_pass_ids
        )
        for key, value in applied_specialization.details.items():
            details[key] = value
        return replace(
            runtime_plan,
            specialization_applied=True,
            specialization_state=SpecializationState.APPLIED,
            applied_specialization_pass_ids=applied_specialization.applied_pass_ids,
            details=details,
        )

    def _build_generic_fallback_plan(
        self,
        execution_model: ResolvedModel,
        runtime_plan: RuntimePlan,
        config: RuntimeConfig,
        error: SpecializationApplicationError | SpecializationLoadError,
    ) -> RuntimePlan | None:
        if runtime_plan.backend_id != "optimized-native":
            return None
        if config.resolved_backend() is not None:
            return None
        if execution_model.generic_model_kind is None:
            return None
        if "transformers-generic" not in self._backends:
            return None
        if config.offload_cpu_layers > 0 or config.offload_gpu_layers > 0:
            return None

        details = dict(runtime_plan.details)
        details["fallback_from_backend_id"] = runtime_plan.backend_id or "unknown"
        details["fallback_error_type"] = type(error).__name__
        details["fallback_provider_id"] = (
            runtime_plan.specialization_provider_id
            or getattr(error, "provider_id", "unknown")
        )
        reason = (
            f"Fell back to transformers-generic for {execution_model.reference.raw} after optimized "
            f"specialization failed: {error}"
        )
        return RuntimePlan(
            resolved_model=execution_model,
            backend_id="transformers-generic",
            model_path=execution_model.model_path,
            support_level=SupportLevel.GENERIC,
            generic_model_kind=execution_model.generic_model_kind,
            supports_disk_cache=False,
            supports_cpu_offload=False,
            supports_gpu_offload=False,
            specialization_enabled=runtime_plan.specialization_enabled,
            specialization_applied=False,
            specialization_provider_id=runtime_plan.specialization_provider_id,
            specialization_state=SpecializationState.FALLBACK,
            reason=reason,
            specialization_pass_ids=runtime_plan.specialization_pass_ids,
            applied_specialization_pass_ids=(),
            fallback_reason=str(error),
            details=details,
        )

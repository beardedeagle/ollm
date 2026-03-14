from ollm.runtime.backends.base import BackendRuntime, ExecutionBackend
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.specialization import SpecializationRegistry, build_default_specialization_registry
from ollm.runtime.specialization.base import OptimizedModelArtifacts
from ollm.utils import Stats


class NativeOptimizedBackend(ExecutionBackend):
    backend_id = "optimized-native"

    def __init__(self, specialization_registry: SpecializationRegistry | None = None):
        self._specialization_registry = (
            build_default_specialization_registry()
            if specialization_registry is None
            else specialization_registry
        )

    def load(self, plan: RuntimePlan, config: RuntimeConfig) -> BackendRuntime:
        if plan.model_path is None or plan.specialization_provider_id is None:
            raise ValueError("optimized-native backend requires a selected specialization provider")

        stats = Stats() if config.stats or config.verbose else None
        artifacts = self._specialization_registry.load(
            plan.specialization_provider_id,
            plan.resolved_model,
            config,
            stats,
        )
        return BackendRuntime(
            backend_id=self.backend_id,
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            processor=artifacts.processor,
            device=artifacts.device,
            stats=artifacts.stats,
            create_cache=artifacts.create_cache,
            apply_offload=lambda runtime_config: _apply_native_offload(artifacts, runtime_config),
        )


def _apply_native_offload(artifacts: OptimizedModelArtifacts, config: RuntimeConfig) -> None:
    if config.offload_gpu_layers > 0:
        if artifacts.apply_gpu_offload is None:
            raise ValueError("The selected optimized specialization does not support GPU layer offload")
        artifacts.apply_gpu_offload(config.offload_gpu_layers, config.offload_cpu_layers)
        return
    if config.offload_cpu_layers > 0:
        if artifacts.apply_cpu_offload is None:
            raise ValueError("The selected optimized specialization does not support CPU layer offload")
        artifacts.apply_cpu_offload(config.offload_cpu_layers)

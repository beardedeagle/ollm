from ollm.runtime.backends.base import BackendRuntime, ExecutionBackend
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.output_control import suppress_module_prints
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.specialization import (
    PlannedSpecialization,
    SpecializationLoadError,
    SpecializationRegistry,
    apply_specialization,
    build_default_specialization_registry,
    get_specialization_pass,
)
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
        try:
            with suppress_module_prints(_modules_for_provider_id(plan.specialization_provider_id)):
                artifacts = self._specialization_registry.load(
                    plan.specialization_provider_id,
                    plan.resolved_model,
                    config,
                    stats,
                )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise SpecializationLoadError(
                (
                    f"Optimized specialization '{plan.specialization_provider_id}' failed to load for "
                    f"{plan.resolved_model.reference.raw}: {exc}"
                ),
                provider_id=plan.specialization_provider_id,
                details={"reason": str(exc)},
            ) from exc
        applied_specialization = apply_specialization(
            PlannedSpecialization(
                provider_id=plan.specialization_provider_id,
                passes=tuple(
                    get_specialization_pass(pass_id)
                    for pass_id in plan.specialization_pass_ids
                ),
                details=dict(plan.details),
            ),
            artifacts,
            config,
        )
        return BackendRuntime(
            backend_id=self.backend_id,
            model=artifacts.model,
            tokenizer=artifacts.tokenizer,
            processor=artifacts.processor,
            device=artifacts.device,
            stats=artifacts.stats,
            print_suppression_modules=artifacts.print_suppression_modules,
            create_cache=artifacts.create_cache,
            apply_offload=lambda runtime_config: _apply_native_offload(artifacts, runtime_config),
            applied_specialization=applied_specialization,
        )


def _apply_native_offload(artifacts: OptimizedModelArtifacts, config: RuntimeConfig) -> None:
    if config.offload_gpu_layers > 0:
        if artifacts.apply_gpu_offload is None:
            raise ValueError("The selected optimized specialization does not support GPU layer offload")
        with suppress_module_prints(artifacts.print_suppression_modules):
            artifacts.apply_gpu_offload(config.offload_gpu_layers, config.offload_cpu_layers)
        return
    if config.offload_cpu_layers > 0:
        if artifacts.apply_cpu_offload is None:
            raise ValueError("The selected optimized specialization does not support CPU layer offload")
        with suppress_module_prints(artifacts.print_suppression_modules):
            artifacts.apply_cpu_offload(config.offload_cpu_layers)


def _modules_for_provider_id(provider_id: str) -> tuple:
    if provider_id == "llama-native":
        from ollm import llama

        return (llama,)
    if provider_id == "gemma3-native":
        from ollm import gemma3

        return (gemma3,)
    if provider_id == "qwen3-next-native":
        from ollm import qwen3_next

        return (qwen3_next,)
    if provider_id == "gpt-oss-native":
        from ollm import gpt_oss

        return (gpt_oss,)
    if provider_id == "voxtral-native":
        from ollm import voxtral

        return (voxtral,)
    return ()

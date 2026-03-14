from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel
from ollm.runtime.specialization import SpecializationRegistry, build_default_specialization_registry


class BackendSelector:
    def __init__(self, specialization_registry: SpecializationRegistry | None = None):
        self._specialization_registry = (
            build_default_specialization_registry()
            if specialization_registry is None
            else specialization_registry
        )

    def select(self, resolved_model: ResolvedModel, config: RuntimeConfig) -> RuntimePlan:
        if resolved_model.source_kind is ModelSourceKind.PROVIDER:
            return RuntimePlan(
                resolved_model=resolved_model,
                backend_id=None,
                model_path=None,
                support_level=SupportLevel.PROVIDER_BACKED,
                generic_model_kind=resolved_model.generic_model_kind,
                supports_disk_cache=False,
                supports_cpu_offload=False,
                supports_gpu_offload=False,
                specialization_enabled=False,
                specialization_provider_id=None,
                reason=f"Provider-backed model references are not executable yet: {resolved_model.reference.raw}",
                details={"source_kind": resolved_model.source_kind.value},
            )

        if config.resolved_adapter_dir() is None:
            specialization_match = self._specialization_registry.select(resolved_model, config)
        else:
            specialization_match = None

        if specialization_match is not None:
            details = {
                "source_kind": resolved_model.source_kind.value,
                "specialization_provider_id": specialization_match.provider_id,
                "native_family": specialization_match.native_family.value,
            }
            for key, value in specialization_match.traits.details.items():
                details[key] = value
            return RuntimePlan(
                resolved_model=resolved_model,
                backend_id="optimized-native",
                model_path=resolved_model.model_path,
                support_level=SupportLevel.OPTIMIZED,
                generic_model_kind=resolved_model.generic_model_kind,
                supports_disk_cache=specialization_match.traits.supports_disk_cache,
                supports_cpu_offload=specialization_match.traits.supports_cpu_offload,
                supports_gpu_offload=specialization_match.traits.supports_gpu_offload,
                specialization_enabled=True,
                specialization_provider_id=specialization_match.provider_id,
                reason=specialization_match.reason,
                details=details,
            )

        if config.resolved_adapter_dir() is not None and resolved_model.generic_model_kind is not None:
            return RuntimePlan(
                resolved_model=resolved_model,
                backend_id="transformers-generic",
                model_path=resolved_model.model_path,
                support_level=SupportLevel.GENERIC,
                generic_model_kind=resolved_model.generic_model_kind,
                supports_disk_cache=False,
                supports_cpu_offload=False,
                supports_gpu_offload=False,
                specialization_enabled=False,
                specialization_provider_id=None,
                reason=(
                    f"Selected transformers-generic backend for {resolved_model.reference.raw} "
                    f"because PEFT adapters require the generic runtime path."
                ),
                details={
                    "source_kind": resolved_model.source_kind.value,
                    "generic_model_kind": resolved_model.generic_model_kind.value,
                },
            )

        if resolved_model.generic_model_kind is not None:
            return RuntimePlan(
                resolved_model=resolved_model,
                backend_id="transformers-generic",
                model_path=resolved_model.model_path,
                support_level=SupportLevel.GENERIC,
                generic_model_kind=resolved_model.generic_model_kind,
                supports_disk_cache=False,
                supports_cpu_offload=False,
                supports_gpu_offload=False,
                specialization_enabled=False,
                specialization_provider_id=None,
                reason=(
                    f"Selected transformers-generic backend for {resolved_model.reference.raw} "
                    f"using {resolved_model.generic_model_kind.value}."
                ),
                details={
                    "source_kind": resolved_model.source_kind.value,
                    "generic_model_kind": resolved_model.generic_model_kind.value,
                },
            )

        reason = resolved_model.capabilities.details.get("reason", resolved_model.resolution_message)
        return RuntimePlan(
            resolved_model=resolved_model,
            backend_id=None,
            model_path=resolved_model.model_path,
            support_level=resolved_model.capabilities.support_level,
            generic_model_kind=resolved_model.generic_model_kind,
            supports_disk_cache=False,
            supports_cpu_offload=False,
            supports_gpu_offload=False,
            specialization_enabled=False,
            specialization_provider_id=None,
            reason=reason,
            details={"source_kind": resolved_model.source_kind.value},
        )

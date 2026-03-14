from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel


class BackendSelector:
    def select(self, resolved_model: ResolvedModel, config: RuntimeConfig) -> RuntimePlan:
        if resolved_model.source_kind is ModelSourceKind.PROVIDER:
            return RuntimePlan(
                resolved_model=resolved_model,
                backend_id=None,
                model_path=None,
                support_level=SupportLevel.PROVIDER_BACKED,
                generic_model_kind=resolved_model.generic_model_kind,
                supports_disk_cache=False,
                supports_offload=False,
                specialization_enabled=False,
                reason=f"Provider-backed model references are not executable yet: {resolved_model.reference.raw}",
                details={"source_kind": resolved_model.source_kind.value},
            )

        if (
            resolved_model.catalog_entry is not None
            and resolved_model.reference.scheme in {None, "hf"}
            and config.resolved_adapter_dir() is None
        ):
            return RuntimePlan(
                resolved_model=resolved_model,
                backend_id="optimized-native",
                model_path=resolved_model.model_path,
                support_level=SupportLevel.OPTIMIZED,
                generic_model_kind=resolved_model.generic_model_kind,
                supports_disk_cache=resolved_model.catalog_entry.supports_disk_cache,
                supports_offload=True,
                specialization_enabled=True,
                reason=f"Selected optimized-native backend for built-in alias '{resolved_model.catalog_entry.model_id}'.",
                details={
                    "source_kind": resolved_model.source_kind.value,
                    "catalog_model_id": resolved_model.catalog_entry.model_id,
                },
            )

        if config.resolved_adapter_dir() is not None and resolved_model.generic_model_kind is not None:
            return RuntimePlan(
                resolved_model=resolved_model,
                backend_id="transformers-generic",
                model_path=resolved_model.model_path,
                support_level=SupportLevel.GENERIC,
                generic_model_kind=resolved_model.generic_model_kind,
                supports_disk_cache=False,
                supports_offload=False,
                specialization_enabled=False,
                reason=(
                    f"Selected transformers-generic backend for {resolved_model.reference.raw} "
                    f"because PEFT adapters require the generic runtime path."
                ),
                details={
                    "source_kind": resolved_model.source_kind.value,
                    "generic_model_kind": resolved_model.generic_model_kind.value,
                },
            )

        if resolved_model.generic_model_kind is not None and (
            resolved_model.capabilities.support_level is SupportLevel.GENERIC
            or config.resolved_adapter_dir() is not None
        ):
            return RuntimePlan(
                resolved_model=resolved_model,
                backend_id="transformers-generic",
                model_path=resolved_model.model_path,
                support_level=SupportLevel.GENERIC,
                generic_model_kind=resolved_model.generic_model_kind,
                supports_disk_cache=False,
                supports_offload=False,
                specialization_enabled=False,
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
            supports_offload=False,
            specialization_enabled=False,
            reason=reason,
            details={"source_kind": resolved_model.source_kind.value},
        )

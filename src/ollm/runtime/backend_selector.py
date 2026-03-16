from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan, SpecializationState
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel
from ollm.runtime.specialization import (
    SpecializationPipeline,
    SpecializationRegistry,
    build_default_specialization_registry,
)


class BackendSelector:
    def __init__(
        self,
        specialization_registry: SpecializationRegistry | None = None,
        specialization_pipeline: SpecializationPipeline | None = None,
    ):
        self._specialization_registry = (
            build_default_specialization_registry()
            if specialization_registry is None
            else specialization_registry
        )
        self._specialization_pipeline = (
            SpecializationPipeline()
            if specialization_pipeline is None
            else specialization_pipeline
        )

    def select(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> RuntimePlan:
        backend_override = config.resolved_backend()
        if backend_override is not None:
            return self._select_explicit_backend(
                resolved_model, config, backend_override
            )

        if resolved_model.source_kind is ModelSourceKind.PROVIDER:
            return self._select_provider_backend(resolved_model)

        if config.use_specialization:
            specialization_plan = self._planned_specialization(resolved_model, config)
            if specialization_plan is not None:
                return specialization_plan

        generic_reason = (
            f"Selected transformers-generic backend for {resolved_model.reference.raw} "
            "because specialization was disabled."
            if not config.use_specialization
            else None
        )
        return self._select_generic_backend(resolved_model, generic_reason)

    def _select_explicit_backend(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        backend_override: str,
    ) -> RuntimePlan:
        if backend_override == "optimized-native":
            if resolved_model.source_kind is ModelSourceKind.PROVIDER:
                return self._override_failure(
                    resolved_model,
                    backend_override,
                    f"Backend override '{backend_override}' does not support provider-backed references.",
                )
            specialization_plan = self._planned_specialization(resolved_model, config)
            if specialization_plan is not None:
                return self._with_override_detail(specialization_plan, backend_override)
            return self._override_failure(
                resolved_model,
                backend_override,
                f"Backend override '{backend_override}' is not available for {resolved_model.reference.raw}.",
            )

        if backend_override == "transformers-generic":
            if resolved_model.source_kind is ModelSourceKind.PROVIDER:
                return self._override_failure(
                    resolved_model,
                    backend_override,
                    f"Backend override '{backend_override}' does not support provider-backed references.",
                )
            return self._with_override_detail(
                self._select_generic_backend(
                    resolved_model,
                    (
                        f"Selected transformers-generic backend for {resolved_model.reference.raw} "
                        "because it was requested explicitly."
                    ),
                    explicit_override=backend_override,
                ),
                backend_override,
            )

        if backend_override == "ollama":
            if resolved_model.provider_name in {"ollama", "msty"}:
                return self._with_override_detail(
                    self._provider_runtime_plan(
                        resolved_model,
                        backend_id="ollama",
                        provider_name=resolved_model.provider_name or "ollama",
                    ),
                    backend_override,
                )
            return self._override_failure(
                resolved_model,
                backend_override,
                f"Backend override '{backend_override}' only supports ollama: or msty: references.",
            )

        if backend_override == "openai-compatible":
            if resolved_model.provider_name in {"openai-compatible", "lmstudio"}:
                return self._with_override_detail(
                    self._provider_runtime_plan(
                        resolved_model,
                        backend_id="openai-compatible",
                        provider_name=resolved_model.provider_name
                        or "openai-compatible",
                    ),
                    backend_override,
                )
            return self._override_failure(
                resolved_model,
                backend_override,
                (
                    f"Backend override '{backend_override}' only supports "
                    "openai-compatible: or lmstudio: references."
                ),
            )

        return self._override_failure(
            resolved_model,
            backend_override,
            f"Backend override '{backend_override}' is not registered.",
        )

    def _select_provider_backend(self, resolved_model: ResolvedModel) -> RuntimePlan:
        provider_name = resolved_model.provider_name or "provider"
        if provider_name in {"ollama", "msty"}:
            return self._provider_runtime_plan(
                resolved_model,
                backend_id="ollama",
                provider_name=provider_name,
            )
        if provider_name in {"openai-compatible", "lmstudio"}:
            return self._provider_runtime_plan(
                resolved_model,
                backend_id="openai-compatible",
                provider_name=provider_name,
            )
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
            specialization_applied=False,
            specialization_provider_id=None,
            specialization_state=SpecializationState.NOT_PLANNED,
            specialization_pass_ids=(),
            reason=(
                f"Provider-backed model references are not executable yet for provider "
                f"'{resolved_model.provider_name}': {resolved_model.reference.raw}"
            ),
            details={
                "source_kind": resolved_model.source_kind.value,
                "provider_name": ""
                if resolved_model.provider_name is None
                else resolved_model.provider_name,
            },
        )

    def _provider_runtime_plan(
        self,
        resolved_model: ResolvedModel,
        *,
        backend_id: str,
        provider_name: str,
    ) -> RuntimePlan:
        return RuntimePlan(
            resolved_model=resolved_model,
            backend_id=backend_id,
            model_path=None,
            support_level=SupportLevel.PROVIDER_BACKED,
            generic_model_kind=resolved_model.generic_model_kind,
            supports_disk_cache=False,
            supports_cpu_offload=False,
            supports_gpu_offload=False,
            specialization_enabled=False,
            specialization_applied=False,
            specialization_provider_id=None,
            specialization_state=SpecializationState.NOT_PLANNED,
            specialization_pass_ids=(),
            reason=f"{provider_name} provider-backed model reference for {resolved_model.reference.raw}.",
            details={
                "source_kind": resolved_model.source_kind.value,
                "provider_name": provider_name,
            },
        )

    def _planned_specialization(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
    ) -> RuntimePlan | None:
        if resolved_model.source_kind is ModelSourceKind.PROVIDER:
            return None
        if config.resolved_adapter_dir() is not None:
            return None
        specialization_match = self._specialization_registry.select(
            resolved_model, config
        )
        if specialization_match is None:
            return None

        planned_specialization = self._specialization_pipeline.plan(
            resolved_model,
            config,
            specialization_match.provider_id,
        )
        details = {
            "source_kind": resolved_model.source_kind.value,
            "specialization_provider_id": specialization_match.provider_id,
            "native_family": specialization_match.native_family.value,
            "specialization_pass_ids": ",".join(
                pass_id.value for pass_id in planned_specialization.pass_ids
            ),
        }
        for key, value in specialization_match.traits.details.items():
            details[key] = value
        for key, value in planned_specialization.details.items():
            details[key] = value
        return RuntimePlan(
            resolved_model=resolved_model,
            backend_id="optimized-native",
            model_path=resolved_model.model_path,
            support_level=SupportLevel.OPTIMIZED,
            generic_model_kind=resolved_model.generic_model_kind,
            supports_disk_cache=planned_specialization.traits.supports_disk_cache,
            supports_cpu_offload=planned_specialization.traits.supports_cpu_offload,
            supports_gpu_offload=planned_specialization.traits.supports_gpu_offload,
            specialization_enabled=True,
            specialization_applied=False,
            specialization_provider_id=specialization_match.provider_id,
            specialization_state=SpecializationState.PLANNED,
            specialization_pass_ids=planned_specialization.pass_ids,
            reason=specialization_match.reason,
            details=details,
        )

    def _select_generic_backend(
        self,
        resolved_model: ResolvedModel,
        reason: str | None = None,
        *,
        explicit_override: str | None = None,
    ) -> RuntimePlan:
        if self._supports_generic_backend(resolved_model):
            generic_reason = reason or (
                f"Selected transformers-generic backend for {resolved_model.reference.raw} "
                f"using {resolved_model.generic_model_kind.value}."
                if resolved_model.generic_model_kind is not None
                else (
                    f"Selected transformers-generic backend for {resolved_model.reference.raw}. "
                    "The exact generic model kind will be confirmed after materialization."
                )
            )
            details = {"source_kind": resolved_model.source_kind.value}
            if resolved_model.generic_model_kind is not None:
                details["generic_model_kind"] = resolved_model.generic_model_kind.value
            if explicit_override is not None:
                details["backend_override"] = explicit_override
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
                specialization_applied=False,
                specialization_provider_id=None,
                specialization_state=SpecializationState.NOT_PLANNED,
                specialization_pass_ids=(),
                reason=generic_reason,
                details=details,
            )

        if config_reason := reason:
            unsupported_reason = config_reason
        elif (
            config_reason is None
            and resolved_model.source_kind is ModelSourceKind.BUILTIN
        ):
            unsupported_reason = (
                f"transformers-generic is unavailable for {resolved_model.reference.raw} "
                "until the model is materialized and inspected."
            )
        else:
            unsupported_reason = resolved_model.capabilities.details.get(
                "reason", resolved_model.resolution_message
            )

        details = {"source_kind": resolved_model.source_kind.value}
        if explicit_override is not None:
            details["backend_override"] = explicit_override
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
            specialization_applied=False,
            specialization_provider_id=None,
            specialization_state=SpecializationState.NOT_PLANNED,
            specialization_pass_ids=(),
            reason=unsupported_reason,
            details=details,
        )

    def _supports_generic_backend(self, resolved_model: ResolvedModel) -> bool:
        if resolved_model.source_kind is ModelSourceKind.PROVIDER:
            return False
        if resolved_model.generic_model_kind is not None:
            return True
        return resolved_model.is_downloadable()

    def _override_failure(
        self,
        resolved_model: ResolvedModel,
        backend_override: str,
        reason: str,
    ) -> RuntimePlan:
        details = {
            "source_kind": resolved_model.source_kind.value,
            "backend_override": backend_override,
        }
        if resolved_model.provider_name is not None:
            details["provider_name"] = resolved_model.provider_name
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
            specialization_applied=False,
            specialization_provider_id=None,
            specialization_state=SpecializationState.NOT_PLANNED,
            specialization_pass_ids=(),
            reason=reason,
            details=details,
        )

    def _with_override_detail(
        self, runtime_plan: RuntimePlan, backend_override: str
    ) -> RuntimePlan:
        details = dict(runtime_plan.details)
        details["backend_override"] = backend_override
        return RuntimePlan(
            resolved_model=runtime_plan.resolved_model,
            backend_id=runtime_plan.backend_id,
            model_path=runtime_plan.model_path,
            support_level=runtime_plan.support_level,
            generic_model_kind=runtime_plan.generic_model_kind,
            supports_disk_cache=runtime_plan.supports_disk_cache,
            supports_cpu_offload=runtime_plan.supports_cpu_offload,
            supports_gpu_offload=runtime_plan.supports_gpu_offload,
            specialization_enabled=runtime_plan.specialization_enabled,
            specialization_applied=runtime_plan.specialization_applied,
            specialization_provider_id=runtime_plan.specialization_provider_id,
            specialization_state=runtime_plan.specialization_state,
            reason=runtime_plan.reason,
            specialization_pass_ids=runtime_plan.specialization_pass_ids,
            applied_specialization_pass_ids=runtime_plan.applied_specialization_pass_ids,
            fallback_reason=runtime_plan.fallback_reason,
            details=details,
        )

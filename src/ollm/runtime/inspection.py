from typing import TypedDict, cast

from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel


class ResolvedModelPayload(TypedDict):
    model_reference: str
    normalized_name: str
    source_kind: str
    support_level: str
    modalities: list[str]
    requires_processor: bool
    supports_disk_cache: bool
    supports_specialization: bool
    repo_id: str | None
    revision: str | None
    path: str | None
    provider_name: str | None
    native_family: str | None
    architecture: str | None
    model_type: str | None
    generic_model_kind: str | None
    resolution_message: str


class RuntimePlanPayload(TypedDict):
    backend_id: str | None
    modalities: list[str]
    requires_processor: bool
    audio_input_support: str
    supports_disk_cache: bool
    supports_cpu_offload: bool
    supports_gpu_offload: bool
    specialization_provider_id: str | None
    specialization_enabled: bool
    specialization_state: str
    planned_specialization_pass_ids: list[str]
    reason: str


class AvailabilityPayload(TypedDict):
    materialized: bool
    available: bool | None
    availability_status: str


class MergedRuntimePayload(TypedDict):
    model_reference: str
    normalized_name: str
    source_kind: str
    support_level: str
    modalities: list[str]
    requires_processor: bool
    supports_disk_cache: bool
    supports_specialization: bool
    repo_id: str | None
    revision: str | None
    path: str | None
    provider_name: str | None
    native_family: str | None
    architecture: str | None
    model_type: str | None
    generic_model_kind: str | None
    resolution_message: str
    materialized: bool
    available: bool | None
    availability_status: str
    resolved_support_level: str
    resolved_modalities: list[str]
    resolved_requires_processor: bool
    resolved_supports_disk_cache: bool
    resolved_resolution_message: str
    runtime_plan: RuntimePlanPayload


class RuntimeConfigPayload(TypedDict):
    model_reference: str
    models_dir: str
    device: str
    backend: str | None
    provider_endpoint: str | None
    adapter_dir: str | None
    multimodal: bool
    use_specialization: bool
    cache_dir: str
    use_cache: bool
    offload_cpu_layers: int
    offload_gpu_layers: int
    force_download: bool
    stats: bool
    verbose: bool
    quiet: bool


class PlanJsonPayload(TypedDict):
    runtime_config: RuntimeConfigPayload
    resolved_model: ResolvedModelPayload
    runtime_plan: RuntimePlanPayload


def resolved_model_payload(resolved_model: ResolvedModel) -> ResolvedModelPayload:
    return {
        "model_reference": resolved_model.reference.raw,
        "normalized_name": resolved_model.normalized_name,
        "source_kind": resolved_model.source_kind.value,
        "support_level": resolved_model.capabilities.support_level.value,
        "modalities": [modality.value for modality in resolved_model.capabilities.modalities],
        "requires_processor": resolved_model.capabilities.requires_processor,
        "supports_disk_cache": resolved_model.capabilities.supports_disk_cache,
        "supports_specialization": resolved_model.capabilities.supports_specialization,
        "repo_id": resolved_model.repo_id,
        "revision": resolved_model.revision,
        "path": None if resolved_model.model_path is None else str(resolved_model.model_path),
        "provider_name": resolved_model.provider_name,
        "native_family": None if resolved_model.native_family is None else resolved_model.native_family.value,
        "architecture": resolved_model.architecture,
        "model_type": resolved_model.model_type,
        "generic_model_kind": None if resolved_model.generic_model_kind is None else resolved_model.generic_model_kind.value,
        "resolution_message": resolved_model.resolution_message,
    }


def runtime_plan_payload(runtime_plan: RuntimePlan) -> RuntimePlanPayload:
    return {
        "backend_id": runtime_plan.backend_id,
        "modalities": [
            modality.value for modality in runtime_plan.resolved_model.capabilities.modalities
        ],
        "requires_processor": runtime_plan.resolved_model.capabilities.requires_processor,
        "audio_input_support": runtime_plan.details.get("audio_input_support", ""),
        "supports_disk_cache": runtime_plan.supports_disk_cache,
        "supports_cpu_offload": runtime_plan.supports_cpu_offload,
        "supports_gpu_offload": runtime_plan.supports_gpu_offload,
        "specialization_provider_id": runtime_plan.specialization_provider_id,
        "specialization_enabled": runtime_plan.specialization_enabled,
        "specialization_state": runtime_plan.specialization_state.value,
        "planned_specialization_pass_ids": [
            pass_id.value for pass_id in runtime_plan.specialization_pass_ids
        ],
        "reason": runtime_plan.reason,
    }


def availability_payload(
    resolved_model: ResolvedModel,
    runtime_plan: RuntimePlan,
    *,
    materialized: bool,
) -> AvailabilityPayload:
    if resolved_model.source_kind is ModelSourceKind.PROVIDER:
        available = runtime_plan.is_executable()
        return {
            "materialized": False,
            "available": available,
            "availability_status": "available" if available else "unavailable",
        }
    return {
        "materialized": materialized,
        "available": None,
        "availability_status": "materialized" if materialized else "not-materialized",
    }


def merged_runtime_payload(
    resolved_model: ResolvedModel,
    runtime_plan: RuntimePlan,
    *,
    materialized: bool,
) -> MergedRuntimePayload:
    payload = cast(MergedRuntimePayload, resolved_model_payload(resolved_model))
    payload.update(
        availability_payload(
            resolved_model,
            runtime_plan,
            materialized=materialized,
        )
    )
    payload["resolved_support_level"] = payload["support_level"]
    payload["resolved_modalities"] = list(payload["modalities"])
    payload["resolved_requires_processor"] = payload["requires_processor"]
    payload["resolved_supports_disk_cache"] = payload["supports_disk_cache"]
    payload["resolved_resolution_message"] = payload["resolution_message"]
    payload["support_level"] = runtime_plan.support_level.value
    payload["modalities"] = [
        modality.value for modality in runtime_plan.resolved_model.capabilities.modalities
    ]
    payload["requires_processor"] = runtime_plan.resolved_model.capabilities.requires_processor
    payload["supports_disk_cache"] = runtime_plan.supports_disk_cache
    payload["resolution_message"] = runtime_plan.reason
    payload["runtime_plan"] = runtime_plan_payload(runtime_plan)
    return payload


def runtime_config_payload(runtime_config: RuntimeConfig) -> RuntimeConfigPayload:
    return {
        "model_reference": runtime_config.model_reference,
        "models_dir": str(runtime_config.resolved_models_dir()),
        "device": runtime_config.device,
        "backend": runtime_config.resolved_backend(),
        "provider_endpoint": runtime_config.resolved_provider_endpoint(),
        "adapter_dir": None if runtime_config.resolved_adapter_dir() is None else str(runtime_config.resolved_adapter_dir()),
        "multimodal": runtime_config.multimodal,
        "use_specialization": runtime_config.use_specialization,
        "cache_dir": str(runtime_config.resolved_cache_dir()),
        "use_cache": runtime_config.use_cache,
        "offload_cpu_layers": runtime_config.offload_cpu_layers,
        "offload_gpu_layers": runtime_config.offload_gpu_layers,
        "force_download": runtime_config.force_download,
        "stats": runtime_config.stats,
        "verbose": runtime_config.verbose,
        "quiet": runtime_config.quiet,
    }


def plan_json_payload(runtime_config: RuntimeConfig, runtime_plan: RuntimePlan) -> PlanJsonPayload:
    return {
        "runtime_config": runtime_config_payload(runtime_config),
        "resolved_model": resolved_model_payload(runtime_plan.resolved_model),
        "runtime_plan": runtime_plan_payload(runtime_plan),
    }

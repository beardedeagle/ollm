from ollm.runtime.config import RuntimeConfig
from ollm.runtime.specialization.base import (
    AppliedSpecialization,
    OptimizedModelArtifacts,
    PlannedSpecialization,
    SpecializationApplicationError,
)
from ollm.runtime.specialization.passes.base import SpecializationPass, SpecializationPassId


def apply_specialization(
    planned_specialization: PlannedSpecialization,
    artifacts: OptimizedModelArtifacts,
    config: RuntimeConfig,
) -> AppliedSpecialization:
    applied_passes: list[SpecializationPass] = []
    for specialization_pass in planned_specialization.passes:
        _validate_specialization_pass(
            specialization_pass=specialization_pass,
            artifacts=artifacts,
            config=config,
            provider_id=planned_specialization.provider_id,
            planned_specialization=planned_specialization,
            applied_passes=tuple(applied_passes),
        )
        applied_passes.append(specialization_pass)

    details = {
        "planned_pass_count": str(len(planned_specialization.passes)),
        "applied_pass_count": str(len(applied_passes)),
    }
    for key, value in planned_specialization.details.items():
        details[key] = value
    return AppliedSpecialization(
        provider_id=planned_specialization.provider_id,
        planned_passes=planned_specialization.passes,
        applied_passes=tuple(applied_passes),
        skipped_passes=(),
        details=details,
    )


def _validate_specialization_pass(
    specialization_pass: SpecializationPass,
    artifacts: OptimizedModelArtifacts,
    config: RuntimeConfig,
    provider_id: str | None,
    planned_specialization: PlannedSpecialization,
    applied_passes: tuple[SpecializationPass, ...],
) -> None:
    if specialization_pass.pass_id is SpecializationPassId.DISK_CACHE:
        if artifacts.supports_disk_cache:
            return
        raise _specialization_application_error(
            provider_id=provider_id,
            planned_specialization=planned_specialization,
            applied_passes=applied_passes,
            failed_pass_id=specialization_pass.pass_id,
            reason="optimized artifacts do not expose disk-cache support",
        )
    if specialization_pass.pass_id is SpecializationPassId.CPU_OFFLOAD:
        if artifacts.supports_cpu_offload and artifacts.apply_cpu_offload is not None:
            return
        raise _specialization_application_error(
            provider_id=provider_id,
            planned_specialization=planned_specialization,
            applied_passes=applied_passes,
            failed_pass_id=specialization_pass.pass_id,
            reason="optimized artifacts do not expose CPU offload support",
        )
    if specialization_pass.pass_id is SpecializationPassId.GPU_OFFLOAD:
        if artifacts.supports_gpu_offload and artifacts.apply_gpu_offload is not None:
            return
        raise _specialization_application_error(
            provider_id=provider_id,
            planned_specialization=planned_specialization,
            applied_passes=applied_passes,
            failed_pass_id=specialization_pass.pass_id,
            reason="optimized artifacts do not expose GPU offload support",
        )
    if specialization_pass.pass_id is SpecializationPassId.MULTIMODAL_SHELL:
        if config.multimodal and artifacts.processor is not None:
            return
        raise _specialization_application_error(
            provider_id=provider_id,
            planned_specialization=planned_specialization,
            applied_passes=applied_passes,
            failed_pass_id=specialization_pass.pass_id,
            reason="optimized artifacts do not expose a processor-backed multimodal shell",
        )
    if specialization_pass.pass_id in artifacts.provided_pass_ids:
        return
    raise _specialization_application_error(
        provider_id=provider_id,
        planned_specialization=planned_specialization,
        applied_passes=applied_passes,
        failed_pass_id=specialization_pass.pass_id,
        reason="optimized artifacts do not explicitly declare the planned specialization pass",
    )


def _specialization_application_error(
    provider_id: str | None,
    planned_specialization: PlannedSpecialization,
    applied_passes: tuple[SpecializationPass, ...],
    failed_pass_id: SpecializationPassId,
    reason: str,
) -> SpecializationApplicationError:
    resolved_provider_id = "unknown" if provider_id is None else provider_id
    return SpecializationApplicationError(
        (
            f"Optimized specialization '{resolved_provider_id}' could not apply "
            f"'{failed_pass_id.value}': {reason}"
        ),
        provider_id=resolved_provider_id,
        planned_pass_ids=planned_specialization.pass_ids,
        applied_pass_ids=tuple(specialization_pass.pass_id for specialization_pass in applied_passes),
        failed_pass_id=failed_pass_id,
        details={"reason": reason},
    )

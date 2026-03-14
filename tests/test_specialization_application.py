from types import ModuleType

import pytest
import torch

from ollm.runtime.config import RuntimeConfig
from ollm.runtime.specialization import apply_specialization
from ollm.runtime.specialization.base import (
    OptimizedModelArtifacts,
    PlannedSpecialization,
    SpecializationApplicationError,
)
from ollm.runtime.specialization.passes.base import (
    SpecializationPass,
    SpecializationPassId,
    SpecializationPassTraits,
)


def _build_artifacts(
    *,
    processor: object | None = None,
    supports_disk_cache: bool = True,
    supports_cpu_offload: bool = True,
    supports_gpu_offload: bool = False,
    provided_pass_ids: tuple[SpecializationPassId, ...] = (),
) -> OptimizedModelArtifacts:
    module = ModuleType("test_specialization_module")
    return OptimizedModelArtifacts(
        model=object(),
        tokenizer=object(),
        processor=processor,
        device=torch.device("cpu"),
        stats=None,
        supports_disk_cache=supports_disk_cache,
        supports_cpu_offload=supports_cpu_offload,
        supports_gpu_offload=supports_gpu_offload,
        print_suppression_modules=(module,),
        create_cache=lambda cache_dir: str(cache_dir),
        apply_cpu_offload=(lambda layers_num: None) if supports_cpu_offload else None,
        apply_gpu_offload=(lambda gpu_layers_num, cpu_layers_num: None) if supports_gpu_offload else None,
        provided_pass_ids=provided_pass_ids,
    )


def test_apply_specialization_marks_planned_passes_as_applied() -> None:
    planned_specialization = PlannedSpecialization(
        provider_id="llama-native",
        passes=(
            SpecializationPass(
                pass_id=SpecializationPassId.DISK_CACHE,
                summary="disk cache",
                traits=SpecializationPassTraits(supports_disk_cache=True),
            ),
            SpecializationPass(
                pass_id=SpecializationPassId.CPU_OFFLOAD,
                summary="cpu offload",
                traits=SpecializationPassTraits(supports_cpu_offload=True),
            ),
        ),
    )

    applied_specialization = apply_specialization(
        planned_specialization,
        _build_artifacts(),
        RuntimeConfig(device="cpu"),
    )

    assert applied_specialization.provider_id == "llama-native"
    assert applied_specialization.applied_pass_ids == (
        SpecializationPassId.DISK_CACHE,
        SpecializationPassId.CPU_OFFLOAD,
    )
    assert applied_specialization.traits.supports_disk_cache is True
    assert applied_specialization.traits.supports_cpu_offload is True


def test_apply_specialization_raises_when_gpu_offload_is_unavailable() -> None:
    planned_specialization = PlannedSpecialization(
        provider_id="qwen3-next-native",
        passes=(
            SpecializationPass(
                pass_id=SpecializationPassId.GPU_OFFLOAD,
                summary="gpu offload",
                traits=SpecializationPassTraits(supports_gpu_offload=True),
            ),
        ),
    )

    with pytest.raises(SpecializationApplicationError) as exc_info:
        apply_specialization(
            planned_specialization,
            _build_artifacts(supports_gpu_offload=False),
            RuntimeConfig(device="cpu"),
        )

    assert exc_info.value.failed_pass_id is SpecializationPassId.GPU_OFFLOAD
    assert exc_info.value.applied_pass_ids == ()


def test_apply_specialization_requires_processor_for_multimodal_shell() -> None:
    planned_specialization = PlannedSpecialization(
        provider_id="gemma3-native",
        passes=(
            SpecializationPass(
                pass_id=SpecializationPassId.MULTIMODAL_SHELL,
                summary="multimodal shell",
            ),
        ),
    )

    with pytest.raises(SpecializationApplicationError) as exc_info:
        apply_specialization(
            planned_specialization,
            _build_artifacts(processor=None),
            RuntimeConfig(device="cpu", multimodal=True),
        )

    assert exc_info.value.failed_pass_id is SpecializationPassId.MULTIMODAL_SHELL


def test_apply_specialization_requires_declared_internal_provider_passes() -> None:
    planned_specialization = PlannedSpecialization(
        provider_id="llama-native",
        passes=(
            SpecializationPass(
                pass_id=SpecializationPassId.MLP_CHUNKING,
                summary="mlp chunking",
            ),
        ),
    )

    with pytest.raises(SpecializationApplicationError) as exc_info:
        apply_specialization(
            planned_specialization,
            _build_artifacts(provided_pass_ids=()),
            RuntimeConfig(device="cpu"),
        )

    assert exc_info.value.failed_pass_id is SpecializationPassId.MLP_CHUNKING

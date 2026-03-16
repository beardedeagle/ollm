from ollm.runtime.specialization.passes.base import (
    SpecializationPass,
    SpecializationPassId,
    SpecializationPassTraits,
)


SPECIALIZATION_PASS_CATALOG: dict[SpecializationPassId, SpecializationPass] = {
    SpecializationPassId.DISK_CACHE: SpecializationPass(
        pass_id=SpecializationPassId.DISK_CACHE,
        summary="Enables the native disk-backed KV cache path for compatible optimized runtimes.",
        traits=SpecializationPassTraits(supports_disk_cache=True),
    ),
    SpecializationPassId.CPU_OFFLOAD: SpecializationPass(
        pass_id=SpecializationPassId.CPU_OFFLOAD,
        summary="Enables native CPU layer offload controls for optimized runtimes.",
        traits=SpecializationPassTraits(supports_cpu_offload=True),
    ),
    SpecializationPassId.GPU_OFFLOAD: SpecializationPass(
        pass_id=SpecializationPassId.GPU_OFFLOAD,
        summary="Enables native GPU and hybrid GPU/CPU layer offload controls for optimized runtimes.",
        traits=SpecializationPassTraits(supports_gpu_offload=True),
    ),
    SpecializationPassId.MLP_CHUNKING: SpecializationPass(
        pass_id=SpecializationPassId.MLP_CHUNKING,
        summary="Uses chunked MLP execution for optimized decoder-layer implementations.",
    ),
    SpecializationPassId.MOE_ROUTING: SpecializationPass(
        pass_id=SpecializationPassId.MOE_ROUTING,
        summary="Uses native mixture-of-experts routing and weight-loading behavior.",
    ),
    SpecializationPassId.ATTENTION_REPLACEMENT: SpecializationPass(
        pass_id=SpecializationPassId.ATTENTION_REPLACEMENT,
        summary="Uses a specialized attention implementation for the optimized runtime path.",
    ),
    SpecializationPassId.MULTIMODAL_SHELL: SpecializationPass(
        pass_id=SpecializationPassId.MULTIMODAL_SHELL,
        summary="Uses optimized multimodal outer-shell normalization with processor-backed inputs.",
    ),
    SpecializationPassId.GDS_EXPORT_WEIGHTS: SpecializationPass(
        pass_id=SpecializationPassId.GDS_EXPORT_WEIGHTS,
        summary="Uses validated GDS-exported weights for the optimized runtime path.",
    ),
}


def get_specialization_pass(pass_id: SpecializationPassId) -> SpecializationPass:
    return SPECIALIZATION_PASS_CATALOG[pass_id]

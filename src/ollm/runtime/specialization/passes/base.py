from dataclasses import dataclass, field
from enum import Enum
from typing import Self, cast


class SpecializationPassId(str, Enum):
    DISK_CACHE = "disk-cache"
    CPU_OFFLOAD = "cpu-offload"
    GPU_OFFLOAD = "gpu-offload"
    MLP_CHUNKING = "mlp-chunking"
    MOE_ROUTING = "moe-routing"
    ATTENTION_REPLACEMENT = "attention-replacement"
    MULTIMODAL_SHELL = "multimodal-shell"
    GDS_EXPORT_WEIGHTS = "gds-export-weights"


@dataclass(frozen=True, slots=True)
class SpecializationPassTraits:
    supports_disk_cache: bool = False
    supports_cpu_offload: bool = False
    supports_gpu_offload: bool = False

    def merge(self, other: Self) -> Self:
        return cast(
            Self,
            SpecializationPassTraits(
                supports_disk_cache=self.supports_disk_cache or other.supports_disk_cache,
                supports_cpu_offload=self.supports_cpu_offload or other.supports_cpu_offload,
                supports_gpu_offload=self.supports_gpu_offload or other.supports_gpu_offload,
            ),
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "supports_disk_cache": self.supports_disk_cache,
            "supports_cpu_offload": self.supports_cpu_offload,
            "supports_gpu_offload": self.supports_gpu_offload,
        }


@dataclass(frozen=True, slots=True)
class SpecializationPass:
    pass_id: SpecializationPassId
    summary: str
    traits: SpecializationPassTraits = field(default_factory=SpecializationPassTraits)
    details: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "pass_id": self.pass_id.value,
            "summary": self.summary,
            "traits": self.traits.as_dict(),
            "details": dict(self.details),
        }

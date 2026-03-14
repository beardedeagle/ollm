from dataclasses import dataclass, field
from pathlib import Path

from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.resolver import ResolvedModel


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    resolved_model: ResolvedModel
    backend_id: str | None
    model_path: Path | None
    support_level: SupportLevel
    generic_model_kind: GenericModelKind | None
    supports_disk_cache: bool
    supports_cpu_offload: bool
    supports_gpu_offload: bool
    specialization_enabled: bool
    specialization_provider_id: str | None
    reason: str
    details: dict[str, str] = field(default_factory=dict)

    def is_executable(self) -> bool:
        return self.backend_id is not None and self.model_path is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "model_path": None if self.model_path is None else str(self.model_path),
            "support_level": self.support_level.value,
            "generic_model_kind": None if self.generic_model_kind is None else self.generic_model_kind.value,
            "supports_disk_cache": self.supports_disk_cache,
            "supports_cpu_offload": self.supports_cpu_offload,
            "supports_gpu_offload": self.supports_gpu_offload,
            "specialization_enabled": self.specialization_enabled,
            "specialization_provider_id": self.specialization_provider_id,
            "reason": self.reason,
            "details": dict(self.details),
        }

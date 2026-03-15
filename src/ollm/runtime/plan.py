"""Runtime planning types used to describe backend selection and specialization state."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.resolver import ResolvedModel
from ollm.runtime.specialization.passes.base import SpecializationPassId


class SpecializationState(str, Enum):
    """Describe whether specialization is absent, planned, applied, or replaced by fallback."""

    NOT_PLANNED = "not-planned"
    PLANNED = "planned"
    APPLIED = "applied"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    """Describe how oLLM intends to execute a resolved model reference."""

    resolved_model: ResolvedModel
    backend_id: str | None
    model_path: Path | None
    support_level: SupportLevel
    generic_model_kind: GenericModelKind | None
    supports_disk_cache: bool
    supports_cpu_offload: bool
    supports_gpu_offload: bool
    specialization_enabled: bool
    specialization_applied: bool
    specialization_provider_id: str | None
    specialization_state: SpecializationState
    reason: str
    specialization_pass_ids: tuple[SpecializationPassId, ...] = ()
    applied_specialization_pass_ids: tuple[SpecializationPassId, ...] = ()
    fallback_reason: str | None = None
    details: dict[str, str] = field(default_factory=dict)

    def is_executable(self) -> bool:
        """Return whether the plan resolved to a runnable backend."""
        return self.backend_id is not None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the runtime plan."""
        return {
            "backend_id": self.backend_id,
            "model_path": None if self.model_path is None else str(self.model_path),
            "support_level": self.support_level.value,
            "generic_model_kind": None if self.generic_model_kind is None else self.generic_model_kind.value,
            "supports_disk_cache": self.supports_disk_cache,
            "supports_cpu_offload": self.supports_cpu_offload,
            "supports_gpu_offload": self.supports_gpu_offload,
            "specialization_enabled": self.specialization_enabled,
            "specialization_applied": self.specialization_applied,
            "specialization_provider_id": self.specialization_provider_id,
            "specialization_state": self.specialization_state.value,
            "specialization_pass_ids": [pass_id.value for pass_id in self.specialization_pass_ids],
            "applied_specialization_pass_ids": [
                pass_id.value for pass_id in self.applied_specialization_pass_ids
            ],
            "fallback_reason": self.fallback_reason,
            "reason": self.reason,
            "details": dict(self.details),
        }

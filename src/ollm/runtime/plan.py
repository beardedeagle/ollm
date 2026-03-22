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
    """Describe how oLLM intends to execute a resolved model reference.

    Attributes:
        resolved_model (ResolvedModel): Final resolved model metadata for the
            plan.
        backend_id (str | None): Selected backend identifier when the plan is
            executable.
        model_path (Path | None): Local materialized model path when one exists.
        support_level (SupportLevel): Planned support level.
        generic_model_kind (GenericModelKind | None): Generic execution family
            when one applies.
        supports_disk_cache (bool): Whether the selected backend supports disk
            KV cache behavior.
        supports_cpu_offload (bool): Whether CPU offload controls are supported.
        supports_gpu_offload (bool): Whether GPU offload controls are supported.
        specialization_enabled (bool): Whether specialization is enabled for the
            current request.
        specialization_applied (bool): Whether specialization has already been
            applied.
        specialization_provider_id (str | None): Matching specialization
            provider identifier.
        specialization_state (SpecializationState): Current specialization
            lifecycle state.
        reason (str): Human-readable plan summary.
        specialization_pass_ids (tuple[SpecializationPassId, ...]): Planned
            specialization passes.
        applied_specialization_pass_ids (tuple[SpecializationPassId, ...]):
            Applied specialization passes.
        fallback_reason (str | None): Fallback reason when specialization failed.
        details (dict[str, str]): Extra serialized inspection details.
    """

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
        """Return whether the plan resolved to a runnable backend.

        Returns:
            bool: ``True`` when a backend ID was selected.
        """
        return self.backend_id is not None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the runtime plan.

        Returns:
            dict[str, object]: Serialized runtime plan payload.
        """
        return {
            "backend_id": self.backend_id,
            "model_path": None if self.model_path is None else str(self.model_path),
            "support_level": self.support_level.value,
            "generic_model_kind": None
            if self.generic_model_kind is None
            else self.generic_model_kind.value,
            "supports_disk_cache": self.supports_disk_cache,
            "supports_cpu_offload": self.supports_cpu_offload,
            "supports_gpu_offload": self.supports_gpu_offload,
            "specialization_enabled": self.specialization_enabled,
            "specialization_applied": self.specialization_applied,
            "specialization_provider_id": self.specialization_provider_id,
            "specialization_state": self.specialization_state.value,
            "specialization_pass_ids": [
                pass_id.value for pass_id in self.specialization_pass_ids
            ],
            "applied_specialization_pass_ids": [
                pass_id.value for pass_id in self.applied_specialization_pass_ids
            ],
            "fallback_reason": self.fallback_reason,
            "reason": self.reason,
            "details": dict(self.details),
        }

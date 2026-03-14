from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import torch

from ollm.runtime.config import RuntimeConfig
from ollm.runtime.resolver import NativeFamily, ResolvedModel
from ollm.runtime.specialization.passes.base import (
    SpecializationPass,
    SpecializationPassId,
    SpecializationPassTraits,
)
from ollm.utils import Stats


@dataclass(frozen=True, slots=True)
class SpecializationTraits:
    supports_disk_cache: bool
    supports_cpu_offload: bool
    supports_gpu_offload: bool
    details: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "supports_disk_cache": self.supports_disk_cache,
            "supports_cpu_offload": self.supports_cpu_offload,
            "supports_gpu_offload": self.supports_gpu_offload,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class SpecializationMatch:
    provider_id: str
    native_family: NativeFamily
    reason: str
    traits: SpecializationTraits

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "native_family": self.native_family.value,
            "reason": self.reason,
            "traits": self.traits.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class PlannedSpecialization:
    provider_id: str | None
    passes: tuple[SpecializationPass, ...] = ()
    details: dict[str, str] = field(default_factory=dict)

    @property
    def pass_ids(self) -> tuple[SpecializationPassId, ...]:
        return tuple(specialization_pass.pass_id for specialization_pass in self.passes)

    @property
    def traits(self) -> SpecializationPassTraits:
        merged_traits = SpecializationPassTraits()
        for specialization_pass in self.passes:
            merged_traits = merged_traits.merge(specialization_pass.traits)
        return merged_traits

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "pass_ids": [pass_id.value for pass_id in self.pass_ids],
            "passes": [specialization_pass.as_dict() for specialization_pass in self.passes],
            "traits": self.traits.as_dict(),
            "details": dict(self.details),
        }


@dataclass(slots=True)
class OptimizedModelArtifacts:
    model: object
    tokenizer: object
    processor: object | None
    device: torch.device
    stats: Stats | None
    print_suppression_modules: tuple[ModuleType, ...]
    create_cache: Callable[[Path], object | None]
    apply_cpu_offload: Callable[[int], None] | None
    apply_gpu_offload: Callable[[int, int], None] | None


class SpecializationProvider(ABC):
    provider_id: str
    native_family: NativeFamily

    @abstractmethod
    def match(self, resolved_model: ResolvedModel, config: RuntimeConfig) -> SpecializationMatch | None:
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        stats: Stats | None,
    ) -> OptimizedModelArtifacts:
        raise NotImplementedError

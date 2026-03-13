from dataclasses import dataclass, field
from enum import Enum

from ollm.runtime.catalog import ModelCatalogEntry, ModelModality


class SupportLevel(str, Enum):
    GENERIC = "generic"
    OPTIMIZED = "optimized"
    PROVIDER_BACKED = "provider-backed"
    UNSUPPORTED = "unsupported"


@dataclass(slots=True)
class CapabilityProfile:
    support_level: SupportLevel
    modalities: tuple[ModelModality, ...] = (ModelModality.TEXT,)
    requires_processor: bool = False
    supports_disk_cache: bool = True
    supports_local_materialization: bool = False
    supports_provider_execution: bool = False
    supports_specialization: bool = False
    details: dict[str, str] = field(default_factory=dict)

    def supports_modality(self, modality: ModelModality) -> bool:
        return modality in self.modalities

    def as_dict(self) -> dict[str, object]:
        return {
            "support_level": self.support_level.value,
            "modalities": [modality.value for modality in self.modalities],
            "requires_processor": self.requires_processor,
            "supports_disk_cache": self.supports_disk_cache,
            "supports_local_materialization": self.supports_local_materialization,
            "supports_provider_execution": self.supports_provider_execution,
            "supports_specialization": self.supports_specialization,
            "details": dict(self.details),
        }


def capabilities_from_catalog_entry(entry: ModelCatalogEntry) -> CapabilityProfile:
    return CapabilityProfile(
        support_level=SupportLevel.OPTIMIZED,
        modalities=entry.modalities,
        requires_processor=entry.requires_processor,
        supports_disk_cache=entry.supports_disk_cache,
        supports_local_materialization=True,
        supports_provider_execution=False,
        supports_specialization=True,
        details={"source": "catalog", "repo_id": entry.repo_id},
    )


def generic_capabilities(
    modalities: tuple[ModelModality, ...] = (ModelModality.TEXT,),
    *,
    requires_processor: bool = False,
    supports_disk_cache: bool = True,
    details: dict[str, str] | None = None,
) -> CapabilityProfile:
    return CapabilityProfile(
        support_level=SupportLevel.GENERIC,
        modalities=modalities,
        requires_processor=requires_processor,
        supports_disk_cache=supports_disk_cache,
        supports_local_materialization=True,
        supports_provider_execution=False,
        supports_specialization=False,
        details={} if details is None else details,
    )


def provider_capabilities(provider_name: str) -> CapabilityProfile:
    return CapabilityProfile(
        support_level=SupportLevel.PROVIDER_BACKED,
        modalities=(ModelModality.TEXT,),
        requires_processor=False,
        supports_disk_cache=False,
        supports_local_materialization=False,
        supports_provider_execution=True,
        supports_specialization=False,
        details={"provider": provider_name},
    )


def unsupported_capabilities(reason: str) -> CapabilityProfile:
    return CapabilityProfile(
        support_level=SupportLevel.UNSUPPORTED,
        modalities=(ModelModality.TEXT,),
        requires_processor=False,
        supports_disk_cache=False,
        supports_local_materialization=False,
        supports_provider_execution=False,
        supports_specialization=False,
        details={"reason": reason},
    )

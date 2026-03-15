"""Resolution from user-supplied model references to normalized runtime inputs."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ollm.runtime.capabilities import (
    CapabilityProfile,
    capabilities_from_catalog_entry,
    provider_capabilities,
    unsupported_capabilities,
)
from ollm.runtime.capability_discovery import CapabilityDiscovery, GenericModelKind
from ollm.runtime.catalog import ModelCatalogEntry, find_model_catalog_entry, list_model_catalog
from ollm.runtime.reference import ModelReference


class ModelSourceKind(str, Enum):
    """Describe where a model reference originates from."""

    BUILTIN = "builtin"
    HUGGING_FACE = "hugging-face"
    LOCAL_PATH = "local-path"
    PROVIDER = "provider"
    OPAQUE = "opaque"


class NativeFamily(str, Enum):
    """Known optimized-native model families."""

    LLAMA = "llama"
    GEMMA3 = "gemma3"
    GPT_OSS = "gpt-oss"
    QWEN3_NEXT = "qwen3-next"
    VOXTRAL = "voxtral"


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Resolved model metadata used by planning, inspection, and loading."""

    reference: ModelReference
    source_kind: ModelSourceKind
    normalized_name: str
    model_path: Path | None
    repo_id: str | None
    revision: str | None
    provider_name: str | None
    catalog_entry: ModelCatalogEntry | None
    capabilities: CapabilityProfile
    native_family: NativeFamily | None
    resolution_message: str
    architecture: str | None
    model_type: str | None
    generic_model_kind: GenericModelKind | None

    def is_downloadable(self) -> bool:
        """Return whether the resolved model can be materialized locally."""
        return self.repo_id is not None and self.model_path is not None


class ModelResolver:
    """Resolve model references into built-in, local, Hugging Face, or provider-backed forms."""

    def __init__(self, capability_discovery: CapabilityDiscovery | None = None):
        self._capability_discovery = capability_discovery or CapabilityDiscovery()

    def resolve(self, raw_reference: str, models_dir: Path) -> ResolvedModel:
        """Resolve a raw model reference without loading model weights."""
        reference = ModelReference.parse(raw_reference)
        model_root = models_dir.expanduser().resolve()

        if reference.has_provider_scheme():
            provider_name = reference.scheme or "provider"
            return ResolvedModel(
                reference=reference,
                source_kind=ModelSourceKind.PROVIDER,
                normalized_name=reference.identifier,
                model_path=None,
                repo_id=None,
                revision=None,
                provider_name=provider_name,
                catalog_entry=None,
                capabilities=provider_capabilities(provider_name),
                native_family=None,
                resolution_message=f"Provider-backed model reference for {provider_name}.",
                architecture=None,
                model_type=None,
                generic_model_kind=None,
            )

        if reference.local_path is not None:
            return self._resolve_local_path(reference)

        catalog_entry = find_model_catalog_entry(reference.identifier)
        if catalog_entry is not None and reference.scheme is None:
            return ResolvedModel(
                reference=reference,
                source_kind=ModelSourceKind.BUILTIN,
                normalized_name=catalog_entry.model_id,
                model_path=model_root / catalog_entry.model_id,
                repo_id=catalog_entry.repo_id,
                revision=reference.revision,
                provider_name=None,
                catalog_entry=catalog_entry,
                capabilities=capabilities_from_catalog_entry(catalog_entry),
                native_family=_native_family_from_catalog_entry(catalog_entry),
                resolution_message=f"Built-in optimized alias '{catalog_entry.model_id}'.",
                architecture=None,
                model_type=None,
                generic_model_kind=None,
            )

        if reference.is_huggingface_reference():
            return self._resolve_hugging_face(reference, model_root)

        implicit_local_path = model_root / reference.identifier
        if implicit_local_path.exists() and implicit_local_path.is_dir():
            local_reference = ModelReference(
                raw=reference.raw,
                scheme="path",
                identifier=str(implicit_local_path),
                revision=None,
                local_path=implicit_local_path,
            )
            return self._resolve_local_path(local_reference)

        return ResolvedModel(
            reference=reference,
            source_kind=ModelSourceKind.OPAQUE,
            normalized_name=reference.identifier,
            model_path=None,
            repo_id=None,
            revision=reference.revision,
            provider_name=None,
            catalog_entry=None,
            capabilities=unsupported_capabilities(
                f"Model reference '{reference.raw}' is not a built-in alias, local directory, Hugging Face repository, or provider-prefixed reference."
            ),
            native_family=None,
            resolution_message=(
                f"Model reference '{reference.raw}' could not be resolved to a runnable backend or materialization path."
            ),
            architecture=None,
            model_type=None,
            generic_model_kind=None,
        )

    def discover_local_models(self, models_dir: Path) -> tuple[ResolvedModel, ...]:
        """Inspect local materialized model directories under a models root."""
        model_root = models_dir.expanduser().resolve()
        if not model_root.exists() or not model_root.is_dir():
            return ()

        discovered: list[ResolvedModel] = []
        for child in sorted(model_root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            discovered.append(self.resolve(str(child), model_root))
        return tuple(discovered)

    def inspect_materialized_model(
        self,
        reference: ModelReference,
        model_path: Path,
        *,
        source_kind: ModelSourceKind,
        repo_id: str | None,
        revision: str | None,
        provider_name: str | None,
        catalog_entry: ModelCatalogEntry | None,
    ) -> ResolvedModel:
        """Inspect a materialized local model directory and derive runtime capabilities."""
        inspection = self._capability_discovery.inspect_model_path(model_path)
        capabilities = inspection.capabilities
        if catalog_entry is not None and source_kind in {ModelSourceKind.BUILTIN, ModelSourceKind.HUGGING_FACE}:
            capabilities = capabilities_from_catalog_entry(catalog_entry)
        native_family = (
            _native_family_from_catalog_entry(catalog_entry)
            if catalog_entry is not None
            else _native_family_from_architecture(inspection.architecture)
        )
        if native_family is not None:
            capabilities.supports_specialization = True
            capabilities.details["native_family"] = native_family.value
        normalized_name = catalog_entry.model_id if catalog_entry is not None else model_path.name
        return ResolvedModel(
            reference=reference,
            source_kind=source_kind,
            normalized_name=normalized_name,
            model_path=model_path,
            repo_id=repo_id,
            revision=revision,
            provider_name=provider_name,
            catalog_entry=catalog_entry,
            capabilities=capabilities,
            native_family=native_family,
            resolution_message=inspection.message,
            architecture=inspection.architecture,
            model_type=inspection.model_type,
            generic_model_kind=inspection.generic_model_kind,
        )

    def _resolve_local_path(self, reference: ModelReference) -> ResolvedModel:
        local_path = reference.local_path
        assert local_path is not None
        catalog_entry = find_model_catalog_entry(local_path.name)
        return self.inspect_materialized_model(
            reference,
            local_path,
            source_kind=ModelSourceKind.LOCAL_PATH,
            repo_id=None,
            revision=None,
            provider_name=None,
            catalog_entry=catalog_entry,
        )

    def _resolve_hugging_face(self, reference: ModelReference, model_root: Path) -> ResolvedModel:
        catalog_entry = _find_catalog_entry_by_repo_id(reference.identifier)
        model_path = model_root / reference.materialization_name()
        if catalog_entry is not None:
            if reference.revision is None:
                model_path = model_root / catalog_entry.model_id
            return ResolvedModel(
                reference=reference,
                source_kind=ModelSourceKind.HUGGING_FACE,
                normalized_name=catalog_entry.model_id,
                model_path=model_path,
                repo_id=reference.identifier,
                revision=reference.revision,
                provider_name=None,
                catalog_entry=catalog_entry,
                capabilities=capabilities_from_catalog_entry(catalog_entry),
                native_family=_native_family_from_catalog_entry(catalog_entry),
                resolution_message=(
                    f"Hugging Face repository '{reference.identifier}' matching built-in optimized alias '{catalog_entry.model_id}'."
                ),
                architecture=None,
                model_type=None,
                generic_model_kind=None,
            )

        if model_path.exists() and model_path.is_dir():
            return self.inspect_materialized_model(
                reference,
                model_path,
                source_kind=ModelSourceKind.HUGGING_FACE,
                repo_id=reference.identifier,
                revision=reference.revision,
                provider_name=None,
                catalog_entry=None,
            )

        capabilities = unsupported_capabilities(
            f"Hugging Face repository '{reference.identifier}' must be materialized locally before generic capability discovery can run."
        )
        capabilities.supports_local_materialization = True
        capabilities.details["source"] = "hugging-face"
        capabilities.details["repo_id"] = reference.identifier
        return ResolvedModel(
            reference=reference,
            source_kind=ModelSourceKind.HUGGING_FACE,
            normalized_name=reference.identifier,
            model_path=model_path,
            repo_id=reference.identifier,
            revision=reference.revision,
            provider_name=None,
            catalog_entry=None,
            capabilities=capabilities,
            native_family=None,
            resolution_message=f"Hugging Face repository '{reference.identifier}'.",
            architecture=None,
            model_type=None,
            generic_model_kind=None,
        )


def _find_catalog_entry_by_repo_id(repo_id: str) -> ModelCatalogEntry | None:
    for entry in list_model_catalog():
        if entry.repo_id == repo_id:
            return entry
    return None


def _native_family_from_architecture(architecture: str | None) -> NativeFamily | None:
    if architecture == "LlamaForCausalLM":
        return NativeFamily.LLAMA
    if architecture in {"Gemma3ForConditionalGeneration", "Gemma3ForCausalLM"}:
        return NativeFamily.GEMMA3
    if architecture in {"GptOssForCausalLM", "OpenAIGptOssForCausalLM"}:
        return NativeFamily.GPT_OSS
    if architecture in {"Qwen3NextForCausalLM", "Qwen3MoeForCausalLM"}:
        return NativeFamily.QWEN3_NEXT
    if architecture in {"VoxtralForConditionalGeneration", "VoxtralForSpeechSeq2Seq"}:
        return NativeFamily.VOXTRAL
    return None


def _native_family_from_catalog_entry(entry: ModelCatalogEntry) -> NativeFamily:
    if entry.model_id.startswith("llama"):
        return NativeFamily.LLAMA
    if entry.model_id.startswith("gemma3"):
        return NativeFamily.GEMMA3
    if entry.model_id.startswith("gpt-oss"):
        return NativeFamily.GPT_OSS
    if entry.model_id.startswith("qwen3-next"):
        return NativeFamily.QWEN3_NEXT
    return NativeFamily.VOXTRAL

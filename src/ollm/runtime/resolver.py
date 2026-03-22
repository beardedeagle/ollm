"""Resolution from user-supplied model references to normalized runtime inputs."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ollm.runtime.capabilities import (
    CapabilityProfile,
    capabilities_from_catalog_entry,
    unsupported_capabilities,
)
from ollm.runtime.capability_discovery import CapabilityDiscovery, GenericModelKind
from ollm.runtime.catalog import (
    ModelCatalogEntry,
    find_model_catalog_entry,
    list_model_catalog,
)
from ollm.runtime.reference import ModelReference


class ModelSourceKind(str, Enum):
    """Describe where a model reference originates from."""

    BUILTIN = "builtin"
    HUGGING_FACE = "hugging-face"
    LOCAL_PATH = "local-path"
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
    """Resolved model metadata used by planning, inspection, and loading.

    Attributes:
        reference (ModelReference): Parsed user-facing model reference.
        source_kind (ModelSourceKind): Where the model was resolved from.
        normalized_name (str): Stable display name used by planning and docs.
        model_path (Path | None): Local materialization path when one exists.
        repo_id (str | None): Hugging Face repository identifier when
            materialization is supported.
        revision (str | None): Optional requested revision.
        catalog_entry (ModelCatalogEntry | None): Built-in catalog entry when
            one matched.
        capabilities (CapabilityProfile): Capability and support metadata.
        native_family (NativeFamily | None): Matching optimized-native family
            when one exists.
        resolution_message (str): Human-readable resolution summary.
        architecture (str | None): Inspected architecture for local models.
        model_type (str | None): Inspected model type for local models.
        generic_model_kind (GenericModelKind | None): Generic execution kind
            when one can be inferred.
    """

    reference: ModelReference
    source_kind: ModelSourceKind
    normalized_name: str
    model_path: Path | None
    repo_id: str | None
    revision: str | None
    catalog_entry: ModelCatalogEntry | None
    capabilities: CapabilityProfile
    native_family: NativeFamily | None
    resolution_message: str
    architecture: str | None
    model_type: str | None
    generic_model_kind: GenericModelKind | None

    def is_downloadable(self) -> bool:
        """Return whether the resolved model can be materialized locally.

        Returns:
            bool: ``True`` when both a repository ID and a local model path are
            available.
        """
        return self.repo_id is not None and self.model_path is not None


class ModelResolver:
    """Resolve model references into built-in, local, or Hugging Face forms.

    Args:
        capability_discovery (CapabilityDiscovery | None): Optional capability
            discovery helper for inspecting local model directories.
    """

    def __init__(self, capability_discovery: CapabilityDiscovery | None = None):
        self._capability_discovery = capability_discovery or CapabilityDiscovery()

    def resolve(self, raw_reference: str, models_dir: Path) -> ResolvedModel:
        """Resolve a raw model reference without loading model weights.

        Args:
            raw_reference (str): User-facing model reference string.
            models_dir (Path): Local models root used for implicit path
                resolution.

        Returns:
            ResolvedModel: Normalized model metadata used by planning and
            loading.
        """
        reference = ModelReference.parse(raw_reference)
        model_root = models_dir.expanduser().resolve()

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
            catalog_entry=None,
            capabilities=unsupported_capabilities(
                f"Model reference '{reference.raw}' is not a built-in alias, local directory, or Hugging Face repository."
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
        """Inspect local materialized model directories under a models root.

        Args:
            models_dir (Path): Local models root to scan.

        Returns:
            tuple[ResolvedModel, ...]: Resolved local model directories found
            under the given root.
        """
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
        catalog_entry: ModelCatalogEntry | None,
    ) -> ResolvedModel:
        """Inspect a materialized local model directory and derive capabilities.

        Args:
            reference (ModelReference): Parsed reference associated with the
                local model path.
            model_path (Path): Materialized model directory to inspect.
            source_kind (ModelSourceKind): Source bucket for the materialized
                model.
            repo_id (str | None): Optional repository ID for materialized
                managed models.
            revision (str | None): Optional requested revision.
            catalog_entry (ModelCatalogEntry | None): Matching built-in catalog
                entry when one exists.

        Returns:
            ResolvedModel: Inspected model metadata with derived capabilities and
            native-family information.
        """
        inspection = self._capability_discovery.inspect_model_path(model_path)
        capabilities = inspection.capabilities
        if catalog_entry is not None and source_kind in {
            ModelSourceKind.BUILTIN,
            ModelSourceKind.HUGGING_FACE,
        }:
            capabilities = capabilities_from_catalog_entry(catalog_entry)
        native_family = (
            _native_family_from_catalog_entry(catalog_entry)
            if catalog_entry is not None
            else _native_family_from_architecture(inspection.architecture)
        )
        if native_family is not None:
            capabilities.supports_specialization = True
            capabilities.details["native_family"] = native_family.value
        normalized_name = (
            catalog_entry.model_id if catalog_entry is not None else model_path.name
        )
        return ResolvedModel(
            reference=reference,
            source_kind=source_kind,
            normalized_name=normalized_name,
            model_path=model_path,
            repo_id=repo_id,
            revision=revision,
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
            catalog_entry=catalog_entry,
        )

    def _resolve_hugging_face(
        self, reference: ModelReference, model_root: Path
    ) -> ResolvedModel:
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

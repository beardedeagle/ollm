from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from transformers import AutoConfig

from ollm.runtime.capabilities import (
    CapabilityProfile,
    SupportLevel,
    capabilities_from_catalog_entry,
    generic_capabilities,
    provider_capabilities,
    unsupported_capabilities,
)
from ollm.runtime.catalog import ModelCatalogEntry, ModelModality, find_model_catalog_entry, list_model_catalog
from ollm.runtime.reference import ModelReference


class ModelSourceKind(str, Enum):
    BUILTIN = "builtin"
    HUGGING_FACE = "hugging-face"
    LOCAL_PATH = "local-path"
    PROVIDER = "provider"
    OPAQUE = "opaque"


class NativeFamily(str, Enum):
    LLAMA = "llama"
    GEMMA3 = "gemma3"
    GPT_OSS = "gpt-oss"
    QWEN3_NEXT = "qwen3-next"
    VOXTRAL = "voxtral"


@dataclass(frozen=True, slots=True)
class ResolvedModel:
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

    def is_downloadable(self) -> bool:
        return self.repo_id is not None and self.model_path is not None


class ModelResolver:
    def resolve(self, raw_reference: str, models_dir: Path) -> ResolvedModel:
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
        )

    def discover_local_models(self, models_dir: Path) -> tuple[ResolvedModel, ...]:
        model_root = models_dir.expanduser().resolve()
        if not model_root.exists() or not model_root.is_dir():
            return ()

        discovered: list[ResolvedModel] = []
        for child in sorted(model_root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            discovered.append(self.resolve(str(child), model_root))
        return tuple(discovered)

    def _resolve_local_path(self, reference: ModelReference) -> ResolvedModel:
        local_path = reference.local_path
        assert local_path is not None
        catalog_entry = find_model_catalog_entry(local_path.name)
        if not local_path.exists():
            return ResolvedModel(
                reference=reference,
                source_kind=ModelSourceKind.LOCAL_PATH,
                normalized_name=local_path.name,
                model_path=local_path,
                repo_id=None,
                revision=None,
                provider_name=None,
                catalog_entry=catalog_entry,
                capabilities=unsupported_capabilities(f"Resolved local model path '{local_path}' does not exist."),
                native_family=_native_family_from_catalog_entry(catalog_entry) if catalog_entry is not None else None,
                resolution_message=f"Local model path '{local_path}' does not exist.",
            )

        architecture = _load_architecture(local_path)
        capabilities = _capabilities_from_local_path(local_path, architecture)
        native_family = _native_family_from_architecture(architecture)
        message = f"Local model directory '{local_path}'."
        if architecture is not None:
            message = f"Local model directory '{local_path}' with detected architecture '{architecture}'."
        return ResolvedModel(
            reference=reference,
            source_kind=ModelSourceKind.LOCAL_PATH,
            normalized_name=local_path.name,
            model_path=local_path,
            repo_id=None,
            revision=None,
            provider_name=None,
            catalog_entry=catalog_entry,
            capabilities=capabilities,
            native_family=native_family,
            resolution_message=message,
        )

    def _resolve_hugging_face(self, reference: ModelReference, model_root: Path) -> ResolvedModel:
        catalog_entry = _find_catalog_entry_by_repo_id(reference.identifier)
        normalized_name = reference.identifier
        model_path = model_root / reference.materialization_name()
        capabilities = unsupported_capabilities(
            f"Hugging Face repository '{reference.identifier}' is materializable, but the current generic runtime only executes materialized Llama and Gemma3 families."
        )
        capabilities.supports_local_materialization = True
        capabilities.details["source"] = "hugging-face"
        capabilities.details["repo_id"] = reference.identifier
        native_family = None
        message = f"Hugging Face repository '{reference.identifier}'."

        if catalog_entry is not None:
            normalized_name = catalog_entry.model_id
            model_path = model_root / catalog_entry.model_id
            capabilities = capabilities_from_catalog_entry(catalog_entry)
            native_family = _native_family_from_catalog_entry(catalog_entry)
            message = (
                f"Hugging Face repository '{reference.identifier}' matching built-in optimized alias '{catalog_entry.model_id}'."
            )
        elif model_path.exists():
            architecture = _load_architecture(model_path)
            capabilities = _capabilities_from_local_path(model_path, architecture)
            native_family = _native_family_from_architecture(architecture)
            if architecture is not None:
                message = (
                    f"Materialized Hugging Face repository '{reference.identifier}' with detected architecture '{architecture}'."
                )

        return ResolvedModel(
            reference=reference,
            source_kind=ModelSourceKind.HUGGING_FACE,
            normalized_name=normalized_name,
            model_path=model_path,
            repo_id=reference.identifier,
            revision=reference.revision,
            provider_name=None,
            catalog_entry=catalog_entry,
            capabilities=capabilities,
            native_family=native_family,
            resolution_message=message,
        )


def _find_catalog_entry_by_repo_id(repo_id: str) -> ModelCatalogEntry | None:
    for entry in list_model_catalog():
        if entry.repo_id == repo_id:
            return entry
    return None


def _load_architecture(model_path: Path) -> str | None:
    try:
        config = AutoConfig.from_pretrained(model_path)
    except Exception:
        return None
    architectures = getattr(config, "architectures", None)
    if not architectures:
        return None
    return str(architectures[0])


def _capabilities_from_local_path(model_path: Path, architecture: str | None) -> CapabilityProfile:
    modalities = (ModelModality.TEXT,)
    requires_processor = any(
        (model_path / file_name).exists()
        for file_name in ("processor_config.json", "preprocessor_config.json")
    )
    supports_disk_cache = architecture not in {"GptOssForCausalLM", "OpenAIGptOssForCausalLM"}
    support_level = SupportLevel.UNSUPPORTED

    if architecture == "LlamaForCausalLM":
        support_level = SupportLevel.GENERIC
    elif architecture in {"Gemma3ForConditionalGeneration", "Gemma3ForCausalLM"}:
        support_level = SupportLevel.GENERIC
    if architecture == "Gemma3ForConditionalGeneration":
        modalities = (ModelModality.TEXT, ModelModality.IMAGE)
        requires_processor = True
    elif architecture in {"VoxtralForConditionalGeneration", "VoxtralForSpeechSeq2Seq"}:
        modalities = (ModelModality.TEXT, ModelModality.AUDIO)
        requires_processor = True

    details = {"source": "local-path"}
    if architecture is not None:
        details["architecture"] = architecture
    if support_level is SupportLevel.UNSUPPORTED:
        if architecture is None:
            details["reason"] = "Model architecture could not be determined from the local path."
        else:
            details["reason"] = (
                f"Architecture '{architecture}' is not executable through the current generic runtime."
            )

    if support_level is SupportLevel.GENERIC:
        return generic_capabilities(
            modalities=modalities,
            requires_processor=requires_processor,
            supports_disk_cache=supports_disk_cache,
            details=details,
        )

    return CapabilityProfile(
        support_level=SupportLevel.UNSUPPORTED,
        modalities=modalities,
        requires_processor=requires_processor,
        supports_disk_cache=supports_disk_cache,
        supports_local_materialization=True,
        supports_provider_execution=False,
        supports_specialization=False,
        details=details,
    )


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

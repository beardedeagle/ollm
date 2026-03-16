from dataclasses import dataclass
from enum import Enum


class ModelModality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class ModelCatalogEntry:
    model_id: str
    summary: str
    repo_id: str
    modalities: tuple[ModelModality, ...]
    requires_processor: bool = False
    supports_disk_cache: bool = True

    def supports_modality(self, modality: ModelModality) -> bool:
        return modality in self.modalities


MODEL_CATALOG: dict[str, ModelCatalogEntry] = {
    "llama3-1B-chat": ModelCatalogEntry(
        model_id="llama3-1B-chat",
        summary="Small local Llama chat alias with optimized disk-cache support.",
        repo_id="unsloth/Llama-3.2-1B-Instruct",
        modalities=(ModelModality.TEXT,),
    ),
    "llama3-3B-chat": ModelCatalogEntry(
        model_id="llama3-3B-chat",
        summary="Mid-sized local Llama chat alias with optimized disk-cache support.",
        repo_id="unsloth/Llama-3.2-3B-Instruct",
        modalities=(ModelModality.TEXT,),
    ),
    "llama3-8B-chat": ModelCatalogEntry(
        model_id="llama3-8B-chat",
        summary="Largest built-in Llama chat alias with optimized disk-cache support.",
        repo_id="unsloth/Meta-Llama-3.1-8B-Instruct",
        modalities=(ModelModality.TEXT,),
    ),
    "gpt-oss-20B": ModelCatalogEntry(
        model_id="gpt-oss-20B",
        summary="Built-in gpt-oss alias with optimized chunked MLP support.",
        repo_id="AnuarSh/gpt-oss-20B",
        modalities=(ModelModality.TEXT,),
        supports_disk_cache=False,
    ),
    "qwen3-next-80B": ModelCatalogEntry(
        model_id="qwen3-next-80B",
        summary="Built-in Qwen3-Next alias with optimized MoE and disk-cache support.",
        repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
        modalities=(ModelModality.TEXT,),
    ),
    "gemma3-12B": ModelCatalogEntry(
        model_id="gemma3-12B",
        summary="Built-in Gemma alias for optimized text and image chat.",
        repo_id="google/gemma-3-12b-it",
        modalities=(ModelModality.TEXT, ModelModality.IMAGE),
        requires_processor=True,
    ),
    "voxtral-small-24B": ModelCatalogEntry(
        model_id="voxtral-small-24B",
        summary="Built-in Voxtral alias for optimized text and audio chat.",
        repo_id="mistralai/Voxtral-Small-24B-2507",
        modalities=(ModelModality.TEXT, ModelModality.AUDIO),
        requires_processor=True,
    ),
}


def find_model_catalog_entry(model_reference: str) -> ModelCatalogEntry | None:
    return MODEL_CATALOG.get(model_reference)


def get_model_catalog_entry(model_reference: str) -> ModelCatalogEntry:
    entry = find_model_catalog_entry(model_reference)
    if entry is None:
        available = ", ".join(sorted(MODEL_CATALOG))
        raise ValueError(
            f"'{model_reference}' is not a built-in oLLM alias. Available built-in aliases: {available}"
        )
    return entry


def list_model_catalog() -> tuple[ModelCatalogEntry, ...]:
    return tuple(MODEL_CATALOG[model_id] for model_id in sorted(MODEL_CATALOG))


def builtin_model_aliases() -> tuple[str, ...]:
    return tuple(entry.model_id for entry in list_model_catalog())

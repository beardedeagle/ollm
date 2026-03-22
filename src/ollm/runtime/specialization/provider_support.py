"""Shared helpers for optimized specialization providers."""

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import torch

from ollm.kv_cache.resident import ResidentKVCache
from ollm.kv_cache.strategy import normalize_kv_cache_strategy
from ollm.runtime.resolver import NativeFamily, ResolvedModel
from ollm.runtime.specialization.base import (
    SpecializationMatch,
    SpecializationTraits,
)

LOGGER = logging.getLogger(__name__)


class _CpuOffloadModel(Protocol):
    num_hidden_layers: int

    def offload_layers_to_cpu_indices(self, layer_indices: tuple[int, ...]) -> None: ...


class _GpuCpuOffloadModel(_CpuOffloadModel, Protocol):
    config: object

    def offload_layers_to_gpu_cpu(
        self,
        gpu_layers_num: int = 0,
        cpu_layers_num: int = 0,
    ) -> None: ...


def get_attention_implementation() -> str | None:
    try:
        importlib.import_module("flash_attn")
        return "flash_attention_2"
    except ImportError:
        LOGGER.debug(
            "flash_attention_2 is not imported. The context length will be limited."
        )
        return None


def resolved_model_path(resolved_model: ResolvedModel) -> Path:
    if resolved_model.model_path is None:
        raise ValueError(
            f"Resolved model path is missing for {resolved_model.reference.raw}"
        )
    model_path = resolved_model.model_path.expanduser().resolve()
    if not model_path.exists() or not model_path.is_dir():
        raise ValueError(f"Resolved model path does not exist: {model_path}")
    return model_path


def build_match(
    *,
    resolved_model: ResolvedModel,
    native_family: NativeFamily,
    provider_id: str,
    supports_disk_cache: bool,
    supports_cpu_offload: bool,
    supports_gpu_offload: bool,
    details: dict[str, str] | None = None,
) -> SpecializationMatch | None:
    if resolved_model.native_family is not native_family:
        return None
    match_details = {"native_family": native_family.value}
    if details is not None:
        match_details.update(details)
    return SpecializationMatch(
        provider_id=provider_id,
        native_family=native_family,
        reason=(
            f"Selected specialization provider '{provider_id}' for native family "
            f"'{native_family.value}' and model reference '{resolved_model.reference.raw}'."
        ),
        traits=SpecializationTraits(
            supports_disk_cache=supports_disk_cache,
            supports_cpu_offload=supports_cpu_offload,
            supports_gpu_offload=supports_gpu_offload,
            details=match_details,
        ),
    )


def matches_architecture(
    resolved_model: ResolvedModel, architectures: tuple[str, ...]
) -> bool:
    if resolved_model.architecture is None:
        return False
    return resolved_model.architecture in architectures


def finalize_model(model: object, device: torch.device) -> object:
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()
    move_method = getattr(model, "to", None)
    if callable(move_method):
        move_method(device)
    return model


def build_execution_device_details(device: str | torch.device) -> dict[str, str]:
    resolved_device = torch.device(device)
    return {
        "execution_device_type": resolved_device.type,
        "specialization_device_profile": (
            "host" if resolved_device.type == "cpu" else "accelerator-resident"
        ),
    }


def unsupported_disk_cache_factory(model_reference: str):
    def create_cache(
        cache_dir: Path,
        cache_strategy: str | None = None,
        cache_lifecycle: str | None = None,
        cache_window_tokens: int | None = None,
    ) -> ResidentKVCache | None:
        del cache_dir, cache_lifecycle, cache_window_tokens
        if normalize_kv_cache_strategy(cache_strategy) == "resident":
            return ResidentKVCache()
        LOGGER.info(
            "%s DiskCache is not supported at the moment. Using default DynamicCache instead.",
            model_reference,
        )
        return None

    return create_cache


def is_sharded_model_dir(model_path: Path) -> bool:
    return any(
        "index.json" in file_path.name
        for file_path in model_path.iterdir()
        if file_path.is_file()
    )


def load_specialized_model(
    loader: Callable[..., object],
    model_path: Path,
    *,
    torch_dtype: torch.dtype | str,
    attn_implementation: str | None = None,
) -> object:
    loader_kwargs: dict[str, object] = {
        "pretrained_model_name_or_path": str(model_path),
        "torch_dtype": torch_dtype,
        "device_map": "cpu",
        "trust_remote_code": False,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "ignore_mismatched_sizes": True,
    }
    if attn_implementation is not None:
        loader_kwargs["attn_implementation"] = attn_implementation
    return _load_specialized_model_with_fallbacks(loader, loader_kwargs)


def _load_specialized_model_with_fallbacks(
    loader: Callable[..., object],
    loader_kwargs: dict[str, object],
) -> object:
    try:
        return loader(**loader_kwargs)
    except (TypeError, ValueError) as exc:
        if "attn_implementation" in loader_kwargs and "attn_implementation" in str(exc):
            refined_kwargs = dict(loader_kwargs)
            refined_kwargs.pop("attn_implementation")
            return _load_specialized_model_with_fallbacks(loader, refined_kwargs)
        if "low_cpu_mem_usage" in loader_kwargs and "low_cpu_mem_usage" in str(exc):
            refined_kwargs = dict(loader_kwargs)
            refined_kwargs.pop("low_cpu_mem_usage")
            return _load_specialized_model_with_fallbacks(loader, refined_kwargs)
        raise

"""Cache-factory helpers shared by optimized specialization providers."""

from typing import Protocol, cast

import torch

from ollm.kvcache import KVCache
from ollm.runtime.config import RuntimeConfig
from ollm.utils import Stats


class _Qwen3NextDiskCacheFactory(Protocol):
    def __call__(
        self,
        model_config: object,
        *,
        cache_dir: str,
        device: torch.device,
        stats: Stats | None,
        cache_strategy: str,
        cache_lifecycle: str,
        cache_window_tokens: int | None,
    ) -> object: ...


class _Qwen3NextCacheModule(Protocol):
    Qwen3NextDiskCache: _Qwen3NextDiskCacheFactory


def build_kv_cache_factory(
    *,
    config: RuntimeConfig,
    device: torch.device,
    stats: Stats | None,
):
    """Return a KV cache factory honoring strategy and lifecycle overrides."""

    return (
        lambda cache_dir, cache_strategy=None, cache_lifecycle=None, cache_window_tokens=None: (
            KVCache(
                cache_dir=str(cache_dir),
                device=device,
                stats=stats,
                cache_strategy=(
                    config.kv_cache_strategy
                    if cache_strategy is None
                    else cache_strategy
                ),
                cache_lifecycle=(
                    config.kv_cache_lifecycle
                    if cache_lifecycle is None
                    else cache_lifecycle
                ),
                cache_window_tokens=(
                    config.resolved_kv_cache_window_tokens()
                    if cache_window_tokens is None
                    else cache_window_tokens
                ),
            )
        )
    )


def build_qwen3_cache_factory(
    *,
    module: object,
    model_config: object,
    config: RuntimeConfig,
    device: torch.device,
    stats: Stats | None,
):
    """Return a Qwen3-Next disk-cache factory honoring lifecycle overrides."""

    typed_module = cast(_Qwen3NextCacheModule, module)
    return (
        lambda cache_dir, cache_strategy=None, cache_lifecycle=None, cache_window_tokens=None: (
            typed_module.Qwen3NextDiskCache(
                model_config,
                cache_dir=str(cache_dir),
                device=device,
                stats=stats,
                cache_strategy=(
                    config.kv_cache_strategy
                    if cache_strategy is None
                    else cache_strategy
                ),
                cache_lifecycle=(
                    config.kv_cache_lifecycle
                    if cache_lifecycle is None
                    else cache_lifecycle
                ),
                cache_window_tokens=(
                    config.resolved_kv_cache_window_tokens()
                    if cache_window_tokens is None
                    else cache_window_tokens
                ),
            )
        )
    )

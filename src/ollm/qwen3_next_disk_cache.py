"""Qwen3-Next disk-cache adapter backed by the shared KV cache layer."""

from pathlib import Path
from typing import cast

import torch
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextConfig,
    Qwen3NextDynamicCache,
)

from ollm.kv_cache import oCache
from ollm.kv_cache.policy import KVCachePolicy
from ollm.kv_cache.strategy import DEFAULT_KV_CACHE_STRATEGY
from ollm.utils import Stats


class Qwen3NextDiskCache(Qwen3NextDynamicCache, oCache):
    """Disk-backed Qwen3-Next cache that delegates persistence to `oCache`."""

    def __init__(
        self,
        config: object,
        cache_dir: str | Path = "./kv_cache",
        device: str | torch.device = "cuda:0",
        stats: Stats | None = None,
        policy: KVCachePolicy | None = None,
        cache_strategy: str = DEFAULT_KV_CACHE_STRATEGY,
        cache_lifecycle: str = "runtime-scoped",
        cache_window_tokens: int | None = None,
    ) -> None:
        super().__init__(cast(Qwen3NextConfig, config))
        self.ini_ocache(
            cache_dir,
            device,
            stats,
            policy,
            cache_strategy,
            cache_lifecycle,
            cache_window_tokens,
        )
        self.seq_lengths = [0 for _ in range(len(self.key_cache))]

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        resolved_layer_idx = 0 if layer_idx is None else layer_idx
        return self.seq_lengths[resolved_layer_idx]

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Beam search is not supported for Qwen3NextDiskCache")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("Beam search is not supported for Qwen3NextDiskCache")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = self.load_from_disk(layer_idx)
        if tensors is not None:
            self.key_cache[layer_idx], self.value_cache[layer_idx] = tensors

        out = super().update(key_states, value_states, layer_idx, cache_kwargs)
        bounded = self._finalize_updated_tensors(
            layer_idx,
            out,
            (key_states, value_states),
        )
        self.seq_lengths[layer_idx] = bounded[0].shape[-2]
        self.key_cache[layer_idx], self.value_cache[layer_idx] = (
            torch.empty(0),
            torch.empty(0),
        )
        return bounded

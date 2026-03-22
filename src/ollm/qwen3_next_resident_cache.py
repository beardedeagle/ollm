"""Resident-only Qwen3-Next cache with reporting hooks."""

from typing import cast

import torch
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextConfig,
    Qwen3NextDynamicCache,
)

from ollm.kv_cache.resident import build_resident_cache_snapshot
from ollm.kv_cache.state import KVCacheStateSnapshot


class Qwen3NextResidentCache(Qwen3NextDynamicCache):
    """Fully resident Qwen3-Next cache with truthful state snapshots."""

    def __init__(self, config: object, *, device: torch.device) -> None:
        super().__init__(cast(Qwen3NextConfig, config))
        self.seq_lengths: list[int] = []
        self._device = device
        self._resident_layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        resolved_layer_idx = 0 if layer_idx is None else layer_idx
        if resolved_layer_idx >= len(self.seq_lengths):
            return 0
        return self.seq_lengths[resolved_layer_idx]

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Beam search is not supported for Qwen3NextResidentCache"
        )

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError(
            "Beam search is not supported for Qwen3NextResidentCache"
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resident = self._resident_layers.get(layer_idx)
        if resident is not None:
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(torch.empty(0, device=self._device))
                self.value_cache.append(torch.empty(0, device=self._device))
            self.key_cache[layer_idx], self.value_cache[layer_idx] = resident
        out = super().update(key_states, value_states, layer_idx, cache_kwargs)
        while len(self.seq_lengths) <= layer_idx:
            self.seq_lengths.append(0)
        self.seq_lengths[layer_idx] = int(out[0].shape[-2])
        self._resident_layers[layer_idx] = (
            out[0].detach().contiguous(),
            out[1].detach().contiguous(),
        )
        self.key_cache[layer_idx], self.value_cache[layer_idx] = (
            torch.empty(0, device=self._device),
            torch.empty(0, device=self._device),
        )
        return out

    def cache_state_snapshot(self) -> KVCacheStateSnapshot:
        resident_layers = tuple(self._resident_layers.values())
        return build_resident_cache_snapshot(
            strategy_id="resident",
            resident_layers=resident_layers,
        )

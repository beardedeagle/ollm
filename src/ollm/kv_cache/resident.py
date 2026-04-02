"""Resident-only KV cache implementations and reporting helpers."""

import torch
from transformers import DynamicCache

from ollm.kv_cache.matrix import describe_kv_cache_strategy
from ollm.kv_cache.state import KVCacheStateSnapshot

_RESIDENT_POLICY_ID = "resident-baseline"
_EMPTY_CACHE_PLACEHOLDER = torch.empty(0)


def _tensor_pair_nbytes(tensors: tuple[torch.Tensor, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def build_resident_cache_snapshot(
    *,
    strategy_id: str,
    resident_layers: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> KVCacheStateSnapshot:
    """Build a truthful cache-state snapshot for resident-only strategies."""

    strategy_axes = describe_kv_cache_strategy(strategy_id)
    resident_tokens = sum(int(pair[0].shape[-2]) for pair in resident_layers)
    resident_bytes = sum(_tensor_pair_nbytes(pair) for pair in resident_layers)
    return KVCacheStateSnapshot(
        strategy_id=strategy_id,
        policy_id=_RESIDENT_POLICY_ID,
        persistence_format=strategy_axes.persistence_format,
        residency_mode=strategy_axes.residency_mode,
        window_policy=strategy_axes.window_policy,
        window_max_tokens=None,
        eviction_policy=None,
        cold_tier_encoding=strategy_axes.cold_tier_encoding,
        cold_tier_representation=None,
        persisted_layer_count=0,
        persisted_tokens=0,
        persisted_artifact_count=0,
        resident_layer_count=len(resident_layers),
        resident_tokens=resident_tokens,
        resident_bytes=resident_bytes,
        hot_layer_count=0,
        hot_tokens=0,
        hot_bytes=0,
        compaction_count=0,
        spill_count=0,
        spilled_tokens=0,
        eviction_count=0,
        evicted_tokens=0,
        cold_store_format=None,
    )


class ResidentKVCache(DynamicCache):
    """A fully resident KV cache with benchmark/reporting hooks."""

    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__()
        self._strategy_id = "resident"
        self._device = device
        self._resident_layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resident = self._resident_layers.get(layer_idx)
        if resident is not None:
            _prime_cache_layer(self, layer_idx, resident)
        out = super().update(key_states, value_states, layer_idx, cache_kwargs)
        self._resident_layers[layer_idx] = _to_resident_tensor_pair(out)
        self.layers[layer_idx].keys, self.layers[layer_idx].values = (
            _EMPTY_CACHE_PLACEHOLDER,
            _EMPTY_CACHE_PLACEHOLDER,
        )
        return out

    def cache_state_snapshot(self) -> KVCacheStateSnapshot:
        resident_layers = tuple(self._resident_layers.values())
        return build_resident_cache_snapshot(
            strategy_id=self._strategy_id,
            resident_layers=resident_layers,
        )


def _ensure_cache_layer_slot(cache: DynamicCache, layer_idx: int) -> None:
    if len(cache.layers) > layer_idx:
        return
    layer_factory = getattr(cache, "layer_class_to_replicate", None)
    if layer_factory is None:
        raise IndexError(
            "DynamicCache layer slot is missing and the cache cannot lazily create one"
        )
    while len(cache.layers) <= layer_idx:
        cache.layers.append(layer_factory())


def _prime_cache_layer(
    cache: DynamicCache,
    layer_idx: int,
    tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _ensure_cache_layer_slot(cache, layer_idx)
    layer = cache.layers[layer_idx]
    lazy_initialization = getattr(layer, "lazy_initialization", None)
    if callable(lazy_initialization):
        lazy_initialization(tensors[0], tensors[1])
    layer.keys, layer.values = tensors


def _to_resident_tensor_pair(
    tensors: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        tensors[0].detach().contiguous(),
        tensors[1].detach().contiguous(),
    )

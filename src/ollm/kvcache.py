import time
from pathlib import Path

import torch
from transformers import DynamicCache

from ollm.async_io import path_exists, path_mkdir, remove_tree
from ollm.kv_cache_policy import KVCachePolicy, select_kv_cache_policy
from ollm.kv_cache_store import ChunkedKVStore
from ollm.kv_cache_strategy import (
    DEFAULT_KV_CACHE_STRATEGY,
    kv_cache_root,
    normalize_kv_cache_strategy,
)
from ollm.kv_cache_streamed_store import StreamedSegmentedKVStore
from ollm.utils import Stats

_EMPTY_CACHE_PLACEHOLDER = torch.empty(0)


class oCache:
    def ini_ocache(
        self,
        cache_dir: str | Path,
        device: str | torch.device,
        stats: Stats | None,
        policy: KVCachePolicy | None = None,
        cache_strategy: str = DEFAULT_KV_CACHE_STRATEGY,
    ) -> None:
        if not cache_dir:
            raise ValueError(
                "cache_dir can not be empty. If you are trying to not use DiskCache, simply set past_key_values=None. This will use default DynamicCache"
            )
        normalized_strategy = normalize_kv_cache_strategy(cache_strategy)
        self.cache_strategy = (
            DEFAULT_KV_CACHE_STRATEGY
            if normalized_strategy is None
            else normalized_strategy
        )
        self.cache_folder = kv_cache_root(Path(cache_dir), self.cache_strategy)
        if path_exists(self.cache_folder):
            remove_tree(self.cache_folder)
        path_mkdir(self.cache_folder, parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.stats = stats
        self.policy = select_kv_cache_policy(self.device) if policy is None else policy
        self._pending_tails: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_store = _build_cache_store(self.cache_folder, self.cache_strategy)
        self._cache_store.initialize(self.policy.policy_id)

    def load_from_disk(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        started_at = time.perf_counter()
        tensors = self._cache_store.load_layer(layer_idx, device=self.device)
        if tensors is not None and self.stats is not None:
            self.stats.set("kvload", started_at)
        pending = self._pending_tails.get(layer_idx)
        if pending is None:
            return tensors
        if tensors is None:
            return pending
        return _concat_tensor_pairs(tensors, pending)

    def save_to_disk(
        self, tensors: tuple[torch.Tensor, torch.Tensor], layer_idx: int
    ) -> None:
        pending = self._pending_tails.get(layer_idx)
        buffered = (
            tensors if pending is None else _concat_tensor_pairs(pending, tensors)
        )
        if not self.policy.should_flush(
            pending_tokens=buffered[0].shape[-2],
            pending_bytes=_tensor_pair_nbytes(buffered),
        ):
            self._pending_tails[layer_idx] = buffered
            return
        started_at = time.perf_counter()
        self._cache_store.append_layer_chunk(layer_idx, buffered)
        self._pending_tails.pop(layer_idx, None)
        if self.stats is not None:
            self.stats.set("kvsave", started_at)


def _concat_tensor_pairs(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.cat((left[0], right[0]), dim=-2),
        torch.cat((left[1], right[1]), dim=-2),
    )


def _tensor_pair_nbytes(tensors: tuple[torch.Tensor, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _build_cache_store(
    cache_folder: Path, cache_strategy: str
) -> ChunkedKVStore | StreamedSegmentedKVStore:
    if cache_strategy == "chunked":
        return ChunkedKVStore(cache_folder)
    if cache_strategy == "streamed-segmented":
        return StreamedSegmentedKVStore(cache_folder)
    raise ValueError(f"Unsupported KV cache strategy: {cache_strategy}")


class KVCache(DynamicCache, oCache):
    def __init__(
        self,
        cache_dir: str | Path = "./kv_cache",
        device: str | torch.device = "cuda:0",
        stats: Stats | None = None,
        policy: KVCachePolicy | None = None,
        cache_strategy: str = DEFAULT_KV_CACHE_STRATEGY,
    ) -> None:
        super().__init__()
        self.ini_ocache(cache_dir, device, stats, policy, cache_strategy)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = self.load_from_disk(layer_idx)
        if tensors is not None:
            self.layers[layer_idx].keys, self.layers[layer_idx].values = tensors

        out = super().update(key_states, value_states, layer_idx, cache_kwargs)
        self.save_to_disk((key_states, value_states), layer_idx)
        self.layers[layer_idx].keys, self.layers[layer_idx].values = (
            _EMPTY_CACHE_PLACEHOLDER,
            _EMPTY_CACHE_PLACEHOLDER,
        )
        return out

import time
from pathlib import Path

import torch
from transformers import DynamicCache

from ollm.async_io import path_exists, path_mkdir, remove_tree
from ollm.kv_cache_store import ChunkedKVStore
from ollm.utils import Stats

_EMPTY_CACHE_PLACEHOLDER = torch.empty(0)


class oCache:
    def ini_ocache(
        self,
        cache_dir: str | Path,
        device: str | torch.device,
        stats: Stats | None,
    ) -> None:
        if not cache_dir:
            raise ValueError(
                "cache_dir can not be empty. If you are trying to not use DiskCache, simply set past_key_values=None. This will use default DynamicCache"
            )
        self.cache_folder = Path(cache_dir) / "kv_cache"
        if path_exists(self.cache_folder):
            remove_tree(self.cache_folder)
        path_mkdir(self.cache_folder, parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.stats = stats
        self._cache_store = ChunkedKVStore(self.cache_folder)
        self._cache_store.initialize()

    def load_from_disk(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        started_at = time.perf_counter()
        tensors = self._cache_store.load_layer(layer_idx, device=self.device)
        if tensors is not None and self.stats is not None:
            self.stats.set("kvload", started_at)
        return tensors

    def save_to_disk(
        self, tensors: tuple[torch.Tensor, torch.Tensor], layer_idx: int
    ) -> None:
        started_at = time.perf_counter()
        self._cache_store.append_layer_chunk(layer_idx, tensors)
        if self.stats is not None:
            self.stats.set("kvsave", started_at)


class KVCache(DynamicCache, oCache):
    def __init__(
        self,
        cache_dir: str | Path = "./kv_cache",
        device: str | torch.device = "cuda:0",
        stats: Stats | None = None,
    ) -> None:
        super().__init__()
        self.ini_ocache(cache_dir, device, stats)

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

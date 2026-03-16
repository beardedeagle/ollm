import time
from pathlib import Path

import torch
from transformers import DynamicCache

from ollm.async_io import (
    path_exists,
    path_mkdir,
    remove_tree,
    torch_load_file,
    torch_save_file,
)


class oCache:
    def ini_ocache(self, cache_dir, device, stats):
        if not cache_dir:
            raise ValueError(
                "cache_dir can not be empty. If you are trying to not use DiskCache, simply set past_key_values=None. This will use default DynamicCache"
            )
        self.cache_folder = Path(cache_dir) / "kv_cache"
        self.key_cache2, self.value_cache2 = [], []
        if path_exists(self.cache_folder):
            remove_tree(self.cache_folder)
        path_mkdir(self.cache_folder)
        self.device = device
        self.stats = stats

    def load_from_disk(self, layer_idx):
        path = self.cache_folder / f"layer_{layer_idx}.pt"
        if not path_exists(path):
            return None
        t1 = time.perf_counter()
        tensors = torch_load_file(path, map_location=self.device)
        if self.stats:
            self.stats.set("kvload", t1)
        return tensors

    def save_to_disk(self, tensors, layer_idx):
        t1 = time.perf_counter()
        path = self.cache_folder / f"layer_{layer_idx}.pt"
        tensors = (tensors[0].cpu(), tensors[1].cpu())
        torch_save_file(tensors, path)
        if self.stats:
            self.stats.set("kvsave", t1)


class KVCache(DynamicCache, oCache):  # DiskCache
    def __init__(self, cache_dir="./kv_cache", device="cuda:0", stats=None):
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
            if layer_idx < len(self.key_cache2):
                self.layers[layer_idx].keys = torch.cat(
                    [self.layers[layer_idx].keys, self.key_cache2[layer_idx]], dim=-2
                )
                self.layers[layer_idx].values = torch.cat(
                    [self.layers[layer_idx].values, self.value_cache2[layer_idx]],
                    dim=-2,
                )
                self.key_cache2[layer_idx] = torch.cat(
                    [self.key_cache2[layer_idx], key_states], dim=-2
                )
                self.value_cache2[layer_idx] = torch.cat(
                    [self.value_cache2[layer_idx], value_states], dim=-2
                )
            else:
                self.key_cache2.append(key_states)
                self.value_cache2.append(value_states)

        out = super().update(
            key_states, value_states, layer_idx, cache_kwargs
        )  # tuple of (self.key_cache[layer_idx], self.value_cache[layer_idx])
        if tensors is None:
            self.save_to_disk(
                out, layer_idx
            )  # save only first time cause it's slow to save
        self.layers[layer_idx].keys, self.layers[layer_idx].values = (
            torch.empty(0),
            torch.empty(0),
        )
        return out

import time
from pathlib import Path
from typing import Protocol

import torch
from transformers import DynamicCache

from ollm.async_io import path_exists, path_mkdir, remove_tree
from ollm.kv_cache_journal_store import JournaledKVStore
from ollm.kv_cache_policy import KVCachePolicy, select_kv_cache_policy
from ollm.kv_cache_state import KVCacheStateSnapshot
from ollm.kv_cache_store import ChunkedKVStore
from ollm.kv_cache_strategy import (
    DEFAULT_KV_CACHE_STRATEGY,
    kv_cache_root,
    normalize_kv_cache_strategy,
)
from ollm.kv_cache_streamed_store import StreamedSegmentedKVStore
from ollm.kv_cache_tiered_store import TieredWriteBackKVStore
from ollm.utils import Stats

_EMPTY_CACHE_PLACEHOLDER = torch.empty(0)


class _KVCacheStoreProtocol(Protocol):
    def initialize(self, policy_id: str) -> None: ...

    def load_layer(
        self, layer_idx: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None: ...

    def append_layer_chunk(
        self, layer_idx: int, tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None: ...

    def persisted_layer_ids(self) -> tuple[int, ...]: ...

    def persisted_token_count(self) -> int: ...

    def persisted_artifact_count(self) -> int: ...

    def cold_store_format_id(self) -> str | None: ...

    def compaction_count(self) -> int: ...

    def consume_last_compaction_elapsed_seconds(self) -> float | None: ...


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
        self.policy = (
            select_kv_cache_policy(self.device, strategy=self.cache_strategy)
            if policy is None
            else policy
        )
        self._resident_layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._pending_tails: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._hot_tails: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._spill_count = 0
        self._spilled_tokens = 0
        self._cache_store = _build_cache_store(
            self.cache_folder, self.cache_strategy, self.policy
        )
        self._cache_store.initialize(self.policy.policy_id)

    def load_from_disk(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        resident = self._resident_layers.get(layer_idx)
        if resident is not None:
            return _resident_layer_for_device(resident, self.device)
        tensors = self._load_cold_layer(layer_idx)
        pending = self._resident_tail(layer_idx)
        if pending is None:
            return tensors
        if tensors is None:
            return pending
        return _concat_tensor_pairs(tensors, pending)

    def save_to_disk(
        self, tensors: tuple[torch.Tensor, torch.Tensor], layer_idx: int
    ) -> None:
        if self.cache_strategy == "tiered-write-back":
            self._save_tiered_write_back(tensors, layer_idx)
            return
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
        self._record_cache_write_metrics(started_at)

    def cache_state_snapshot(self) -> KVCacheStateSnapshot:
        hot_tails = (
            self._hot_tails
            if self.cache_strategy == "tiered-write-back"
            else self._pending_tails
        )
        return KVCacheStateSnapshot(
            strategy_id=self.cache_strategy,
            policy_id=self.policy.policy_id,
            persisted_layer_count=len(self._cache_store.persisted_layer_ids()),
            persisted_tokens=self._cache_store.persisted_token_count(),
            persisted_artifact_count=self._cache_store.persisted_artifact_count(),
            hot_layer_count=len(hot_tails),
            hot_tokens=sum(_sequence_length(pair[0]) for pair in hot_tails.values()),
            hot_bytes=sum(_tensor_pair_nbytes(pair) for pair in hot_tails.values()),
            compaction_count=self._cache_store.compaction_count(),
            spill_count=self._spill_count,
            spilled_tokens=self._spilled_tokens,
            cold_store_format=self._cache_store.cold_store_format_id(),
        )

    def _resident_tail(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.cache_strategy == "tiered-write-back":
            return self._hot_tails.get(layer_idx)
        return self._pending_tails.get(layer_idx)

    def _save_tiered_write_back(
        self, tensors: tuple[torch.Tensor, torch.Tensor], layer_idx: int
    ) -> None:
        pending = self._hot_tails.get(layer_idx)
        buffered = (
            tensors if pending is None else _concat_tensor_pairs(pending, tensors)
        )
        spill_tokens = self.policy.write_back_spill_token_count(
            pending_tokens=_sequence_length(buffered[0]),
            pending_bytes=_tensor_pair_nbytes(buffered),
        )
        if spill_tokens <= 0:
            self._hot_tails[layer_idx] = buffered
            return

        cold_tensors, hot_tensors = _split_tensor_pair(buffered, spill_tokens)
        started_at = time.perf_counter()
        self._cache_store.append_layer_chunk(layer_idx, cold_tensors)
        self._spill_count += 1
        self._spilled_tokens += spill_tokens
        if hot_tensors is None:
            self._hot_tails.pop(layer_idx, None)
        else:
            self._hot_tails[layer_idx] = hot_tensors
        self._record_cache_write_metrics(started_at)

    def _load_cold_layer(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        started_at = time.perf_counter()
        tensors = self._cache_store.load_layer(layer_idx, device=self.device)
        if tensors is not None and self.stats is not None:
            self.stats.set("kvload", started_at)
        return tensors

    def _remember_resident_layer(
        self,
        layer_idx: int,
        tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self._resident_layers[layer_idx] = _to_resident_tensor_pair(
            tensors, device=self.device
        )

    def _record_cache_write_metrics(self, started_at: float) -> None:
        elapsed_seconds = self._cache_store.consume_last_compaction_elapsed_seconds()
        if self.stats is None:
            return
        total_elapsed_seconds = time.perf_counter() - started_at
        append_elapsed_seconds = total_elapsed_seconds
        if elapsed_seconds is not None:
            append_elapsed_seconds = max(0.0, total_elapsed_seconds - elapsed_seconds)
            self.stats.record_elapsed_seconds("kvcompact", elapsed_seconds)
        self.stats.record_elapsed_seconds("kvsave", append_elapsed_seconds)


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


def _to_cpu_tensor_pair(
    tensors: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(tensor.detach().cpu().contiguous() for tensor in tensors)  # type: ignore[return-value]


def _to_resident_tensor_pair(
    tensors: tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if device.type == "cpu":
        return tuple(tensor.detach().contiguous() for tensor in tensors)  # type: ignore[return-value]
    return _to_cpu_tensor_pair(tensors)


def _move_tensor_pair(
    tensors: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(tensor.to(device) for tensor in tensors)  # type: ignore[return-value]


def _resident_layer_for_device(
    tensors: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if device.type == "cpu":
        return tensors
    return _move_tensor_pair(tensors, device)


def _build_cache_store(
    cache_folder: Path, cache_strategy: str, policy: KVCachePolicy
) -> _KVCacheStoreProtocol:
    if cache_strategy == "chunked":
        return ChunkedKVStore(cache_folder)
    if cache_strategy == "streamed-segmented":
        return StreamedSegmentedKVStore(cache_folder)
    if cache_strategy == "log-structured-journal":
        return JournaledKVStore(
            cache_folder,
            compaction_entry_threshold=policy.journal_compaction_entry_threshold,
        )
    if cache_strategy == "tiered-write-back":
        return TieredWriteBackKVStore(cache_folder)
    raise ValueError(f"Unsupported KV cache strategy: {cache_strategy}")


def _sequence_length(tensor: torch.Tensor) -> int:
    return int(tensor.shape[-2])


def _split_tensor_pair(
    tensors: tuple[torch.Tensor, torch.Tensor],
    split_tokens: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor] | None,
]:
    total_tokens = _sequence_length(tensors[0])
    if split_tokens <= 0 or split_tokens > total_tokens:
        raise ValueError("split_tokens must stay within the tensor sequence length")
    prefix = (
        tensors[0][..., :split_tokens, :].contiguous(),
        tensors[1][..., :split_tokens, :].contiguous(),
    )
    if split_tokens == total_tokens:
        return prefix, None
    suffix = (
        tensors[0][..., split_tokens:, :].contiguous(),
        tensors[1][..., split_tokens:, :].contiguous(),
    )
    return prefix, suffix


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
        self._remember_resident_layer(layer_idx, out)
        self.save_to_disk((key_states, value_states), layer_idx)
        self.layers[layer_idx].keys, self.layers[layer_idx].values = (
            _EMPTY_CACHE_PLACEHOLDER,
            _EMPTY_CACHE_PLACEHOLDER,
        )
        return out

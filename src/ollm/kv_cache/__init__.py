"""Shared KV cache implementations and helpers."""

import time
from pathlib import Path
from typing import Protocol

import torch
from transformers import DynamicCache

from ollm.async_io import path_exists, path_mkdir, remove_tree
from ollm.kv_cache.journal_store import JournaledKVStore
from ollm.kv_cache.matrix import (
    DEFAULT_KV_CACHE_LIFECYCLE,
    describe_kv_cache_strategy,
    normalize_kv_cache_lifecycle,
    resolve_kv_cache_eviction_policy,
    resolve_kv_cache_window_tokens,
)
from ollm.kv_cache.paged_store import PagedKVStore
from ollm.kv_cache.policy import KVCachePolicy, select_kv_cache_policy
from ollm.kv_cache.quantized_store import QuantizedJournaledKVStore
from ollm.kv_cache.sliding_window_store import SlidingWindowRingBufferKVStore
from ollm.kv_cache.state import KVCacheStateSnapshot
from ollm.kv_cache.store import ChunkedKVStore
from ollm.kv_cache.strategy import (
    DEFAULT_KV_CACHE_STRATEGY,
    kv_cache_root,
    normalize_kv_cache_strategy,
)
from ollm.kv_cache.streamed_store import StreamedSegmentedKVStore
from ollm.kv_cache.tiered_store import TieredWriteBackKVStore
from ollm.utils import Stats

__all__ = ["KVCache", "oCache"]

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

    def cold_tier_representation_id(self) -> str | None: ...

    def compaction_count(self) -> int: ...

    def eviction_count(self) -> int: ...

    def evicted_token_count(self) -> int: ...

    def consume_last_compaction_elapsed_seconds(self) -> float | None: ...


class oCache:
    def ini_ocache(
        self,
        cache_dir: str | Path,
        device: str | torch.device,
        stats: Stats | None,
        policy: KVCachePolicy | None = None,
        cache_strategy: str = DEFAULT_KV_CACHE_STRATEGY,
        cache_lifecycle: str = DEFAULT_KV_CACHE_LIFECYCLE,
        cache_window_tokens: int | None = None,
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
        normalized_lifecycle = normalize_kv_cache_lifecycle(cache_lifecycle)
        self.cache_lifecycle = (
            DEFAULT_KV_CACHE_LIFECYCLE
            if normalized_lifecycle is None
            else normalized_lifecycle
        )
        self.cache_window_tokens = resolve_kv_cache_window_tokens(
            self.cache_strategy,
            cache_window_tokens,
        )
        self.cache_folder = kv_cache_root(Path(cache_dir), self.cache_strategy)
        root_manifest_path = self.cache_folder / "manifest.json"
        cache_root_exists = path_exists(self.cache_folder)
        root_manifest_exists = path_exists(root_manifest_path)
        if self.cache_lifecycle == "runtime-scoped" and cache_root_exists:
            remove_tree(self.cache_folder)
            cache_root_exists = False
            root_manifest_exists = False
        if (
            self.cache_lifecycle == "persistent"
            and cache_root_exists
            and not root_manifest_exists
        ):
            remove_tree(self.cache_folder)
            cache_root_exists = False
            root_manifest_exists = False
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
            self.cache_folder,
            self.cache_strategy,
            self.policy,
            self.cache_window_tokens,
        )
        if self.cache_lifecycle == "runtime-scoped" or not root_manifest_exists:
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
        if self.cache_strategy == "sliding-window-ring-buffer":
            self._save_sliding_window(tensors, layer_idx)
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
        if self.cache_strategy == "sliding-window-ring-buffer":
            hot_tails = {}
        strategy_axes = describe_kv_cache_strategy(self.cache_strategy)
        return KVCacheStateSnapshot(
            strategy_id=self.cache_strategy,
            policy_id=self.policy.policy_id,
            persistence_format=strategy_axes.persistence_format,
            residency_mode=strategy_axes.residency_mode,
            window_policy=strategy_axes.window_policy,
            window_max_tokens=(
                self.cache_window_tokens
                if strategy_axes.window_policy == "sliding-window"
                else None
            ),
            eviction_policy=resolve_kv_cache_eviction_policy(self.cache_strategy),
            cold_tier_encoding=strategy_axes.cold_tier_encoding,
            cold_tier_representation=self._cache_store.cold_tier_representation_id(),
            persisted_layer_count=len(self._cache_store.persisted_layer_ids()),
            persisted_tokens=self._cache_store.persisted_token_count(),
            persisted_artifact_count=self._cache_store.persisted_artifact_count(),
            resident_layer_count=len(self._resident_layers),
            resident_tokens=sum(
                _sequence_length(pair[0]) for pair in self._resident_layers.values()
            ),
            resident_bytes=sum(
                _tensor_pair_nbytes(pair) for pair in self._resident_layers.values()
            ),
            hot_layer_count=len(hot_tails),
            hot_tokens=sum(_sequence_length(pair[0]) for pair in hot_tails.values()),
            hot_bytes=sum(_tensor_pair_nbytes(pair) for pair in hot_tails.values()),
            compaction_count=self._cache_store.compaction_count(),
            spill_count=self._spill_count,
            spilled_tokens=self._spilled_tokens,
            eviction_count=self._cache_store.eviction_count(),
            evicted_tokens=self._cache_store.evicted_token_count(),
            cold_store_format=self._cache_store.cold_store_format_id(),
        )

    def _resident_tail(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.cache_strategy == "sliding-window-ring-buffer":
            return None
        if self.cache_strategy == "tiered-write-back":
            return self._hot_tails.get(layer_idx)
        return self._pending_tails.get(layer_idx)

    def _save_sliding_window(
        self, tensors: tuple[torch.Tensor, torch.Tensor], layer_idx: int
    ) -> None:
        started_at = time.perf_counter()
        self._cache_store.append_layer_chunk(layer_idx, tensors)
        self._record_cache_write_metrics(started_at)

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

    def _finalize_updated_tensors(
        self,
        layer_idx: int,
        full_tensors: tuple[torch.Tensor, torch.Tensor],
        delta_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_strategy == "sliding-window-ring-buffer":
            if self.cache_window_tokens is None:
                raise ValueError(
                    "sliding-window-ring-buffer requires an explicit resolved window"
                )
            bounded_tensors = _trim_tensor_pair_to_recent_window(
                full_tensors, self.cache_window_tokens
            )
            self._remember_resident_layer(layer_idx, bounded_tensors)
            self.save_to_disk(delta_tensors, layer_idx)
            return bounded_tensors
        self._remember_resident_layer(layer_idx, full_tensors)
        self.save_to_disk(delta_tensors, layer_idx)
        return full_tensors


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


def _build_cache_store(
    cache_folder: Path,
    cache_strategy: str,
    policy: KVCachePolicy,
    cache_window_tokens: int | None,
) -> _KVCacheStoreProtocol:
    if cache_strategy == "chunked":
        return ChunkedKVStore(cache_folder)
    if cache_strategy == "paged":
        return PagedKVStore(cache_folder)
    if cache_strategy == "streamed-segmented":
        return StreamedSegmentedKVStore(cache_folder)
    if cache_strategy == "log-structured-journal":
        return JournaledKVStore(
            cache_folder,
            compaction_entry_threshold=policy.journal_compaction_entry_threshold,
        )
    if cache_strategy == "quantized-cold-tier":
        return QuantizedJournaledKVStore(
            cache_folder,
            compaction_entry_threshold=policy.journal_compaction_entry_threshold,
        )
    if cache_strategy == "sliding-window-ring-buffer":
        if cache_window_tokens is None:
            raise ValueError(
                "sliding-window-ring-buffer requires an explicit resolved window"
            )
        return SlidingWindowRingBufferKVStore(
            cache_folder,
            window_max_tokens=cache_window_tokens,
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


def _trim_tensor_pair_to_recent_window(
    tensors: tuple[torch.Tensor, torch.Tensor],
    window_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_tokens = _sequence_length(tensors[0])
    if total_tokens <= window_tokens:
        return tensors
    return (
        tensors[0][..., -window_tokens:, :].contiguous(),
        tensors[1][..., -window_tokens:, :].contiguous(),
    )


class KVCache(DynamicCache, oCache):
    def __init__(
        self,
        cache_dir: str | Path = "./kv_cache",
        device: str | torch.device = "cuda:0",
        stats: Stats | None = None,
        policy: KVCachePolicy | None = None,
        cache_strategy: str = DEFAULT_KV_CACHE_STRATEGY,
        cache_lifecycle: str = DEFAULT_KV_CACHE_LIFECYCLE,
        cache_window_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.ini_ocache(
            cache_dir,
            device,
            stats,
            policy,
            cache_strategy,
            cache_lifecycle,
            cache_window_tokens,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = self.load_from_disk(layer_idx)
        if tensors is not None:
            _prime_cache_layer(self, layer_idx, tensors)

        out = super().update(key_states, value_states, layer_idx, cache_kwargs)
        bounded = self._finalize_updated_tensors(
            layer_idx,
            out,
            (key_states, value_states),
        )
        self.layers[layer_idx].keys, self.layers[layer_idx].values = (
            _EMPTY_CACHE_PLACEHOLDER,
            _EMPTY_CACHE_PLACEHOLDER,
        )
        return bounded

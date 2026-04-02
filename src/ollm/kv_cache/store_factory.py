"""KV cache store protocol and factory helpers."""

from pathlib import Path
from typing import Protocol

import torch

from ollm.kv_cache.journal_store import JournaledKVStore
from ollm.kv_cache.paged_store import PagedKVStore
from ollm.kv_cache.policy import KVCachePolicy
from ollm.kv_cache.quantized_store import QuantizedJournaledKVStore
from ollm.kv_cache.sliding_window_store import SlidingWindowRingBufferKVStore
from ollm.kv_cache.store import ChunkedKVStore
from ollm.kv_cache.streamed_store import StreamedSegmentedKVStore
from ollm.kv_cache.tiered_store import TieredWriteBackKVStore


class KVCacheStoreProtocol(Protocol):
    """Minimal protocol shared by disk-backed KV cache stores."""

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


def build_cache_store(
    cache_folder: Path,
    cache_strategy: str,
    policy: KVCachePolicy,
    cache_window_tokens: int | None,
) -> KVCacheStoreProtocol:
    """Construct the concrete cold-store implementation for a KV strategy.

    Args:
        cache_folder (Path): Cache directory for the store implementation.
        cache_strategy (str): Resolved KV cache strategy ID.
        policy (KVCachePolicy): Active KV cache policy.
        cache_window_tokens (int | None): Sliding-window token budget.

    Returns:
        KVCacheStoreProtocol: Concrete store implementation for the strategy.
    """
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

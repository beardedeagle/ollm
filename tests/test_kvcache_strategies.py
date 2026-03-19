import json
from pathlib import Path
from typing import cast

import pytest
import torch

from ollm.kv_cache_policy import KVCachePolicy
from ollm.kv_cache_strategy import kv_cache_root
from ollm.kvcache import KVCache
from ollm.utils import Stats


def _chunk_tensor(token_count: int, offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset, offset + (token_count * 8), dtype=torch.float32)
    return values.reshape(1, 2, token_count, 4)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _immediate_flush_policy() -> KVCachePolicy:
    return KVCachePolicy(
        policy_id="test-immediate-flush",
        flush_token_threshold=1,
        flush_byte_threshold=1,
    )


@pytest.mark.parametrize(
    "cache_strategy",
    [
        "chunked",
        "streamed-segmented",
        "log-structured-journal",
        "tiered-write-back",
    ],
)
def test_kvcache_strategies_use_separate_roots(
    tmp_path: Path, cache_strategy: str
) -> None:
    base_cache_root = tmp_path / "cache-root"
    cache = KVCache(
        cache_dir=base_cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy=cache_strategy,
    )
    cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=100), 0)

    assert cache.cache_folder == kv_cache_root(base_cache_root, cache_strategy)
    assert cache.cache_folder.exists()


def test_kvcache_strategy_roots_do_not_cross_contaminate(tmp_path: Path) -> None:
    base_cache_root = tmp_path / "cache-root"
    chunked_cache = KVCache(
        cache_dir=base_cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="chunked",
    )
    streamed_cache = KVCache(
        cache_dir=base_cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="streamed-segmented",
    )
    journal_cache = KVCache(
        cache_dir=base_cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="log-structured-journal",
    )
    tiered_cache = KVCache(
        cache_dir=base_cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="tiered-write-back",
    )

    chunked_cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=100), 0)
    streamed_cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=200), 0)
    journal_cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=250), 0)
    tiered_cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=300), 0)

    assert chunked_cache.cache_folder != streamed_cache.cache_folder
    assert chunked_cache.cache_folder != journal_cache.cache_folder
    assert streamed_cache.cache_folder != tiered_cache.cache_folder
    assert streamed_cache.cache_folder != journal_cache.cache_folder
    assert journal_cache.cache_folder != tiered_cache.cache_folder
    assert chunked_cache.cache_folder != tiered_cache.cache_folder


def test_log_structured_journal_kvcache_compacts_after_threshold(
    tmp_path: Path,
) -> None:
    stats = Stats()
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=stats,
        policy=_immediate_flush_policy(),
        cache_strategy="log-structured-journal",
    )

    for index in range(8):
        cache.update(
            _chunk_tensor(1, offset=index * 1000),
            _chunk_tensor(1, offset=(index * 1000) + 100),
            0,
        )

    state = cache.cache_state_snapshot()
    summary = stats.collect_and_clear_ms()
    layer_manifest = _read_json(cache.cache_folder / "layers" / "0" / "manifest.json")
    entries = cast(list[dict[str, object]], layer_manifest["entries"])

    assert state.strategy_id == "log-structured-journal"
    assert state.compaction_count == 1
    assert state.persisted_tokens == 8
    assert state.persisted_artifact_count == 1
    assert layer_manifest["compaction_count"] == 1
    assert len(entries) == 1
    assert "kvcompact" in summary


@pytest.mark.parametrize(
    "cache_strategy",
    [
        "chunked",
        "streamed-segmented",
        "log-structured-journal",
        "tiered-write-back",
    ],
)
def test_kvcache_resident_layer_stays_on_cpu_for_accelerator_devices(
    tmp_path: Path, cache_strategy: str
) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cuda:0",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy=cache_strategy,
    )

    cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=100), 0)

    resident = cache._resident_layers[0]
    assert resident[0].device.type == "cpu"
    assert resident[1].device.type == "cpu"

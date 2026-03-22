import json
from pathlib import Path
from typing import cast

import pytest
import torch

from ollm.kv_cache import KVCache
from ollm.kv_cache.policy import KVCachePolicy
from ollm.kv_cache.strategy import kv_cache_root
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
        "paged",
        "streamed-segmented",
        "log-structured-journal",
        "sliding-window-ring-buffer",
        "quantized-cold-tier",
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
    paged_cache = KVCache(
        cache_dir=base_cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="paged",
    )
    journal_cache = KVCache(
        cache_dir=base_cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="log-structured-journal",
    )
    sliding_window_cache = KVCache(
        cache_dir=base_cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="sliding-window-ring-buffer",
        cache_window_tokens=4,
    )
    quantized_cache = KVCache(
        cache_dir=base_cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="quantized-cold-tier",
    )
    tiered_cache = KVCache(
        cache_dir=base_cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="tiered-write-back",
    )

    chunked_cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=100), 0)
    paged_cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=150), 0)
    streamed_cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=200), 0)
    journal_cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=250), 0)
    sliding_window_cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=260), 0)
    quantized_cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=275), 0)
    tiered_cache.update(_chunk_tensor(2), _chunk_tensor(2, offset=300), 0)

    assert chunked_cache.cache_folder != paged_cache.cache_folder
    assert chunked_cache.cache_folder != streamed_cache.cache_folder
    assert chunked_cache.cache_folder != journal_cache.cache_folder
    assert chunked_cache.cache_folder != quantized_cache.cache_folder
    assert paged_cache.cache_folder != streamed_cache.cache_folder
    assert paged_cache.cache_folder != journal_cache.cache_folder
    assert paged_cache.cache_folder != quantized_cache.cache_folder
    assert paged_cache.cache_folder != tiered_cache.cache_folder
    assert paged_cache.cache_folder != sliding_window_cache.cache_folder
    assert streamed_cache.cache_folder != tiered_cache.cache_folder
    assert streamed_cache.cache_folder != journal_cache.cache_folder
    assert streamed_cache.cache_folder != quantized_cache.cache_folder
    assert streamed_cache.cache_folder != sliding_window_cache.cache_folder
    assert journal_cache.cache_folder != quantized_cache.cache_folder
    assert journal_cache.cache_folder != tiered_cache.cache_folder
    assert journal_cache.cache_folder != sliding_window_cache.cache_folder
    assert quantized_cache.cache_folder != tiered_cache.cache_folder
    assert quantized_cache.cache_folder != sliding_window_cache.cache_folder
    assert chunked_cache.cache_folder != tiered_cache.cache_folder
    assert chunked_cache.cache_folder != sliding_window_cache.cache_folder
    assert sliding_window_cache.cache_folder != tiered_cache.cache_folder


def test_log_structured_journal_kvcache_compacts_after_threshold(
    tmp_path: Path,
) -> None:
    stats = Stats()
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=stats,
        policy=KVCachePolicy(
            policy_id="test-journal-flush",
            flush_token_threshold=1,
            flush_byte_threshold=1,
            journal_compaction_entry_threshold=4,
        ),
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
    assert state.cold_store_format == "ollm-kv-journal"
    assert state.compaction_count == 2
    assert state.persisted_tokens == 8
    assert state.persisted_artifact_count == 2
    assert layer_manifest["compaction_count"] == 2
    assert len(entries) == 2
    assert "kvsave" in summary
    assert "kvcompact" in summary


def test_quantized_cold_tier_kvcache_reports_quantized_state(tmp_path: Path) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=KVCachePolicy(
            policy_id="test-quantized-flush",
            flush_token_threshold=1,
            flush_byte_threshold=1,
            journal_compaction_entry_threshold=0,
        ),
        cache_strategy="quantized-cold-tier",
    )
    first_key = _chunk_tensor(3)
    first_value = _chunk_tensor(3, offset=100)
    second_key = _chunk_tensor(2, offset=1000)
    second_value = _chunk_tensor(2, offset=2000)

    cache.update(first_key, first_value, 0)
    out = cache.update(second_key, second_value, 0)
    persisted = cache.load_from_disk(0)
    state = cache.cache_state_snapshot()

    expected_key = torch.cat((first_key, second_key), dim=-2)
    expected_value = torch.cat((first_value, second_value), dim=-2)

    assert persisted is not None
    assert torch.allclose(out[0], expected_key, atol=8.0, rtol=0.05)
    assert torch.allclose(out[1], expected_value, atol=16.0, rtol=0.05)
    assert torch.allclose(persisted[0], expected_key, atol=8.0, rtol=0.05)
    assert torch.allclose(persisted[1], expected_value, atol=16.0, rtol=0.05)
    assert state.strategy_id == "quantized-cold-tier"
    assert state.cold_tier_encoding == "quantized"
    assert state.cold_tier_representation == "int8-symmetric-per-tensor"
    assert state.cold_store_format == "ollm-kv-journal-quantized"


def test_paged_kvcache_reports_paged_state(tmp_path: Path) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="paged",
    )
    first_key = _chunk_tensor(3)
    first_value = _chunk_tensor(3, offset=100)
    second_key = _chunk_tensor(2, offset=1000)
    second_value = _chunk_tensor(2, offset=2000)

    cache.update(first_key, first_value, 0)
    out = cache.update(second_key, second_value, 0)
    persisted = cache.load_from_disk(0)
    state = cache.cache_state_snapshot()

    expected_key = torch.cat((first_key, second_key), dim=-2)
    expected_value = torch.cat((first_value, second_value), dim=-2)

    assert persisted is not None
    assert torch.equal(out[0], expected_key)
    assert torch.equal(out[1], expected_value)
    assert torch.equal(persisted[0], expected_key)
    assert torch.equal(persisted[1], expected_value)
    assert state.strategy_id == "paged"
    assert state.persistence_format == "paged-manifest"
    assert state.cold_store_format == "ollm-kv-paged"
    assert state.cold_tier_encoding == "full-precision"
    assert state.persisted_artifact_count == 1


def test_sliding_window_ring_buffer_bounds_growth_and_tracks_eviction(
    tmp_path: Path,
) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="sliding-window-ring-buffer",
        cache_window_tokens=4,
    )
    first_key = _chunk_tensor(3)
    first_value = _chunk_tensor(3, offset=100)
    second_key = _chunk_tensor(3, offset=1000)
    second_value = _chunk_tensor(3, offset=2000)

    cache.update(first_key, first_value, 0)
    out = cache.update(second_key, second_value, 0)
    persisted = cache.load_from_disk(0)
    state = cache.cache_state_snapshot()

    expected_key = torch.cat((first_key, second_key), dim=-2)[..., -4:, :]
    expected_value = torch.cat((first_value, second_value), dim=-2)[..., -4:, :]

    assert persisted is not None
    assert torch.equal(out[0], expected_key)
    assert torch.equal(out[1], expected_value)
    assert torch.equal(persisted[0], expected_key)
    assert torch.equal(persisted[1], expected_value)
    assert state.strategy_id == "sliding-window-ring-buffer"
    assert state.window_policy == "sliding-window"
    assert state.window_max_tokens == 4
    assert state.eviction_policy == "drop-oldest"
    assert state.persisted_tokens == 4
    assert state.persisted_artifact_count == 1
    assert state.eviction_count == 1
    assert state.evicted_tokens == 2
    assert state.cold_store_format == "ollm-kv-sliding-window"


def test_sliding_window_ring_buffer_bounds_oversized_first_append(
    tmp_path: Path,
) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="sliding-window-ring-buffer",
        cache_window_tokens=4,
    )
    key_states = _chunk_tensor(6)
    value_states = _chunk_tensor(6, offset=100)

    out = cache.update(key_states, value_states, 0)
    persisted = cache.load_from_disk(0)
    state = cache.cache_state_snapshot()

    expected_key = key_states[..., -4:, :]
    expected_value = value_states[..., -4:, :]

    assert persisted is not None
    assert torch.equal(out[0], expected_key)
    assert torch.equal(out[1], expected_value)
    assert torch.equal(persisted[0], expected_key)
    assert torch.equal(persisted[1], expected_value)
    assert state.persisted_tokens == 4
    assert state.eviction_count == 1
    assert state.evicted_tokens == 2


@pytest.mark.parametrize(
    "cache_strategy",
    [
        "chunked",
        "paged",
        "streamed-segmented",
        "log-structured-journal",
        "sliding-window-ring-buffer",
        "quantized-cold-tier",
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

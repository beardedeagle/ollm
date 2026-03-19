import json
from pathlib import Path

import torch

from ollm.kv_cache_policy import KVCachePolicy
from ollm.kvcache import KVCache
from ollm.utils import Stats


def _chunk_tensor(token_count: int, offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset, offset + (token_count * 8), dtype=torch.float32)
    return values.reshape(1, 2, token_count, 4)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tiered_policy() -> KVCachePolicy:
    return KVCachePolicy(
        policy_id="test-tiered-write-back",
        flush_token_threshold=6,
        flush_byte_threshold=1_000_000,
        write_back_retained_tokens=3,
        write_back_retained_bytes=1_000_000,
    )


def test_tiered_write_back_spills_oldest_prefix_and_keeps_hot_tail(
    tmp_path: Path,
) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_tiered_policy(),
        cache_strategy="tiered-write-back",
    )
    first_key = _chunk_tensor(3)
    first_value = _chunk_tensor(3, offset=100)
    second_key = _chunk_tensor(4, offset=1000)
    second_value = _chunk_tensor(4, offset=2000)

    cache.update(first_key, first_value, 0)
    out = cache.update(second_key, second_value, 0)
    persisted = cache.load_from_disk(0)
    assert persisted is not None

    expected_key = torch.cat((first_key, second_key), dim=-2)
    expected_value = torch.cat((first_value, second_value), dim=-2)
    state = cache.cache_state_snapshot()

    assert torch.equal(out[0], expected_key)
    assert torch.equal(out[1], expected_value)
    assert torch.equal(persisted[0], expected_key)
    assert torch.equal(persisted[1], expected_value)
    assert state.strategy_id == "tiered-write-back"
    assert state.policy_id == "test-tiered-write-back"
    assert state.persisted_tokens == 4
    assert state.hot_tokens == 3
    assert state.spill_count == 1
    assert state.spilled_tokens == 4

    root_manifest = _read_json(cache.cache_folder / "manifest.json")
    cold_root_manifest = _read_json(cache.cache_folder / "cold" / "manifest.json")
    cold_layer_manifest = _read_json(
        cache.cache_folder / "cold" / "layers" / "0" / "manifest.json"
    )

    assert root_manifest["format"] == "ollm-kv-tiered-write-back"
    assert root_manifest["cold_store_root"] == "cold"
    assert cold_root_manifest["format"] == "ollm-kv-chunked"
    assert cold_layer_manifest["persisted_tokens"] == 4


def test_tiered_write_back_round_trips_across_repeated_updates(tmp_path: Path) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_tiered_policy(),
        cache_strategy="tiered-write-back",
    )
    first_key = _chunk_tensor(3)
    first_value = _chunk_tensor(3, offset=100)
    second_key = _chunk_tensor(4, offset=1000)
    second_value = _chunk_tensor(4, offset=2000)
    third_key = _chunk_tensor(3, offset=3000)
    third_value = _chunk_tensor(3, offset=4000)

    cache.update(first_key, first_value, 0)
    cache.update(second_key, second_value, 0)
    out = cache.update(third_key, third_value, 0)
    persisted = cache.load_from_disk(0)
    assert persisted is not None

    expected_key = torch.cat((first_key, second_key, third_key), dim=-2)
    expected_value = torch.cat((first_value, second_value, third_value), dim=-2)
    state = cache.cache_state_snapshot()

    assert torch.equal(out[0], expected_key)
    assert torch.equal(out[1], expected_value)
    assert torch.equal(persisted[0], expected_key)
    assert torch.equal(persisted[1], expected_value)
    assert state.persisted_tokens == 7
    assert state.hot_tokens == 3
    assert state.spill_count == 2
    assert state.spilled_tokens == 7


def test_tiered_write_back_caches_cold_layer_after_first_load(tmp_path: Path) -> None:
    stats = Stats()
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=stats,
        policy=_tiered_policy(),
        cache_strategy="tiered-write-back",
    )

    cache.update(_chunk_tensor(3), _chunk_tensor(3, offset=100), 0)
    cache.update(_chunk_tensor(4, offset=1000), _chunk_tensor(4, offset=2000), 0)

    stats.clear()
    first = cache.load_from_disk(0)
    second = cache.load_from_disk(0)
    summary = stats.collect_and_clear_ms()

    assert first is not None
    assert second is not None
    assert torch.equal(first[0], second[0])
    assert "kvload" not in summary

from pathlib import Path

import torch

from ollm.kv_cache_policy import KVCachePolicy
from ollm.kvcache import KVCache


def _immediate_flush_policy() -> KVCachePolicy:
    return KVCachePolicy(
        policy_id="test-immediate-flush",
        flush_token_threshold=1,
        flush_byte_threshold=1,
    )


def test_quantized_kvcache_persistent_lifecycle_reloads_dequantized_layer(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache-root"
    key_states = torch.linspace(-1.0, 1.0, steps=24, dtype=torch.float32).reshape(
        1, 2, 3, 4
    )
    value_states = torch.linspace(-0.5, 0.75, steps=24, dtype=torch.float32).reshape(
        1, 2, 3, 4
    )
    first_cache = KVCache(
        cache_dir=cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="quantized-cold-tier",
        cache_lifecycle="persistent",
    )
    first_cache.update(key_states, value_states, 0)

    second_cache = KVCache(
        cache_dir=cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="quantized-cold-tier",
        cache_lifecycle="persistent",
    )
    persisted = second_cache.load_from_disk(0)
    state = second_cache.cache_state_snapshot()

    assert persisted is not None
    assert persisted[0].dtype == key_states.dtype
    assert persisted[1].dtype == value_states.dtype
    assert torch.allclose(persisted[0], key_states, atol=0.02, rtol=0.05)
    assert torch.allclose(persisted[1], value_states, atol=0.02, rtol=0.05)
    assert state.cold_tier_encoding == "quantized"
    assert state.cold_tier_representation == "int8-symmetric-per-tensor"
    assert state.cold_store_format == "ollm-kv-journal-quantized"


def test_quantized_kvcache_reopened_update_reuses_persisted_layer(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache-root"
    first_key = torch.linspace(-1.0, 1.0, steps=24, dtype=torch.float32).reshape(
        1, 2, 3, 4
    )
    first_value = torch.linspace(-0.5, 0.75, steps=24, dtype=torch.float32).reshape(
        1, 2, 3, 4
    )
    second_key = torch.linspace(1.0, 2.0, steps=16, dtype=torch.float32).reshape(
        1, 2, 2, 4
    )
    second_value = torch.linspace(-1.25, -0.25, steps=16, dtype=torch.float32).reshape(
        1, 2, 2, 4
    )
    first_cache = KVCache(
        cache_dir=cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="quantized-cold-tier",
        cache_lifecycle="persistent",
    )
    first_cache.update(first_key, first_value, 0)

    second_cache = KVCache(
        cache_dir=cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="quantized-cold-tier",
        cache_lifecycle="persistent",
    )
    out = second_cache.update(second_key, second_value, 0)

    expected_key = torch.cat((first_key, second_key), dim=-2)
    expected_value = torch.cat((first_value, second_value), dim=-2)

    assert out[0].dtype == first_key.dtype
    assert out[1].dtype == first_value.dtype
    assert torch.allclose(out[0], expected_key, atol=0.02, rtol=0.05)
    assert torch.allclose(out[1], expected_value, atol=0.02, rtol=0.05)

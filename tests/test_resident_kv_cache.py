import torch

from ollm.kv_cache.resident import ResidentKVCache


def test_resident_kv_cache_reports_truthful_resident_state() -> None:
    cache = ResidentKVCache(device=torch.device("cpu"))
    key_states = torch.ones((1, 2, 4, 8), dtype=torch.float32)
    value_states = torch.ones((1, 2, 4, 8), dtype=torch.float32)

    updated = cache.update(key_states, value_states, 0)
    snapshot = cache.cache_state_snapshot()

    assert updated[0].shape[-2] == 4
    assert snapshot.strategy_id == "resident"
    assert snapshot.policy_id == "resident-baseline"
    assert snapshot.persistence_format == "resident-only"
    assert snapshot.residency_mode == "fully-resident"
    assert snapshot.persisted_tokens == 0
    assert snapshot.persisted_artifact_count == 0
    assert snapshot.resident_layer_count == 1
    assert snapshot.resident_tokens == 4
    assert snapshot.resident_bytes > 0
    assert snapshot.hot_tokens == 0
    assert snapshot.cold_store_format is None

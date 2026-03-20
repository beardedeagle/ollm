import json
from pathlib import Path

import pytest
import torch

from ollm.kv_cache_quantized_store import QuantizedJournaledKVStore


def _chunk_tensor(token_count: int, offset: float = 0.0) -> torch.Tensor:
    values = torch.linspace(
        -1.0 + offset,
        1.0 + offset,
        steps=token_count * 8,
        dtype=torch.float32,
    )
    return values.reshape(1, 2, token_count, 4)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_quantized_store_appends_and_replays_layer_round_trip(tmp_path: Path) -> None:
    store = QuantizedJournaledKVStore(
        tmp_path / "cache-root",
        compaction_entry_threshold=0,
    )
    store.initialize("test-policy")

    store.append_layer_chunk(0, (_chunk_tensor(3), _chunk_tensor(3, offset=0.25)))
    store.append_layer_chunk(
        0, (_chunk_tensor(2, offset=0.5), _chunk_tensor(2, offset=0.75))
    )

    loaded = store.load_layer(0, device=torch.device("cpu"))
    assert loaded is not None

    expected_key = torch.cat((_chunk_tensor(3), _chunk_tensor(2, offset=0.5)), dim=-2)
    expected_value = torch.cat(
        (_chunk_tensor(3, offset=0.25), _chunk_tensor(2, offset=0.75)),
        dim=-2,
    )
    layer_manifest = _read_json(store.cache_folder / "layers" / "0" / "manifest.json")
    entries = layer_manifest["entries"]
    assert isinstance(entries, list)

    assert loaded[0].dtype == expected_key.dtype
    assert loaded[1].dtype == expected_value.dtype
    assert torch.allclose(loaded[0], expected_key, atol=0.02, rtol=0.05)
    assert torch.allclose(loaded[1], expected_value, atol=0.02, rtol=0.05)
    assert layer_manifest["layout"] == "quantized-journal-append"
    assert layer_manifest["persisted_tokens"] == 5
    assert layer_manifest["compaction_count"] == 0
    assert len(entries) == 2
    assert store.persisted_artifact_count() == 2
    assert store.compaction_count() == 0
    assert store.cold_store_format_id() == "ollm-kv-journal-quantized"
    assert store.cold_tier_representation_id() == "int8-symmetric-per-tensor"


def test_quantized_store_root_manifest_records_representation(tmp_path: Path) -> None:
    store = QuantizedJournaledKVStore(
        tmp_path / "cache-root",
        compaction_entry_threshold=2,
    )
    store.initialize("test-policy")

    root_manifest = _read_json(store.cache_folder / "manifest.json")

    assert root_manifest["format"] == "ollm-kv-journal-quantized"
    assert root_manifest["quantized_dtype"] == "int8"
    assert root_manifest["cold_tier_representation"] == "int8-symmetric-per-tensor"
    assert root_manifest["compaction_entry_threshold"] == 2


def test_quantized_store_compacts_after_threshold(tmp_path: Path) -> None:
    store = QuantizedJournaledKVStore(
        tmp_path / "cache-root",
        compaction_entry_threshold=2,
    )
    store.initialize("test-policy")

    store.append_layer_chunk(0, (_chunk_tensor(2), _chunk_tensor(2, offset=0.25)))
    store.append_layer_chunk(
        0, (_chunk_tensor(2, offset=0.5), _chunk_tensor(2, offset=0.75))
    )

    layer_manifest = _read_json(store.cache_folder / "layers" / "0" / "manifest.json")
    entries = layer_manifest["entries"]
    assert isinstance(entries, list)

    assert layer_manifest["persisted_tokens"] == 4
    assert layer_manifest["compaction_count"] == 1
    assert len(entries) == 1
    assert store.persisted_artifact_count() == 1
    assert store.compaction_count() == 1
    assert store.consume_last_compaction_elapsed_seconds() is not None
    assert store.consume_last_compaction_elapsed_seconds() is None

    reopened = QuantizedJournaledKVStore(
        tmp_path / "cache-root",
        compaction_entry_threshold=2,
    )
    loaded = reopened.load_layer(0, device=torch.device("cpu"))
    assert loaded is not None
    assert loaded[0].shape[-2] == 4


def test_quantized_store_rejects_non_finite_tensors(tmp_path: Path) -> None:
    store = QuantizedJournaledKVStore(
        tmp_path / "cache-root",
        compaction_entry_threshold=0,
    )
    store.initialize("test-policy")
    key = _chunk_tensor(2)
    key[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        store.append_layer_chunk(0, (key, _chunk_tensor(2, offset=0.25)))

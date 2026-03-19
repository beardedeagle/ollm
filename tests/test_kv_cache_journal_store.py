import json
from pathlib import Path

import pytest
import torch

from ollm.kv_cache_journal_store import JournaledKVStore


def _chunk_tensor(token_count: int, offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset, offset + (token_count * 8), dtype=torch.float32)
    return values.reshape(1, 2, token_count, 4)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_journal_store_appends_and_replays_layer_round_trip(tmp_path: Path) -> None:
    store = JournaledKVStore(tmp_path / "cache-root")
    store.initialize("test-policy")

    store.append_layer_chunk(0, (_chunk_tensor(3), _chunk_tensor(3, offset=100)))
    store.append_layer_chunk(
        0, (_chunk_tensor(2, offset=1000), _chunk_tensor(2, offset=2000))
    )

    loaded = store.load_layer(0, device=torch.device("cpu"))
    assert loaded is not None

    expected_key = torch.cat((_chunk_tensor(3), _chunk_tensor(2, offset=1000)), dim=-2)
    expected_value = torch.cat(
        (_chunk_tensor(3, offset=100), _chunk_tensor(2, offset=2000)),
        dim=-2,
    )
    layer_manifest = _read_json(store.cache_folder / "layers" / "0" / "manifest.json")
    entries = layer_manifest["entries"]
    assert isinstance(entries, list)

    assert torch.equal(loaded[0], expected_key)
    assert torch.equal(loaded[1], expected_value)
    assert layer_manifest["layout"] == "journal-append"
    assert layer_manifest["persisted_tokens"] == 5
    assert len(entries) == 2
    assert store.persisted_artifact_count() == 2


def test_journal_store_rejects_paths_outside_cache_root(tmp_path: Path) -> None:
    store = JournaledKVStore(tmp_path / "cache-root")
    store.initialize("test-policy")
    store.append_layer_chunk(0, (_chunk_tensor(3), _chunk_tensor(3, offset=100)))

    layer_manifest_path = store.cache_folder / "layers" / "0" / "manifest.json"
    layer_manifest = _read_json(layer_manifest_path)
    layer_manifest["key_journal_path"] = "../escape.bin"
    layer_manifest_path.write_text(
        json.dumps(layer_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reloaded_store = JournaledKVStore(store.cache_folder)

    with pytest.raises(ValueError, match="must stay within the KV cache root"):
        reloaded_store.load_layer(0, device=torch.device("cpu"))

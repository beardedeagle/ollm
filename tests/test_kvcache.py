import json
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

from ollm.kvcache import KVCache
from ollm.qwen3_next import Qwen3NextDiskCache


def _chunk_tensor(token_count: int, offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset, offset + (token_count * 8), dtype=torch.float32)
    return values.reshape(1, 2, token_count, 4)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_chunks(manifest: dict[str, object]) -> list[dict[str, object]]:
    raw_chunks = manifest["chunks"]
    assert isinstance(raw_chunks, list)
    chunks: list[dict[str, object]] = []
    for chunk in raw_chunks:
        assert isinstance(chunk, dict)
        chunks.append(cast(dict[str, object], chunk))
    return chunks


def test_kvcache_creates_nested_cache_directory(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache-root"

    cache = KVCache(cache_dir=cache_root, device="cpu", stats=None)

    assert cache.cache_folder == cache_root / "kv_cache"
    assert cache.cache_folder.exists()
    assert (cache.cache_folder / "manifest.json").exists()


def test_kvcache_writes_manifest_backed_chunk_artifacts(tmp_path: Path) -> None:
    cache = KVCache(cache_dir=tmp_path / "cache-root", device="cpu", stats=None)
    key_states = _chunk_tensor(3)
    value_states = _chunk_tensor(3, offset=100)

    out = cache.update(key_states, value_states, 0)

    assert torch.equal(out[0], key_states)
    assert torch.equal(out[1], value_states)
    assert not any(cache.cache_folder.rglob("*.pt"))

    root_manifest = _read_json(cache.cache_folder / "manifest.json")
    layer_manifest = _read_json(cache.cache_folder / "layers" / "0" / "manifest.json")

    assert root_manifest["format"] == "ollm-kv-chunked"
    assert root_manifest["layers"] == [0]
    assert layer_manifest["persisted_tokens"] == 3
    assert layer_manifest["layout"] == "contiguous"

    chunks = _read_chunks(layer_manifest)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["start_token"] == 0
    assert chunk["end_token"] == 3
    assert chunk["key_dtype"] == "float32"
    assert chunk["value_dtype"] == "float32"
    assert chunk["key_shape"] == [1, 2, 3, 4]
    assert chunk["value_shape"] == [1, 2, 3, 4]
    assert (cache.cache_folder / chunk["key_path"]).exists()
    assert (cache.cache_folder / chunk["value_path"]).exists()


def test_kvcache_reload_and_append_chunks_round_trip(tmp_path: Path) -> None:
    cache = KVCache(cache_dir=tmp_path / "cache-root", device="cpu", stats=None)
    first_key = _chunk_tensor(3)
    first_value = _chunk_tensor(3, offset=100)
    second_key = _chunk_tensor(2, offset=1000)
    second_value = _chunk_tensor(2, offset=2000)

    cache.update(first_key, first_value, 0)
    out = cache.update(second_key, second_value, 0)
    persisted = cache.load_from_disk(0)
    assert persisted is not None

    expected_key = torch.cat((first_key, second_key), dim=-2)
    expected_value = torch.cat((first_value, second_value), dim=-2)

    assert torch.equal(out[0], expected_key)
    assert torch.equal(out[1], expected_value)
    assert torch.equal(persisted[0], expected_key)
    assert torch.equal(persisted[1], expected_value)

    layer_manifest = _read_json(cache.cache_folder / "layers" / "0" / "manifest.json")
    chunks = _read_chunks(layer_manifest)
    assert [chunk["start_token"] for chunk in chunks] == [0, 3]
    assert [chunk["end_token"] for chunk in chunks] == [3, 5]
    assert layer_manifest["persisted_tokens"] == 5


def test_kvcache_rejects_missing_chunk_files(tmp_path: Path) -> None:
    cache = KVCache(cache_dir=tmp_path / "cache-root", device="cpu", stats=None)
    cache.update(_chunk_tensor(3), _chunk_tensor(3, offset=100), 0)

    layer_manifest_path = cache.cache_folder / "layers" / "0" / "manifest.json"
    layer_manifest = _read_json(layer_manifest_path)
    chunks = _read_chunks(layer_manifest)
    missing_path = cache.cache_folder / chunks[0]["key_path"]
    missing_path.unlink()

    with pytest.raises(ValueError, match="chunk file is missing"):
        cache.load_from_disk(0)


def test_kvcache_rejects_non_contiguous_manifest_ranges(tmp_path: Path) -> None:
    cache = KVCache(cache_dir=tmp_path / "cache-root", device="cpu", stats=None)
    cache.update(_chunk_tensor(3), _chunk_tensor(3, offset=100), 0)
    cache.update(_chunk_tensor(2, offset=1000), _chunk_tensor(2, offset=2000), 0)

    layer_manifest_path = cache.cache_folder / "layers" / "0" / "manifest.json"
    layer_manifest = _read_json(layer_manifest_path)
    chunks = _read_chunks(layer_manifest)
    chunks[1]["start_token"] = 4
    _write_json(layer_manifest_path, layer_manifest)

    with pytest.raises(ValueError, match="non-contiguous chunk ranges"):
        cache.load_from_disk(0)


def test_kvcache_rejects_chunk_paths_outside_cache_root(tmp_path: Path) -> None:
    cache = KVCache(cache_dir=tmp_path / "cache-root", device="cpu", stats=None)
    cache.update(_chunk_tensor(3), _chunk_tensor(3, offset=100), 0)

    layer_manifest_path = cache.cache_folder / "layers" / "0" / "manifest.json"
    layer_manifest = _read_json(layer_manifest_path)
    chunks = _read_chunks(layer_manifest)
    chunks[0]["key_path"] = "../escape.bin"
    _write_json(layer_manifest_path, layer_manifest)

    with pytest.raises(ValueError, match="must stay within the KV cache root"):
        cache.load_from_disk(0)


def test_qwen3_next_disk_cache_persists_chunked_sequence_growth(
    tmp_path: Path,
) -> None:
    cache = Qwen3NextDiskCache(
        Qwen3NextConfig(num_hidden_layers=2),
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
    )
    first_key = _chunk_tensor(3)
    first_value = _chunk_tensor(3, offset=100)
    second_key = _chunk_tensor(2, offset=1000)
    second_value = _chunk_tensor(2, offset=2000)

    cache.update(first_key, first_value, 0)
    out = cache.update(second_key, second_value, 0)
    persisted = cache.load_from_disk(0)
    assert persisted is not None

    expected = torch.cat((first_key, second_key), dim=-2)
    assert torch.equal(out[0], expected)
    assert torch.equal(persisted[0], expected)
    assert cache.get_seq_length(0) == 5
    assert not any(cache.cache_folder.rglob("*.pt"))

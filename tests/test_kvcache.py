import json
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig

from ollm.kv_cache_policy import KVCachePolicy
from ollm.kv_cache_store import ChunkedKVStore
from ollm.kv_cache_strategy import kv_cache_root
from ollm.kv_cache_streamed_store import StreamedSegmentedKVStore
from ollm.kvcache import KVCache
from ollm.qwen3_next import Qwen3NextDiskCache
from ollm.utils import Stats


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


def _immediate_flush_policy() -> KVCachePolicy:
    return KVCachePolicy(
        policy_id="test-immediate-flush",
        flush_token_threshold=1,
        flush_byte_threshold=1,
    )


def _buffered_policy() -> KVCachePolicy:
    return KVCachePolicy(
        policy_id="test-buffered",
        flush_token_threshold=8,
        flush_byte_threshold=1_000_000,
    )


def test_kvcache_creates_nested_cache_directory(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache-root"

    cache = KVCache(
        cache_dir=cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
    )

    assert cache.cache_folder == kv_cache_root(cache_root, "chunked")
    assert cache.cache_folder.exists()
    assert (cache.cache_folder / "manifest.json").exists()


def test_kvcache_persistent_lifecycle_reuses_existing_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache-root"
    first_cache = KVCache(
        cache_dir=cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_lifecycle="persistent",
    )
    key_states = _chunk_tensor(3)
    value_states = _chunk_tensor(3, offset=100)
    first_cache.update(key_states, value_states, 0)

    second_cache = KVCache(
        cache_dir=cache_root,
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_lifecycle="persistent",
    )
    persisted = second_cache.load_from_disk(0)

    assert persisted is not None
    assert torch.equal(persisted[0], key_states)
    assert torch.equal(persisted[1], value_states)


def test_kvcache_writes_manifest_backed_chunk_artifacts(tmp_path: Path) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
    )
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
    assert root_manifest["policy_id"] == "test-immediate-flush"
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
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
    )
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
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
    )
    cache.update(_chunk_tensor(3), _chunk_tensor(3, offset=100), 0)

    layer_manifest_path = cache.cache_folder / "layers" / "0" / "manifest.json"
    layer_manifest = _read_json(layer_manifest_path)
    chunks = _read_chunks(layer_manifest)
    missing_path = cache.cache_folder / chunks[0]["key_path"]
    missing_path.unlink()
    reloaded_store = ChunkedKVStore(cache.cache_folder)

    with pytest.raises(ValueError, match="chunk file is missing"):
        reloaded_store.load_layer(0, device=torch.device("cpu"))


def test_kvcache_rejects_non_contiguous_manifest_ranges(tmp_path: Path) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
    )
    cache.update(_chunk_tensor(3), _chunk_tensor(3, offset=100), 0)
    cache.update(_chunk_tensor(2, offset=1000), _chunk_tensor(2, offset=2000), 0)

    layer_manifest_path = cache.cache_folder / "layers" / "0" / "manifest.json"
    layer_manifest = _read_json(layer_manifest_path)
    chunks = _read_chunks(layer_manifest)
    chunks[1]["start_token"] = 4
    _write_json(layer_manifest_path, layer_manifest)
    reloaded_store = ChunkedKVStore(cache.cache_folder)

    with pytest.raises(ValueError, match="non-contiguous chunk ranges"):
        reloaded_store.load_layer(0, device=torch.device("cpu"))


def test_kvcache_rejects_chunk_paths_outside_cache_root(tmp_path: Path) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
    )
    cache.update(_chunk_tensor(3), _chunk_tensor(3, offset=100), 0)

    layer_manifest_path = cache.cache_folder / "layers" / "0" / "manifest.json"
    layer_manifest = _read_json(layer_manifest_path)
    chunks = _read_chunks(layer_manifest)
    chunks[0]["key_path"] = "../escape.bin"
    _write_json(layer_manifest_path, layer_manifest)
    reloaded_store = ChunkedKVStore(cache.cache_folder)

    with pytest.raises(ValueError, match="must stay within the KV cache root"):
        reloaded_store.load_layer(0, device=torch.device("cpu"))


@pytest.mark.parametrize(
    "cache_strategy",
    ["chunked", "streamed-segmented", "sliding-window-ring-buffer"],
)
def test_qwen3_next_disk_cache_persists_sequence_growth(
    tmp_path: Path, cache_strategy: str
) -> None:
    cache = Qwen3NextDiskCache(
        Qwen3NextConfig(num_hidden_layers=2),
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy=cache_strategy,
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
    if cache_strategy == "sliding-window-ring-buffer":
        expected = expected[..., -256:, :]
    assert torch.equal(out[0], expected)
    assert torch.equal(persisted[0], expected)
    assert cache.get_seq_length(0) == 5
    assert not any(cache.cache_folder.rglob("*.pt"))


def test_streamed_kvcache_writes_manifest_backed_segment_artifacts(
    tmp_path: Path,
) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="streamed-segmented",
    )

    cache.update(_chunk_tensor(3), _chunk_tensor(3, offset=100), 0)

    root_manifest = _read_json(cache.cache_folder / "manifest.json")
    layer_manifest = _read_json(cache.cache_folder / "layers" / "0" / "manifest.json")
    extents = cast(list[dict[str, object]], layer_manifest["extents"])
    extent = extents[0]

    assert cache.cache_folder == kv_cache_root(
        tmp_path / "cache-root", "streamed-segmented"
    )
    assert root_manifest["format"] == "ollm-kv-streamed-segmented"
    assert layer_manifest["layout"] == "streamed-segmented"
    assert layer_manifest["persisted_tokens"] == 3
    assert extent["key_path"] == "layers/0/key/segment-000000.bin"
    assert extent["value_path"] == "layers/0/value/segment-000000.bin"
    assert extent["key_offset"] == 0
    assert extent["value_offset"] == 0
    assert (cache.cache_folder / cast(str, extent["key_path"])).exists()
    assert (cache.cache_folder / cast(str, extent["value_path"])).exists()


def test_streamed_kvcache_reload_and_append_extents_round_trip(tmp_path: Path) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="streamed-segmented",
    )
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
    extents = cast(list[dict[str, object]], layer_manifest["extents"])
    assert [extent["start_token"] for extent in extents] == [0, 3]
    assert [extent["end_token"] for extent in extents] == [3, 5]


def test_streamed_kvcache_rejects_paths_outside_cache_root(tmp_path: Path) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_immediate_flush_policy(),
        cache_strategy="streamed-segmented",
    )
    cache.update(_chunk_tensor(3), _chunk_tensor(3, offset=100), 0)

    layer_manifest_path = cache.cache_folder / "layers" / "0" / "manifest.json"
    layer_manifest = _read_json(layer_manifest_path)
    extents = cast(list[dict[str, object]], layer_manifest["extents"])
    extents[0]["key_path"] = "../escape.bin"
    _write_json(layer_manifest_path, layer_manifest)
    reloaded_store = StreamedSegmentedKVStore(cache.cache_folder)

    with pytest.raises(ValueError, match="must stay within the KV cache root"):
        reloaded_store.load_layer(0, device=torch.device("cpu"))


def test_kvcache_buffers_tail_until_policy_threshold_then_flushes(
    tmp_path: Path,
) -> None:
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=None,
        policy=_buffered_policy(),
    )

    first_key = _chunk_tensor(3)
    first_value = _chunk_tensor(3, offset=100)
    second_key = _chunk_tensor(2, offset=1000)
    second_value = _chunk_tensor(2, offset=2000)

    cache.update(first_key, first_value, 0)
    cache.update(second_key, second_value, 0)

    persisted_manifest_path = cache.cache_folder / "layers" / "0" / "manifest.json"
    assert not persisted_manifest_path.exists()

    persisted = cache.load_from_disk(0)
    assert persisted is not None
    assert torch.equal(
        persisted[0],
        torch.cat((first_key, second_key), dim=-2),
    )

    third_key = _chunk_tensor(4, offset=3000)
    third_value = _chunk_tensor(4, offset=4000)
    cache.update(third_key, third_value, 0)

    layer_manifest = _read_json(persisted_manifest_path)
    assert layer_manifest["persisted_tokens"] == 9
    chunks = _read_chunks(layer_manifest)
    assert len(chunks) == 1
    assert chunks[0]["start_token"] == 0
    assert chunks[0]["end_token"] == 9


@pytest.mark.parametrize(
    "cache_strategy",
    [
        "chunked",
        "streamed-segmented",
        "log-structured-journal",
        "quantized-cold-tier",
    ],
)
def test_kvcache_reuses_resident_layer_after_update(
    tmp_path: Path, cache_strategy: str
) -> None:
    stats = Stats()
    cache = KVCache(
        cache_dir=tmp_path / "cache-root",
        device="cpu",
        stats=stats,
        policy=_immediate_flush_policy(),
        cache_strategy=cache_strategy,
    )

    cache.update(_chunk_tensor(3), _chunk_tensor(3, offset=100), 0)
    cache.update(_chunk_tensor(2, offset=1000), _chunk_tensor(2, offset=2000), 0)

    stats.clear()
    first = cache.load_from_disk(0)
    second = cache.load_from_disk(0)
    summary = stats.collect_and_clear_ms()

    assert first is not None
    assert second is not None
    assert torch.equal(first[0], second[0])
    assert "kvload" not in summary


def test_streamed_store_reads_shared_segment_once_per_tensor_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StreamedSegmentedKVStore(tmp_path / "cache-root")
    store.initialize("test-policy")
    store.append_layer_chunk(0, (_chunk_tensor(3), _chunk_tensor(3, offset=100)))
    store.append_layer_chunk(
        0,
        (
            _chunk_tensor(2, offset=1000),
            _chunk_tensor(2, offset=2000),
        ),
    )

    import ollm.kv_cache_streamed_store as streamed_module

    calls: list[tuple[Path, int, int]] = []
    original = streamed_module.path_read_bytes_range

    def _recording_read(path: Path, *, offset: int, length: int) -> bytes:
        calls.append((path, offset, length))
        return original(path, offset=offset, length=length)

    monkeypatch.setattr(streamed_module, "path_read_bytes_range", _recording_read)

    loaded = store.load_layer(0, device=torch.device("cpu"))

    assert loaded is not None
    assert len(calls) == 2
    assert calls[0][0].name == "segment-000000.bin"
    assert calls[1][0].name == "segment-000000.bin"

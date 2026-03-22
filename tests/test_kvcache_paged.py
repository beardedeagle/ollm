import json
from pathlib import Path
from typing import cast

import pytest
import torch

from ollm.kv_cache.paged_store import PagedKVStore


def _chunk_tensor(token_count: int, offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset, offset + (token_count * 8), dtype=torch.float32)
    return values.reshape(1, 2, token_count, 4)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_pages(manifest: dict[str, object]) -> list[dict[str, object]]:
    raw_pages = manifest["pages"]
    assert isinstance(raw_pages, list)
    pages: list[dict[str, object]] = []
    for page in raw_pages:
        assert isinstance(page, dict)
        pages.append(cast(dict[str, object], page))
    return pages


def test_paged_store_writes_root_and_layer_manifests(tmp_path: Path) -> None:
    store = PagedKVStore(tmp_path / "cache-root", page_token_capacity=4)
    store.initialize("test-paged")
    store.append_layer_chunk(0, (_chunk_tensor(3), _chunk_tensor(3, offset=100)))

    root_manifest = _read_json(store.cache_folder / "manifest.json")
    layer_manifest = _read_json(store.cache_folder / "layers" / "0" / "manifest.json")
    pages = _read_pages(layer_manifest)

    assert root_manifest["format"] == "ollm-kv-paged"
    assert root_manifest["layers"] == [0]
    assert root_manifest["page_token_capacity"] == 4
    assert layer_manifest["persisted_tokens"] == 3
    assert layer_manifest["page_token_capacity"] == 4
    assert layer_manifest["key_pages_path"] == "layers/0/key/pages.bin"
    assert layer_manifest["value_pages_path"] == "layers/0/value/pages.bin"
    assert len(pages) == 1
    assert pages[0]["page_index"] == 0
    assert pages[0]["start_token"] == 0
    assert pages[0]["end_token"] == 3


def test_paged_store_rewrites_partial_tail_and_emits_new_page(tmp_path: Path) -> None:
    store = PagedKVStore(tmp_path / "cache-root", page_token_capacity=4)
    store.initialize("test-paged")
    first_key = _chunk_tensor(3)
    first_value = _chunk_tensor(3, offset=100)
    second_key = _chunk_tensor(3, offset=1000)
    second_value = _chunk_tensor(3, offset=2000)

    store.append_layer_chunk(0, (first_key, first_value))
    store.append_layer_chunk(0, (second_key, second_value))
    persisted = store.load_layer(0, device=torch.device("cpu"))

    layer_manifest = _read_json(store.cache_folder / "layers" / "0" / "manifest.json")
    pages = _read_pages(layer_manifest)
    assert persisted is not None

    expected_key = torch.cat((first_key, second_key), dim=-2)
    expected_value = torch.cat((first_value, second_value), dim=-2)

    assert torch.equal(persisted[0], expected_key)
    assert torch.equal(persisted[1], expected_value)
    assert [page["page_index"] for page in pages] == [0, 1]
    assert [page["start_token"] for page in pages] == [0, 4]
    assert [page["end_token"] for page in pages] == [4, 6]
    assert store.persisted_artifact_count() == 2


def test_paged_store_persistent_reopen_round_trips(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache-root"
    first_store = PagedKVStore(cache_root, page_token_capacity=4)
    first_store.initialize("test-paged")
    key_states = _chunk_tensor(5)
    value_states = _chunk_tensor(5, offset=100)
    first_store.append_layer_chunk(0, (key_states, value_states))

    reopened_store = PagedKVStore(cache_root, page_token_capacity=4)
    persisted = reopened_store.load_layer(0, device=torch.device("cpu"))

    assert persisted is not None
    assert torch.equal(persisted[0], key_states)
    assert torch.equal(persisted[1], value_states)


def test_paged_store_rejects_blob_paths_outside_cache_root(tmp_path: Path) -> None:
    store = PagedKVStore(tmp_path / "cache-root", page_token_capacity=4)
    store.initialize("test-paged")
    store.append_layer_chunk(0, (_chunk_tensor(3), _chunk_tensor(3, offset=100)))

    manifest_path = store.cache_folder / "layers" / "0" / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["key_pages_path"] = "../escape.bin"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="must stay within the KV cache root"):
        PagedKVStore(store.cache_folder, page_token_capacity=4).load_layer(
            0, device=torch.device("cpu")
        )


def test_paged_store_rejects_out_of_bounds_offsets(tmp_path: Path) -> None:
    store = PagedKVStore(tmp_path / "cache-root", page_token_capacity=4)
    store.initialize("test-paged")
    store.append_layer_chunk(0, (_chunk_tensor(3), _chunk_tensor(3, offset=100)))

    manifest_path = store.cache_folder / "layers" / "0" / "manifest.json"
    manifest = _read_json(manifest_path)
    pages = _read_pages(manifest)
    pages[0]["key_offset"] = 999_999
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="exceeds blob bounds"):
        PagedKVStore(store.cache_folder, page_token_capacity=4).load_layer(
            0, device=torch.device("cpu")
        )


def test_paged_store_rejects_non_contiguous_page_ranges(tmp_path: Path) -> None:
    store = PagedKVStore(tmp_path / "cache-root", page_token_capacity=4)
    store.initialize("test-paged")
    store.append_layer_chunk(0, (_chunk_tensor(5), _chunk_tensor(5, offset=100)))

    manifest_path = store.cache_folder / "layers" / "0" / "manifest.json"
    manifest = _read_json(manifest_path)
    pages = _read_pages(manifest)
    pages[1]["start_token"] = 3
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="unexpected start token"):
        PagedKVStore(store.cache_folder, page_token_capacity=4).load_layer(
            0, device=torch.device("cpu")
        )


def test_paged_store_rewrites_tail_without_leaking_dead_blob_bytes(
    tmp_path: Path,
) -> None:
    store = PagedKVStore(tmp_path / "cache-root", page_token_capacity=4)
    store.initialize("test-paged")

    for offset in (0, 1000, 2000, 3000):
        store.append_layer_chunk(
            0, (_chunk_tensor(1, offset=offset), _chunk_tensor(1, offset=offset + 100))
        )

    manifest = _read_json(store.cache_folder / "layers" / "0" / "manifest.json")
    pages = _read_pages(manifest)
    key_blob = store.cache_folder / "layers" / "0" / "key" / "pages.bin"
    value_blob = store.cache_folder / "layers" / "0" / "value" / "pages.bin"

    assert key_blob.stat().st_size == sum(
        cast(int, page["key_nbytes"]) for page in pages
    )
    assert value_blob.stat().st_size == sum(
        cast(int, page["value_nbytes"]) for page in pages
    )

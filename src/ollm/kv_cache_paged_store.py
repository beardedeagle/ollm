"""Fixed-page disk KV store with explicit page-table metadata."""

import json
from pathlib import Path

import torch

from ollm.async_io import (
    path_append_bytes,
    path_exists,
    path_file_size,
    path_mkdir,
    path_read_bytes_range,
    path_write_bytes_at_offset,
)
from ollm.kv_cache_paged_manifest import (
    PAGED_BLOB_FILE_NAME,
    PAGED_CACHE_LAYOUT,
    KVPagedLayerManifest,
    KVPageMetadata,
    validate_paged_layer_manifest,
)
from ollm.kv_cache_store_common import (
    CACHE_SCHEMA_VERSION,
    PERSISTED_DEVICE,
    SEQUENCE_AXIS,
    atomic_write_text,
    decode_tensor_bytes,
    dtype_from_name,
    dtype_name,
    encode_tensor_bytes,
    read_json_object,
    require_int,
    require_int_value,
    require_str,
    sequence_length,
    shape_prefix,
)

DEFAULT_PAGE_TOKEN_CAPACITY = 128
_CACHE_FORMAT = "ollm-kv-paged"


class PagedKVStore:
    def __init__(
        self,
        cache_folder: Path,
        *,
        page_token_capacity: int = DEFAULT_PAGE_TOKEN_CAPACITY,
    ) -> None:
        if page_token_capacity <= 0:
            raise ValueError("page_token_capacity must be greater than zero")
        self.cache_folder = cache_folder
        self.layers_folder = cache_folder / "layers"
        self.root_manifest_path = cache_folder / "manifest.json"
        self.page_token_capacity = page_token_capacity
        self._root_manifest_cache: tuple[tuple[int, ...], str] | None = None
        self._layer_manifest_cache: dict[int, KVPagedLayerManifest | None] = {}

    def initialize(self, policy_id: str) -> None:
        path_mkdir(self.layers_folder, parents=True, exist_ok=True)
        self._layer_manifest_cache.clear()
        self._write_root_manifest((), policy_id)

    def load_layer(
        self, layer_idx: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        layer_manifest = self._read_layer_manifest(layer_idx)
        if layer_manifest is None:
            return None
        _ = self._read_root_manifest()
        if not layer_manifest.pages:
            return None
        key_blob_path = self._blob_path(layer_manifest.key_pages_path)
        value_blob_path = self._blob_path(layer_manifest.value_pages_path)
        key_pages = [
            self._read_page_tensor(
                key_blob_path,
                offset=page.key_offset,
                nbytes=page.key_nbytes,
                dtype_name_value=page.key_dtype,
                shape=page.key_shape,
            )
            for page in layer_manifest.pages
        ]
        value_pages = [
            self._read_page_tensor(
                value_blob_path,
                offset=page.value_offset,
                nbytes=page.value_nbytes,
                dtype_name_value=page.value_dtype,
                shape=page.value_shape,
            )
            for page in layer_manifest.pages
        ]
        return (
            torch.cat(key_pages, dim=SEQUENCE_AXIS).to(device),
            torch.cat(value_pages, dim=SEQUENCE_AXIS).to(device),
        )

    def append_layer_chunk(
        self, layer_idx: int, tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        key_tensor = tensors[0].detach().cpu().contiguous()
        value_tensor = tensors[1].detach().cpu().contiguous()
        self._validate_chunk_pair(layer_idx, key_tensor, value_tensor)
        token_count = sequence_length(tuple(key_tensor.shape))
        if token_count == 0:
            return

        layer_manifest = self._read_layer_manifest(layer_idx)
        if layer_manifest is None:
            key_pages_path, value_pages_path = self._relative_blob_paths(layer_idx)
            layer_manifest = KVPagedLayerManifest.new(
                layer_idx,
                page_token_capacity=self.page_token_capacity,
                key_pages_path=key_pages_path,
                value_pages_path=value_pages_path,
            )

        pages = list(layer_manifest.pages)
        write_key = key_tensor
        write_value = value_tensor
        next_page_index = len(pages)
        next_start_token = layer_manifest.persisted_tokens
        partial_page: KVPageMetadata | None = None

        if pages and pages[-1].token_count < self.page_token_capacity:
            partial_page = pages.pop()
            key_blob_path = self._blob_path(layer_manifest.key_pages_path)
            value_blob_path = self._blob_path(layer_manifest.value_pages_path)
            partial_key = self._read_page_tensor(
                key_blob_path,
                offset=partial_page.key_offset,
                nbytes=partial_page.key_nbytes,
                dtype_name_value=partial_page.key_dtype,
                shape=partial_page.key_shape,
            )
            partial_value = self._read_page_tensor(
                value_blob_path,
                offset=partial_page.value_offset,
                nbytes=partial_page.value_nbytes,
                dtype_name_value=partial_page.value_dtype,
                shape=partial_page.value_shape,
            )
            write_key = torch.cat((partial_key, write_key), dim=SEQUENCE_AXIS)
            write_value = torch.cat((partial_value, write_value), dim=SEQUENCE_AXIS)
            next_page_index = partial_page.page_index
            next_start_token = partial_page.start_token

        key_blob_path = self._blob_path(layer_manifest.key_pages_path)
        value_blob_path = self._blob_path(layer_manifest.value_pages_path)
        path_mkdir(key_blob_path.parent, parents=True, exist_ok=True)
        path_mkdir(value_blob_path.parent, parents=True, exist_ok=True)
        if partial_page is not None:
            self._validate_rewritable_tail_page(
                partial_page,
                key_blob_path=key_blob_path,
                value_blob_path=value_blob_path,
            )

        for key_page, value_page in _split_tensor_pair_into_pages(
            (write_key, write_value),
            page_token_capacity=self.page_token_capacity,
        ):
            key_bytes = encode_tensor_bytes(key_page)
            value_bytes = encode_tensor_bytes(value_page)
            page_token_count = sequence_length(tuple(key_page.shape))
            end_token = next_start_token + page_token_count
            if partial_page is not None and next_page_index == partial_page.page_index:
                key_offset = partial_page.key_offset
                value_offset = partial_page.value_offset
                path_write_bytes_at_offset(
                    key_blob_path, offset=key_offset, content=key_bytes
                )
                path_write_bytes_at_offset(
                    value_blob_path, offset=value_offset, content=value_bytes
                )
            else:
                key_offset = path_append_bytes(key_blob_path, key_bytes)
                value_offset = path_append_bytes(value_blob_path, value_bytes)
            pages.append(
                KVPageMetadata(
                    page_index=next_page_index,
                    start_token=next_start_token,
                    end_token=end_token,
                    key_dtype=dtype_name(key_page.dtype),
                    value_dtype=dtype_name(value_page.dtype),
                    key_shape=tuple(key_page.shape),
                    value_shape=tuple(value_page.shape),
                    key_offset=key_offset,
                    value_offset=value_offset,
                    key_nbytes=len(key_bytes),
                    value_nbytes=len(value_bytes),
                )
            )
            next_page_index += 1
            next_start_token = end_token

        updated_manifest = KVPagedLayerManifest(
            layer_idx=layer_idx,
            layout=PAGED_CACHE_LAYOUT,
            sequence_axis=SEQUENCE_AXIS,
            persisted_tokens=next_start_token,
            page_token_capacity=self.page_token_capacity,
            key_pages_path=layer_manifest.key_pages_path,
            value_pages_path=layer_manifest.value_pages_path,
            pages=tuple(pages),
        )
        validate_paged_layer_manifest(updated_manifest)
        self._write_layer_manifest(updated_manifest)

        root_layers, policy_id = self._read_root_manifest()
        if layer_idx not in root_layers:
            self._write_root_manifest(
                tuple(sorted(root_layers + (layer_idx,))), policy_id
            )

    def persisted_layer_ids(self) -> tuple[int, ...]:
        if not path_exists(self.root_manifest_path):
            return ()
        return self._read_root_manifest()[0]

    def persisted_token_count(self) -> int:
        total_tokens = 0
        for layer_idx in self.persisted_layer_ids():
            manifest = self._read_layer_manifest(layer_idx)
            if manifest is None:
                continue
            total_tokens += manifest.persisted_tokens
        return total_tokens

    def persisted_artifact_count(self) -> int:
        total_pages = 0
        for layer_idx in self.persisted_layer_ids():
            manifest = self._read_layer_manifest(layer_idx)
            if manifest is None:
                continue
            total_pages += len(manifest.pages)
        return total_pages

    def cold_store_format_id(self) -> str | None:
        return _CACHE_FORMAT

    def cold_tier_representation_id(self) -> str | None:
        return None

    def compaction_count(self) -> int:
        return 0

    def eviction_count(self) -> int:
        return 0

    def evicted_token_count(self) -> int:
        return 0

    def consume_last_compaction_elapsed_seconds(self) -> float | None:
        return None

    def _validate_chunk_pair(
        self, layer_idx: int, key_tensor: torch.Tensor, value_tensor: torch.Tensor
    ) -> None:
        key_shape = tuple(key_tensor.shape)
        value_shape = tuple(value_tensor.shape)
        if len(key_shape) < 2 or len(value_shape) < 2:
            raise ValueError(
                f"KV cache chunk tensors must have rank >= 2 for layer {layer_idx}"
            )
        if sequence_length(key_shape) != sequence_length(value_shape):
            raise ValueError(
                f"KV cache key/value token counts differ for layer {layer_idx}"
            )
        if shape_prefix(key_shape) != shape_prefix(value_shape):
            raise ValueError(
                f"KV cache key/value non-sequence dimensions differ for layer {layer_idx}"
            )
        if sequence_length(key_shape) <= 0:
            raise ValueError(
                f"KV cache chunk token count must be positive for layer {layer_idx}"
            )

    def _read_root_manifest(self) -> tuple[tuple[int, ...], str]:
        if self._root_manifest_cache is not None:
            return self._root_manifest_cache
        if not path_exists(self.root_manifest_path):
            raise ValueError(
                f"KV cache root manifest is missing: {self.root_manifest_path}"
            )
        payload = read_json_object(self.root_manifest_path)
        if require_int(payload, "schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported KV cache schema version: {payload['schema_version']}"
            )
        if require_str(payload, "format") != _CACHE_FORMAT:
            raise ValueError(f"Unsupported KV cache format: {payload['format']!r}")
        if require_int(payload, "chunk_axis") != SEQUENCE_AXIS:
            raise ValueError(
                f"Unsupported KV cache chunk axis: {payload['chunk_axis']}"
            )
        if require_str(payload, "persisted_device") != PERSISTED_DEVICE:
            raise ValueError(
                f"Unsupported KV cache persisted device: {payload['persisted_device']!r}"
            )
        policy_id = require_str(payload, "policy_id")
        page_token_capacity = require_int(payload, "page_token_capacity")
        if page_token_capacity != self.page_token_capacity:
            raise ValueError(
                f"Unsupported KV paged page_token_capacity: {page_token_capacity}"
            )
        layers_payload = payload.get("layers")
        if not isinstance(layers_payload, list):
            raise ValueError("KV root manifest layers must be a JSON list")
        manifest = (
            tuple(require_int_value(value, "layers[]") for value in layers_payload),
            policy_id,
        )
        self._root_manifest_cache = manifest
        return manifest

    def _write_root_manifest(self, layers: tuple[int, ...], policy_id: str) -> None:
        atomic_write_text(
            self.root_manifest_path,
            json.dumps(
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "format": _CACHE_FORMAT,
                    "chunk_axis": SEQUENCE_AXIS,
                    "persisted_device": PERSISTED_DEVICE,
                    "policy_id": policy_id,
                    "page_token_capacity": self.page_token_capacity,
                    "layers": list(layers),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        self._root_manifest_cache = (layers, policy_id)

    def _relative_blob_paths(self, layer_idx: int) -> tuple[str, str]:
        layer_folder = self.layers_folder / str(layer_idx)
        key_blob = layer_folder / "key" / PAGED_BLOB_FILE_NAME
        value_blob = layer_folder / "value" / PAGED_BLOB_FILE_NAME
        return (
            str(key_blob.relative_to(self.cache_folder)),
            str(value_blob.relative_to(self.cache_folder)),
        )

    def _blob_path(self, relative_path: str) -> Path:
        resolved_path = (self.cache_folder / relative_path).resolve()
        cache_root = self.cache_folder.resolve()
        if not resolved_path.is_relative_to(cache_root):
            raise ValueError(f"KV paged blob path escapes cache root: {relative_path}")
        return resolved_path

    def _read_layer_manifest(self, layer_idx: int) -> KVPagedLayerManifest | None:
        if layer_idx in self._layer_manifest_cache:
            return self._layer_manifest_cache[layer_idx]
        layer_folder = self.layers_folder / str(layer_idx)
        if not path_exists(layer_folder):
            self._layer_manifest_cache[layer_idx] = None
            return None
        manifest_path = layer_folder / "manifest.json"
        if not path_exists(manifest_path):
            raise ValueError(f"KV paged layer manifest is missing: {manifest_path}")
        manifest = KVPagedLayerManifest.from_dict(read_json_object(manifest_path))
        key_blob_path = self._blob_path(manifest.key_pages_path)
        value_blob_path = self._blob_path(manifest.value_pages_path)
        self._validate_page_offsets(manifest, key_blob_path, value_blob_path)
        self._layer_manifest_cache[layer_idx] = manifest
        return manifest

    def _validate_page_offsets(
        self,
        manifest: KVPagedLayerManifest,
        key_blob_path: Path,
        value_blob_path: Path,
    ) -> None:
        if not path_exists(key_blob_path):
            raise ValueError(f"KV paged key blob is missing: {key_blob_path}")
        if not path_exists(value_blob_path):
            raise ValueError(f"KV paged value blob is missing: {value_blob_path}")
        key_blob_size = path_file_size(key_blob_path)
        value_blob_size = path_file_size(value_blob_path)
        for page in manifest.pages:
            if page.key_offset + page.key_nbytes > key_blob_size:
                raise ValueError(
                    f"KV paged key page exceeds blob bounds for layer {manifest.layer_idx}"
                )
            if page.value_offset + page.value_nbytes > value_blob_size:
                raise ValueError(
                    f"KV paged value page exceeds blob bounds for layer {manifest.layer_idx}"
                )

    def _validate_rewritable_tail_page(
        self,
        page: KVPageMetadata,
        *,
        key_blob_path: Path,
        value_blob_path: Path,
    ) -> None:
        if page.key_offset + page.key_nbytes != path_file_size(key_blob_path):
            raise ValueError("KV paged key tail page is not rewritable at blob EOF")
        if page.value_offset + page.value_nbytes != path_file_size(value_blob_path):
            raise ValueError("KV paged value tail page is not rewritable at blob EOF")

    def _write_layer_manifest(self, manifest: KVPagedLayerManifest) -> None:
        layer_folder = self.layers_folder / str(manifest.layer_idx)
        path_mkdir(layer_folder, parents=True, exist_ok=True)
        manifest_path = layer_folder / "manifest.json"
        atomic_write_text(
            manifest_path,
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        self._layer_manifest_cache[manifest.layer_idx] = manifest

    def _read_page_tensor(
        self,
        path: Path,
        *,
        offset: int,
        nbytes: int,
        dtype_name_value: str,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        resolved_path = path.resolve()
        cache_root = self.cache_folder.resolve()
        if not resolved_path.is_relative_to(cache_root):
            raise ValueError(f"KV paged blob path escapes cache root: {path}")
        if not path_exists(path):
            raise ValueError(f"KV paged blob is missing: {path}")
        raw_bytes = path_read_bytes_range(path, offset=offset, length=nbytes)
        return decode_tensor_bytes(
            raw_bytes,
            dtype=dtype_from_name(dtype_name_value),
            shape=shape,
        )


def _split_tensor_pair_into_pages(
    tensors: tuple[torch.Tensor, torch.Tensor],
    *,
    page_token_capacity: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    total_tokens = sequence_length(tuple(tensors[0].shape))
    pages: list[tuple[torch.Tensor, torch.Tensor]] = []
    start_token = 0
    while start_token < total_tokens:
        end_token = min(total_tokens, start_token + page_token_capacity)
        pages.append(
            (
                tensors[0][..., start_token:end_token, :].contiguous(),
                tensors[1][..., start_token:end_token, :].contiguous(),
            )
        )
        start_token = end_token
    return tuple(pages)

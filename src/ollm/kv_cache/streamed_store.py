"""Streamed append/read disk KV store with explicit segment metadata."""

import json
from pathlib import Path

import torch

from ollm.async_io import (
    path_append_bytes,
    path_exists,
    path_file_size,
    path_mkdir,
    path_read_bytes_range,
)
from ollm.kv_cache.store_common import (
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
from ollm.kv_cache.streamed_manifest import (
    DEFAULT_SEGMENT_BYTES_TARGET,
    STREAMED_CACHE_LAYOUT,
    KVStreamExtentMetadata,
    KVStreamLayerManifest,
    validate_stream_layer_manifest,
)

_CACHE_FORMAT = "ollm-kv-streamed-segmented"


class StreamedSegmentedKVStore:
    def __init__(
        self,
        cache_folder: Path,
        *,
        segment_bytes_target: int = DEFAULT_SEGMENT_BYTES_TARGET,
    ) -> None:
        self.cache_folder = cache_folder
        self.layers_folder = cache_folder / "layers"
        self.root_manifest_path = cache_folder / "manifest.json"
        self.segment_bytes_target = segment_bytes_target
        self._root_manifest_cache: tuple[tuple[int, ...], str] | None = None
        self._layer_manifest_cache: dict[int, KVStreamLayerManifest | None] = {}

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
        key_chunks = self._read_extent_tensors(layer_manifest.extents, kind="key")
        value_chunks = self._read_extent_tensors(layer_manifest.extents, kind="value")
        return (
            torch.cat(key_chunks, dim=SEQUENCE_AXIS).to(device),
            torch.cat(value_chunks, dim=SEQUENCE_AXIS).to(device),
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
        key_bytes = encode_tensor_bytes(key_tensor)
        value_bytes = encode_tensor_bytes(value_tensor)
        layer_manifest = self._read_layer_manifest(
            layer_idx
        ) or KVStreamLayerManifest.new(
            layer_idx,
            segment_bytes_target=self.segment_bytes_target,
        )
        start_token = layer_manifest.persisted_tokens
        end_token = start_token + token_count
        key_path, value_path = self._target_segment_paths(
            layer_idx=layer_idx,
            key_bytes=len(key_bytes),
            value_bytes=len(value_bytes),
            layer_manifest=layer_manifest,
        )
        key_offset = path_append_bytes(key_path, key_bytes)
        value_offset = path_append_bytes(value_path, value_bytes)
        extent = KVStreamExtentMetadata(
            start_token=start_token,
            end_token=end_token,
            key_dtype=dtype_name(key_tensor.dtype),
            value_dtype=dtype_name(value_tensor.dtype),
            key_shape=tuple(key_tensor.shape),
            value_shape=tuple(value_tensor.shape),
            key_path=str(key_path.relative_to(self.cache_folder)),
            value_path=str(value_path.relative_to(self.cache_folder)),
            key_offset=key_offset,
            value_offset=value_offset,
            key_nbytes=len(key_bytes),
            value_nbytes=len(value_bytes),
        )
        updated_manifest = KVStreamLayerManifest(
            layer_idx=layer_idx,
            layout=STREAMED_CACHE_LAYOUT,
            sequence_axis=SEQUENCE_AXIS,
            persisted_tokens=end_token,
            segment_bytes_target=layer_manifest.segment_bytes_target,
            extents=layer_manifest.extents + (extent,),
        )
        validate_stream_layer_manifest(updated_manifest)
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
        total_extents = 0
        for layer_idx in self.persisted_layer_ids():
            manifest = self._read_layer_manifest(layer_idx)
            if manifest is None:
                continue
            total_extents += len(manifest.extents)
        return total_extents

    def cold_store_format_id(self) -> str | None:
        return None

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

    def _target_segment_paths(
        self,
        *,
        layer_idx: int,
        key_bytes: int,
        value_bytes: int,
        layer_manifest: KVStreamLayerManifest,
    ) -> tuple[Path, Path]:
        layer_folder = self.layers_folder / str(layer_idx)
        key_folder = layer_folder / "key"
        value_folder = layer_folder / "value"
        path_mkdir(key_folder, parents=True, exist_ok=True)
        path_mkdir(value_folder, parents=True, exist_ok=True)

        segment_index = 0
        if layer_manifest.extents:
            last_extent = layer_manifest.extents[-1]
            last_key_path = self.cache_folder / last_extent.key_path
            last_value_path = self.cache_folder / last_extent.value_path
            segment_index = self._segment_index_from_path(last_key_path)
            if (
                path_file_size(last_key_path) + key_bytes
                > layer_manifest.segment_bytes_target
                or path_file_size(last_value_path) + value_bytes
                > layer_manifest.segment_bytes_target
            ):
                segment_index += 1

        segment_name = f"segment-{segment_index:06d}.bin"
        return key_folder / segment_name, value_folder / segment_name

    def _segment_index_from_path(self, path: Path) -> int:
        stem = path.stem
        prefix = "segment-"
        if not stem.startswith(prefix):
            raise ValueError(f"Unexpected streamed KV segment name: {path.name}")
        try:
            return int(stem.removeprefix(prefix))
        except ValueError as exc:
            raise ValueError(
                f"Unexpected streamed KV segment name: {path.name}"
            ) from exc

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
        if require_int(payload, "segment_bytes_target") <= 0:
            raise ValueError("segment_bytes_target must be greater than zero")
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
                    "segment_bytes_target": self.segment_bytes_target,
                    "layers": list(layers),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        self._root_manifest_cache = (layers, policy_id)

    def _read_layer_manifest(self, layer_idx: int) -> KVStreamLayerManifest | None:
        if layer_idx in self._layer_manifest_cache:
            return self._layer_manifest_cache[layer_idx]
        layer_folder = self.layers_folder / str(layer_idx)
        if not path_exists(layer_folder):
            self._layer_manifest_cache[layer_idx] = None
            return None
        manifest_path = layer_folder / "manifest.json"
        if not path_exists(manifest_path):
            raise ValueError(f"KV layer manifest is missing: {manifest_path}")
        manifest = KVStreamLayerManifest.from_dict(read_json_object(manifest_path))
        self._layer_manifest_cache[layer_idx] = manifest
        return manifest

    def _write_layer_manifest(self, manifest: KVStreamLayerManifest) -> None:
        layer_folder = self.layers_folder / str(manifest.layer_idx)
        path_mkdir(layer_folder, parents=True, exist_ok=True)
        manifest_path = layer_folder / "manifest.json"
        atomic_write_text(
            manifest_path,
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        self._layer_manifest_cache[manifest.layer_idx] = manifest

    def _read_extent_tensors(
        self,
        extents: tuple[KVStreamExtentMetadata, ...],
        *,
        kind: str,
    ) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        grouped_extents: list[KVStreamExtentMetadata] = []
        current_path: Path | None = None
        for extent in extents:
            extent_path = self._extent_path(extent, kind=kind)
            if current_path is None:
                current_path = extent_path
                grouped_extents.append(extent)
                continue
            if extent_path == current_path:
                grouped_extents.append(extent)
                continue
            tensors.extend(
                self._decode_extent_group(
                    current_path, tuple(grouped_extents), kind=kind
                )
            )
            current_path = extent_path
            grouped_extents = [extent]
        if current_path is not None:
            tensors.extend(
                self._decode_extent_group(
                    current_path, tuple(grouped_extents), kind=kind
                )
            )
        return tensors

    def _decode_extent_group(
        self,
        path: Path,
        extents: tuple[KVStreamExtentMetadata, ...],
        *,
        kind: str,
    ) -> list[torch.Tensor]:
        self._validate_extent_path(path)
        offsets = [self._extent_offset(extent, kind=kind) for extent in extents]
        lengths = [self._extent_length(extent, kind=kind) for extent in extents]
        window_start = min(offsets)
        window_end = max(offset + length for offset, length in zip(offsets, lengths))
        raw_window = path_read_bytes_range(
            path, offset=window_start, length=window_end - window_start
        )
        tensors: list[torch.Tensor] = []
        for extent in extents:
            offset = self._extent_offset(extent, kind=kind)
            length = self._extent_length(extent, kind=kind)
            relative_offset = offset - window_start
            relative_end = relative_offset + length
            tensors.append(
                decode_tensor_bytes(
                    raw_window[relative_offset:relative_end],
                    dtype=dtype_from_name(self._extent_dtype(extent, kind=kind)),
                    shape=self._extent_shape(extent, kind=kind),
                )
            )
        return tensors

    def _extent_path(self, extent: KVStreamExtentMetadata, *, kind: str) -> Path:
        if kind == "key":
            return self.cache_folder / extent.key_path
        if kind == "value":
            return self.cache_folder / extent.value_path
        raise ValueError(f"Unsupported streamed extent kind: {kind}")

    def _extent_dtype(self, extent: KVStreamExtentMetadata, *, kind: str) -> str:
        if kind == "key":
            return extent.key_dtype
        if kind == "value":
            return extent.value_dtype
        raise ValueError(f"Unsupported streamed extent kind: {kind}")

    def _extent_shape(
        self, extent: KVStreamExtentMetadata, *, kind: str
    ) -> tuple[int, ...]:
        if kind == "key":
            return extent.key_shape
        if kind == "value":
            return extent.value_shape
        raise ValueError(f"Unsupported streamed extent kind: {kind}")

    def _extent_offset(self, extent: KVStreamExtentMetadata, *, kind: str) -> int:
        if kind == "key":
            return extent.key_offset
        if kind == "value":
            return extent.value_offset
        raise ValueError(f"Unsupported streamed extent kind: {kind}")

    def _extent_length(self, extent: KVStreamExtentMetadata, *, kind: str) -> int:
        if kind == "key":
            return extent.key_nbytes
        if kind == "value":
            return extent.value_nbytes
        raise ValueError(f"Unsupported streamed extent kind: {kind}")

    def _validate_extent_path(self, path: Path) -> None:
        resolved_path = path.resolve()
        cache_root = self.cache_folder.resolve()
        if not resolved_path.is_relative_to(cache_root):
            raise ValueError(f"KV cache chunk path escapes cache root: {path}")
        if not path_exists(path):
            raise ValueError(f"KV cache chunk file is missing: {path}")

    def _read_extent_tensor(
        self,
        path: Path,
        *,
        dtype_name_value: str,
        shape: tuple[int, ...],
        offset: int,
        length: int,
    ) -> torch.Tensor:
        resolved_path = path.resolve()
        cache_root = self.cache_folder.resolve()
        if not resolved_path.is_relative_to(cache_root):
            raise ValueError(f"KV cache chunk path escapes cache root: {path}")
        if not path_exists(path):
            raise ValueError(f"KV cache chunk file is missing: {path}")
        return decode_tensor_bytes(
            path_read_bytes_range(path, offset=offset, length=length),
            dtype=dtype_from_name(dtype_name_value),
            shape=shape,
        )

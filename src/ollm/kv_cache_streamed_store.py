"""Streamed append/read disk KV store with explicit segment metadata."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ollm.async_io import (
    path_append_bytes,
    path_exists,
    path_file_size,
    path_mkdir,
    path_read_bytes_range,
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
    require_object_list,
    require_relative_path,
    require_shape,
    require_str,
    sequence_length,
    shape_prefix,
)

_CACHE_FORMAT = "ollm-kv-streamed-segmented"
_CACHE_LAYOUT = "streamed-segmented"
_DEFAULT_SEGMENT_BYTES_TARGET = 8 * 1024 * 1024


@dataclass(slots=True, frozen=True)
class KVStreamExtentMetadata:
    start_token: int
    end_token: int
    key_dtype: str
    value_dtype: str
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_path: str
    value_path: str
    key_offset: int
    value_offset: int
    key_nbytes: int
    value_nbytes: int

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    def to_dict(self) -> dict[str, object]:
        return {
            "start_token": self.start_token,
            "end_token": self.end_token,
            "key_dtype": self.key_dtype,
            "value_dtype": self.value_dtype,
            "key_shape": list(self.key_shape),
            "value_shape": list(self.value_shape),
            "key_path": self.key_path,
            "value_path": self.value_path,
            "key_offset": self.key_offset,
            "value_offset": self.value_offset,
            "key_nbytes": self.key_nbytes,
            "value_nbytes": self.value_nbytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        start_token = require_int(payload, "start_token")
        end_token = require_int(payload, "end_token")
        if end_token <= start_token:
            raise ValueError(
                f"Invalid KV extent token range: start={start_token} end={end_token}"
            )
        key_offset = require_int(payload, "key_offset")
        value_offset = require_int(payload, "value_offset")
        key_nbytes = require_int(payload, "key_nbytes")
        value_nbytes = require_int(payload, "value_nbytes")
        if key_offset < 0 or value_offset < 0:
            raise ValueError("KV extent offsets must be zero or greater")
        if key_nbytes <= 0 or value_nbytes <= 0:
            raise ValueError("KV extent byte lengths must be greater than zero")
        return cls(
            start_token=start_token,
            end_token=end_token,
            key_dtype=require_str(payload, "key_dtype"),
            value_dtype=require_str(payload, "value_dtype"),
            key_shape=require_shape(payload, "key_shape"),
            value_shape=require_shape(payload, "value_shape"),
            key_path=require_relative_path(payload, "key_path"),
            value_path=require_relative_path(payload, "value_path"),
            key_offset=key_offset,
            value_offset=value_offset,
            key_nbytes=key_nbytes,
            value_nbytes=value_nbytes,
        )


@dataclass(slots=True, frozen=True)
class KVStreamLayerManifest:
    layer_idx: int
    layout: str
    sequence_axis: int
    persisted_tokens: int
    segment_bytes_target: int
    extents: tuple[KVStreamExtentMetadata, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_idx": self.layer_idx,
            "layout": self.layout,
            "sequence_axis": self.sequence_axis,
            "persisted_tokens": self.persisted_tokens,
            "segment_bytes_target": self.segment_bytes_target,
            "extents": [extent.to_dict() for extent in self.extents],
        }

    @classmethod
    def new(cls, layer_idx: int, *, segment_bytes_target: int):
        return cls(
            layer_idx=layer_idx,
            layout=_CACHE_LAYOUT,
            sequence_axis=SEQUENCE_AXIS,
            persisted_tokens=0,
            segment_bytes_target=segment_bytes_target,
            extents=(),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        manifest = cls(
            layer_idx=require_int(payload, "layer_idx"),
            layout=require_str(payload, "layout"),
            sequence_axis=require_int(payload, "sequence_axis"),
            persisted_tokens=require_int(payload, "persisted_tokens"),
            segment_bytes_target=require_int(payload, "segment_bytes_target"),
            extents=tuple(
                KVStreamExtentMetadata.from_dict(extent_payload)
                for extent_payload in require_object_list(
                    payload.get("extents"), "extents"
                )
            ),
        )
        _validate_layer_manifest(manifest)
        return manifest


def _validate_layer_manifest(manifest: KVStreamLayerManifest) -> None:
    if manifest.layout != _CACHE_LAYOUT:
        raise ValueError(f"Unsupported streamed KV layout: {manifest.layout!r}")
    if manifest.sequence_axis != SEQUENCE_AXIS:
        raise ValueError(
            f"Unsupported streamed KV sequence axis: {manifest.sequence_axis}"
        )
    if manifest.segment_bytes_target <= 0:
        raise ValueError("segment_bytes_target must be greater than zero")
    if not manifest.extents:
        raise ValueError(f"KV layer manifest {manifest.layer_idx} has no extents")
    next_expected_start = 0
    reference_key_dtype = manifest.extents[0].key_dtype
    reference_value_dtype = manifest.extents[0].value_dtype
    reference_key_prefix = shape_prefix(manifest.extents[0].key_shape)
    reference_value_prefix = shape_prefix(manifest.extents[0].value_shape)
    for extent in manifest.extents:
        if extent.start_token != next_expected_start:
            raise ValueError(
                f"KV layer {manifest.layer_idx} has non-contiguous extent ranges"
            )
        if sequence_length(extent.key_shape) != extent.token_count:
            raise ValueError(
                f"KV key extent shape does not match token range for layer {manifest.layer_idx}"
            )
        if sequence_length(extent.value_shape) != extent.token_count:
            raise ValueError(
                f"KV value extent shape does not match token range for layer {manifest.layer_idx}"
            )
        if extent.key_dtype != reference_key_dtype:
            raise ValueError(
                f"KV layer {manifest.layer_idx} key dtype changed across extents"
            )
        if extent.value_dtype != reference_value_dtype:
            raise ValueError(
                f"KV layer {manifest.layer_idx} value dtype changed across extents"
            )
        if shape_prefix(extent.key_shape) != reference_key_prefix:
            raise ValueError(
                f"KV layer {manifest.layer_idx} key shape prefix changed across extents"
            )
        if shape_prefix(extent.value_shape) != reference_value_prefix:
            raise ValueError(
                f"KV layer {manifest.layer_idx} value shape prefix changed across extents"
            )
        next_expected_start = extent.end_token
    if manifest.persisted_tokens != next_expected_start:
        raise ValueError(
            f"KV layer {manifest.layer_idx} persisted_tokens does not match extent coverage"
        )


class StreamedSegmentedKVStore:
    def __init__(
        self,
        cache_folder: Path,
        *,
        segment_bytes_target: int = _DEFAULT_SEGMENT_BYTES_TARGET,
    ) -> None:
        self.cache_folder = cache_folder
        self.layers_folder = cache_folder / "layers"
        self.root_manifest_path = cache_folder / "manifest.json"
        self.segment_bytes_target = segment_bytes_target

    def initialize(self, policy_id: str) -> None:
        path_mkdir(self.layers_folder, parents=True, exist_ok=True)
        self._write_root_manifest((), policy_id)

    def load_layer(
        self, layer_idx: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        layer_manifest = self._read_layer_manifest(layer_idx)
        if layer_manifest is None:
            return None
        _ = self._read_root_manifest()
        key_chunks: list[torch.Tensor] = []
        value_chunks: list[torch.Tensor] = []
        for extent in layer_manifest.extents:
            key_chunks.append(
                self._read_extent_tensor(
                    self.cache_folder / extent.key_path,
                    dtype_name_value=extent.key_dtype,
                    shape=extent.key_shape,
                    offset=extent.key_offset,
                    length=extent.key_nbytes,
                )
            )
            value_chunks.append(
                self._read_extent_tensor(
                    self.cache_folder / extent.value_path,
                    dtype_name_value=extent.value_dtype,
                    shape=extent.value_shape,
                    offset=extent.value_offset,
                    length=extent.value_nbytes,
                )
            )
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
            layout=_CACHE_LAYOUT,
            sequence_axis=SEQUENCE_AXIS,
            persisted_tokens=end_token,
            segment_bytes_target=layer_manifest.segment_bytes_target,
            extents=layer_manifest.extents + (extent,),
        )
        _validate_layer_manifest(updated_manifest)
        self._write_layer_manifest(updated_manifest)

        root_layers, policy_id = self._read_root_manifest()
        if layer_idx not in root_layers:
            self._write_root_manifest(
                tuple(sorted(root_layers + (layer_idx,))), policy_id
            )

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
        return (
            tuple(require_int_value(value, "layers[]") for value in layers_payload),
            policy_id,
        )

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

    def _read_layer_manifest(self, layer_idx: int) -> KVStreamLayerManifest | None:
        layer_folder = self.layers_folder / str(layer_idx)
        if not path_exists(layer_folder):
            return None
        manifest_path = layer_folder / "manifest.json"
        if not path_exists(manifest_path):
            raise ValueError(f"KV layer manifest is missing: {manifest_path}")
        return KVStreamLayerManifest.from_dict(read_json_object(manifest_path))

    def _write_layer_manifest(self, manifest: KVStreamLayerManifest) -> None:
        layer_folder = self.layers_folder / str(manifest.layer_idx)
        path_mkdir(layer_folder, parents=True, exist_ok=True)
        manifest_path = layer_folder / "manifest.json"
        atomic_write_text(
            manifest_path,
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        )

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

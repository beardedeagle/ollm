import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ollm.async_io import path_exists, path_mkdir
from ollm.kv_cache_store_common import (
    CACHE_SCHEMA_VERSION,
    PERSISTED_DEVICE,
    SEQUENCE_AXIS,
    atomic_write_bytes,
    atomic_write_text,
    dtype_name,
    encode_tensor_bytes,
    read_and_decode_tensor,
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

_CACHE_FORMAT = "ollm-kv-chunked"
_CACHE_LAYOUT = "contiguous"


@dataclass(slots=True, frozen=True)
class KVChunkMetadata:
    start_token: int
    end_token: int
    key_dtype: str
    value_dtype: str
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_path: str
    value_path: str

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
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        start_token = require_int(payload, "start_token")
        end_token = require_int(payload, "end_token")
        if end_token <= start_token:
            raise ValueError(
                f"Invalid KV chunk token range: start={start_token} end={end_token}"
            )
        return cls(
            start_token=start_token,
            end_token=end_token,
            key_dtype=require_str(payload, "key_dtype"),
            value_dtype=require_str(payload, "value_dtype"),
            key_shape=require_shape(payload, "key_shape"),
            value_shape=require_shape(payload, "value_shape"),
            key_path=require_relative_path(payload, "key_path"),
            value_path=require_relative_path(payload, "value_path"),
        )


@dataclass(slots=True, frozen=True)
class KVLayerManifest:
    layer_idx: int
    layout: str
    sequence_axis: int
    persisted_tokens: int
    chunks: tuple[KVChunkMetadata, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_idx": self.layer_idx,
            "layout": self.layout,
            "sequence_axis": self.sequence_axis,
            "persisted_tokens": self.persisted_tokens,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def new(cls, layer_idx: int):
        return cls(
            layer_idx=layer_idx,
            layout=_CACHE_LAYOUT,
            sequence_axis=SEQUENCE_AXIS,
            persisted_tokens=0,
            chunks=(),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        manifest = cls(
            layer_idx=require_int(payload, "layer_idx"),
            layout=require_str(payload, "layout"),
            sequence_axis=require_int(payload, "sequence_axis"),
            persisted_tokens=require_int(payload, "persisted_tokens"),
            chunks=tuple(
                KVChunkMetadata.from_dict(chunk_payload)
                for chunk_payload in require_object_list(
                    payload.get("chunks"), "chunks"
                )
            ),
        )
        _validate_layer_manifest(manifest)
        return manifest


def _validate_layer_manifest(manifest: KVLayerManifest) -> None:
    if manifest.layout != _CACHE_LAYOUT:
        raise ValueError(f"Unsupported KV cache layout: {manifest.layout!r}")
    if manifest.sequence_axis != SEQUENCE_AXIS:
        raise ValueError(
            f"Unsupported KV cache sequence axis: {manifest.sequence_axis}"
        )
    if not manifest.chunks:
        raise ValueError(f"KV layer manifest {manifest.layer_idx} has no chunks")
    next_expected_start = 0
    reference_key_dtype = manifest.chunks[0].key_dtype
    reference_value_dtype = manifest.chunks[0].value_dtype
    reference_key_prefix = shape_prefix(manifest.chunks[0].key_shape)
    reference_value_prefix = shape_prefix(manifest.chunks[0].value_shape)
    for chunk in manifest.chunks:
        if chunk.start_token != next_expected_start:
            raise ValueError(
                f"KV layer {manifest.layer_idx} has non-contiguous chunk ranges"
            )
        if sequence_length(chunk.key_shape) != chunk.token_count:
            raise ValueError(
                f"KV key chunk shape does not match token range for layer {manifest.layer_idx}"
            )
        if sequence_length(chunk.value_shape) != chunk.token_count:
            raise ValueError(
                f"KV value chunk shape does not match token range for layer {manifest.layer_idx}"
            )
        if chunk.key_dtype != reference_key_dtype:
            raise ValueError(
                f"KV layer {manifest.layer_idx} key dtype changed across chunks"
            )
        if chunk.value_dtype != reference_value_dtype:
            raise ValueError(
                f"KV layer {manifest.layer_idx} value dtype changed across chunks"
            )
        if shape_prefix(chunk.key_shape) != reference_key_prefix:
            raise ValueError(
                f"KV layer {manifest.layer_idx} key shape prefix changed across chunks"
            )
        if shape_prefix(chunk.value_shape) != reference_value_prefix:
            raise ValueError(
                f"KV layer {manifest.layer_idx} value shape prefix changed across chunks"
            )
        next_expected_start = chunk.end_token
    if manifest.persisted_tokens != next_expected_start:
        raise ValueError(
            f"KV layer {manifest.layer_idx} persisted_tokens does not match chunk coverage"
        )


class ChunkedKVStore:
    def __init__(self, cache_folder: Path) -> None:
        self.cache_folder = cache_folder
        self.layers_folder = cache_folder / "layers"
        self.root_manifest_path = cache_folder / "manifest.json"
        self._root_manifest_cache: tuple[tuple[int, ...], str] | None = None
        self._layer_manifest_cache: dict[int, KVLayerManifest | None] = {}

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
        key_chunks: list[torch.Tensor] = []
        value_chunks: list[torch.Tensor] = []
        for chunk in layer_manifest.chunks:
            key_chunks.append(
                self._read_chunk_tensor(
                    self.cache_folder / chunk.key_path,
                    dtype_name=chunk.key_dtype,
                    shape=chunk.key_shape,
                )
            )
            value_chunks.append(
                self._read_chunk_tensor(
                    self.cache_folder / chunk.value_path,
                    dtype_name=chunk.value_dtype,
                    shape=chunk.value_shape,
                )
            )
        key_tensor = torch.cat(key_chunks, dim=SEQUENCE_AXIS).to(device)
        value_tensor = torch.cat(value_chunks, dim=SEQUENCE_AXIS).to(device)
        return key_tensor, value_tensor

    def append_layer_chunk(
        self, layer_idx: int, tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        key_tensor = tensors[0].detach().cpu().contiguous()
        value_tensor = tensors[1].detach().cpu().contiguous()
        self._validate_chunk_pair(layer_idx, key_tensor, value_tensor)
        token_count = sequence_length(tuple(key_tensor.shape))
        if token_count == 0:
            return
        layer_manifest = self._read_layer_manifest(layer_idx) or KVLayerManifest.new(
            layer_idx
        )
        start_token = layer_manifest.persisted_tokens
        end_token = start_token + token_count
        layer_folder = self.layers_folder / str(layer_idx)
        key_folder = layer_folder / "key"
        value_folder = layer_folder / "value"
        path_mkdir(key_folder, parents=True, exist_ok=True)
        path_mkdir(value_folder, parents=True, exist_ok=True)

        chunk_name = f"{start_token:012d}-{end_token:012d}.bin"
        key_path = key_folder / chunk_name
        value_path = value_folder / chunk_name
        atomic_write_bytes(key_path, encode_tensor_bytes(key_tensor))
        atomic_write_bytes(value_path, encode_tensor_bytes(value_tensor))

        chunk = KVChunkMetadata(
            start_token=start_token,
            end_token=end_token,
            key_dtype=dtype_name(key_tensor.dtype),
            value_dtype=dtype_name(value_tensor.dtype),
            key_shape=tuple(key_tensor.shape),
            value_shape=tuple(value_tensor.shape),
            key_path=str(key_path.relative_to(self.cache_folder)),
            value_path=str(value_path.relative_to(self.cache_folder)),
        )
        updated_manifest = KVLayerManifest(
            layer_idx=layer_idx,
            layout=_CACHE_LAYOUT,
            sequence_axis=SEQUENCE_AXIS,
            persisted_tokens=end_token,
            chunks=layer_manifest.chunks + (chunk,),
        )
        _validate_layer_manifest(updated_manifest)
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
        total_chunks = 0
        for layer_idx in self.persisted_layer_ids():
            manifest = self._read_layer_manifest(layer_idx)
            if manifest is None:
                continue
            total_chunks += len(manifest.chunks)
        return total_chunks

    def cold_store_format_id(self) -> str | None:
        return None

    def cold_tier_representation_id(self) -> str | None:
        return None

    def compaction_count(self) -> int:
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
                    "layers": list(layers),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        self._root_manifest_cache = (layers, policy_id)

    def _read_layer_manifest(self, layer_idx: int) -> KVLayerManifest | None:
        if layer_idx in self._layer_manifest_cache:
            return self._layer_manifest_cache[layer_idx]
        layer_folder = self.layers_folder / str(layer_idx)
        if not path_exists(layer_folder):
            self._layer_manifest_cache[layer_idx] = None
            return None
        manifest_path = layer_folder / "manifest.json"
        if not path_exists(manifest_path):
            raise ValueError(f"KV layer manifest is missing: {manifest_path}")
        manifest = KVLayerManifest.from_dict(read_json_object(manifest_path))
        self._layer_manifest_cache[layer_idx] = manifest
        return manifest

    def _write_layer_manifest(self, manifest: KVLayerManifest) -> None:
        layer_folder = self.layers_folder / str(manifest.layer_idx)
        path_mkdir(layer_folder, parents=True, exist_ok=True)
        manifest_path = layer_folder / "manifest.json"
        atomic_write_text(
            manifest_path,
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        self._layer_manifest_cache[manifest.layer_idx] = manifest

    def _read_chunk_tensor(
        self, path: Path, *, dtype_name: str, shape: tuple[int, ...]
    ) -> torch.Tensor:
        resolved_path = path.resolve()
        cache_root = self.cache_folder.resolve()
        if not resolved_path.is_relative_to(cache_root):
            raise ValueError(f"KV cache chunk path escapes cache root: {path}")
        if not path_exists(path):
            raise ValueError(f"KV cache chunk file is missing: {path}")
        return read_and_decode_tensor(path, dtype_name_value=dtype_name, shape=shape)

"""Bounded sliding-window KV store with overwrite semantics."""

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
    require_relative_path,
    require_shape,
    require_str,
    sequence_length,
    shape_prefix,
)

_CACHE_FORMAT = "ollm-kv-sliding-window"
_CACHE_LAYOUT = "ring-buffer-tail"


@dataclass(slots=True, frozen=True)
class KVSlidingWindowLayerManifest:
    """Describe one bounded persisted layer snapshot."""

    layer_idx: int
    layout: str
    sequence_axis: int
    persisted_tokens: int
    window_max_tokens: int
    eviction_count: int
    evicted_tokens: int
    key_dtype: str
    value_dtype: str
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_path: str
    value_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_idx": self.layer_idx,
            "layout": self.layout,
            "sequence_axis": self.sequence_axis,
            "persisted_tokens": self.persisted_tokens,
            "window_max_tokens": self.window_max_tokens,
            "eviction_count": self.eviction_count,
            "evicted_tokens": self.evicted_tokens,
            "key_dtype": self.key_dtype,
            "value_dtype": self.value_dtype,
            "key_shape": list(self.key_shape),
            "value_shape": list(self.value_shape),
            "key_path": self.key_path,
            "value_path": self.value_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        manifest = cls(
            layer_idx=require_int(payload, "layer_idx"),
            layout=require_str(payload, "layout"),
            sequence_axis=require_int(payload, "sequence_axis"),
            persisted_tokens=require_int(payload, "persisted_tokens"),
            window_max_tokens=require_int(payload, "window_max_tokens"),
            eviction_count=require_int(payload, "eviction_count"),
            evicted_tokens=require_int(payload, "evicted_tokens"),
            key_dtype=require_str(payload, "key_dtype"),
            value_dtype=require_str(payload, "value_dtype"),
            key_shape=require_shape(payload, "key_shape"),
            value_shape=require_shape(payload, "value_shape"),
            key_path=require_relative_path(payload, "key_path"),
            value_path=require_relative_path(payload, "value_path"),
        )
        _validate_layer_manifest(manifest)
        return manifest


def _validate_layer_manifest(manifest: KVSlidingWindowLayerManifest) -> None:
    if manifest.layout != _CACHE_LAYOUT:
        raise ValueError(f"Unsupported KV cache layout: {manifest.layout!r}")
    if manifest.sequence_axis != SEQUENCE_AXIS:
        raise ValueError(
            f"Unsupported KV cache sequence axis: {manifest.sequence_axis}"
        )
    if manifest.window_max_tokens <= 0:
        raise ValueError("Sliding-window max tokens must be greater than zero")
    if manifest.persisted_tokens <= 0:
        raise ValueError(
            "Sliding-window layer manifests must retain at least one token"
        )
    if manifest.persisted_tokens > manifest.window_max_tokens:
        raise ValueError("Sliding-window persisted tokens exceed the configured window")
    if sequence_length(manifest.key_shape) != manifest.persisted_tokens:
        raise ValueError("Sliding-window key shape does not match persisted_tokens")
    if sequence_length(manifest.value_shape) != manifest.persisted_tokens:
        raise ValueError("Sliding-window value shape does not match persisted_tokens")
    if shape_prefix(manifest.key_shape) != shape_prefix(manifest.value_shape):
        raise ValueError("Sliding-window key/value shape prefixes must match")


class SlidingWindowRingBufferKVStore:
    """Persist only the bounded recent tail for each layer."""

    def __init__(self, cache_folder: Path, *, window_max_tokens: int) -> None:
        if window_max_tokens <= 0:
            raise ValueError("window_max_tokens must be greater than zero")
        self.cache_folder = cache_folder
        self.layers_folder = cache_folder / "layers"
        self.root_manifest_path = cache_folder / "manifest.json"
        self.window_max_tokens = window_max_tokens
        self._root_manifest_cache: tuple[tuple[int, ...], str] | None = None
        self._layer_manifest_cache: dict[int, KVSlidingWindowLayerManifest | None] = {}

    def initialize(self, policy_id: str) -> None:
        path_mkdir(self.layers_folder, parents=True, exist_ok=True)
        self._layer_manifest_cache.clear()
        self._write_root_manifest((), policy_id)

    def load_layer(
        self, layer_idx: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        manifest = self._read_layer_manifest(layer_idx)
        if manifest is None:
            return None
        _ = self._read_root_manifest()
        return (
            read_and_decode_tensor(
                self._validate_data_path(self.cache_folder / manifest.key_path),
                dtype_name_value=manifest.key_dtype,
                shape=manifest.key_shape,
            ).to(device),
            read_and_decode_tensor(
                self._validate_data_path(self.cache_folder / manifest.value_path),
                dtype_name_value=manifest.value_dtype,
                shape=manifest.value_shape,
            ).to(device),
        )

    def append_layer_chunk(
        self, layer_idx: int, tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        key_delta = tensors[0].detach().cpu().contiguous()
        value_delta = tensors[1].detach().cpu().contiguous()
        self._validate_chunk_pair(layer_idx, key_delta, value_delta)
        token_count = sequence_length(tuple(key_delta.shape))
        if token_count == 0:
            return

        current_manifest = self._read_layer_manifest(layer_idx)
        if current_manifest is None:
            combined_key = key_delta
            combined_value = value_delta
            eviction_count = 0
            evicted_tokens = 0
        else:
            current_key = read_and_decode_tensor(
                self._validate_data_path(self.cache_folder / current_manifest.key_path),
                dtype_name_value=current_manifest.key_dtype,
                shape=current_manifest.key_shape,
            )
            current_value = read_and_decode_tensor(
                self._validate_data_path(
                    self.cache_folder / current_manifest.value_path
                ),
                dtype_name_value=current_manifest.value_dtype,
                shape=current_manifest.value_shape,
            )
            combined_key = torch.cat((current_key, key_delta), dim=SEQUENCE_AXIS)
            combined_value = torch.cat((current_value, value_delta), dim=SEQUENCE_AXIS)
            eviction_count = current_manifest.eviction_count
            evicted_tokens = current_manifest.evicted_tokens
        combined_tokens = sequence_length(tuple(combined_key.shape))
        evicted_now = max(0, combined_tokens - self.window_max_tokens)
        if evicted_now > 0:
            combined_key = combined_key[..., -self.window_max_tokens :, :].contiguous()
            combined_value = combined_value[
                ..., -self.window_max_tokens :, :
            ].contiguous()
            eviction_count += 1
            evicted_tokens += evicted_now

        retained_tokens = sequence_length(tuple(combined_key.shape))
        layer_folder = self.layers_folder / str(layer_idx)
        key_folder = layer_folder / "key"
        value_folder = layer_folder / "value"
        path_mkdir(key_folder, parents=True, exist_ok=True)
        path_mkdir(value_folder, parents=True, exist_ok=True)

        key_path = key_folder / "window.bin"
        value_path = value_folder / "window.bin"
        atomic_write_bytes(key_path, encode_tensor_bytes(combined_key))
        atomic_write_bytes(value_path, encode_tensor_bytes(combined_value))
        manifest = KVSlidingWindowLayerManifest(
            layer_idx=layer_idx,
            layout=_CACHE_LAYOUT,
            sequence_axis=SEQUENCE_AXIS,
            persisted_tokens=retained_tokens,
            window_max_tokens=self.window_max_tokens,
            eviction_count=eviction_count,
            evicted_tokens=evicted_tokens,
            key_dtype=dtype_name(combined_key.dtype),
            value_dtype=dtype_name(combined_value.dtype),
            key_shape=tuple(combined_key.shape),
            value_shape=tuple(combined_value.shape),
            key_path=str(key_path.relative_to(self.cache_folder)),
            value_path=str(value_path.relative_to(self.cache_folder)),
        )
        _validate_layer_manifest(manifest)
        self._write_layer_manifest(manifest)

        root_layers, policy_id = self._read_root_manifest()
        if layer_idx not in root_layers:
            self._write_root_manifest(
                tuple(sorted(root_layers + (layer_idx,))),
                policy_id,
            )

    def persisted_layer_ids(self) -> tuple[int, ...]:
        if not path_exists(self.root_manifest_path):
            return ()
        return self._read_root_manifest()[0]

    def persisted_token_count(self) -> int:
        total_tokens = 0
        for layer_idx in self.persisted_layer_ids():
            manifest = self._read_layer_manifest(layer_idx)
            if manifest is not None:
                total_tokens += manifest.persisted_tokens
        return total_tokens

    def persisted_artifact_count(self) -> int:
        return len(self.persisted_layer_ids())

    def cold_store_format_id(self) -> str | None:
        return _CACHE_FORMAT

    def cold_tier_representation_id(self) -> str | None:
        return None

    def compaction_count(self) -> int:
        return 0

    def consume_last_compaction_elapsed_seconds(self) -> float | None:
        return None

    def eviction_count(self) -> int:
        total_evictions = 0
        for layer_idx in self.persisted_layer_ids():
            manifest = self._read_layer_manifest(layer_idx)
            if manifest is not None:
                total_evictions += manifest.eviction_count
        return total_evictions

    def evicted_token_count(self) -> int:
        total_evicted_tokens = 0
        for layer_idx in self.persisted_layer_ids():
            manifest = self._read_layer_manifest(layer_idx)
            if manifest is not None:
                total_evicted_tokens += manifest.evicted_tokens
        return total_evicted_tokens

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
        if require_int(payload, "window_max_tokens") != self.window_max_tokens:
            raise ValueError(
                "Sliding-window cache manifest does not match the configured window"
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
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "format": _CACHE_FORMAT,
            "chunk_axis": SEQUENCE_AXIS,
            "persisted_device": PERSISTED_DEVICE,
            "policy_id": policy_id,
            "window_max_tokens": self.window_max_tokens,
            "layers": list(layers),
        }
        atomic_write_text(
            self.root_manifest_path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        self._root_manifest_cache = (layers, policy_id)

    def _read_layer_manifest(
        self, layer_idx: int
    ) -> KVSlidingWindowLayerManifest | None:
        if layer_idx in self._layer_manifest_cache:
            return self._layer_manifest_cache[layer_idx]
        manifest_path = self.layers_folder / str(layer_idx) / "manifest.json"
        if not path_exists(manifest_path):
            self._layer_manifest_cache[layer_idx] = None
            return None
        manifest = KVSlidingWindowLayerManifest.from_dict(
            read_json_object(manifest_path)
        )
        self._layer_manifest_cache[layer_idx] = manifest
        return manifest

    def _write_layer_manifest(self, manifest: KVSlidingWindowLayerManifest) -> None:
        manifest_path = self.layers_folder / str(manifest.layer_idx) / "manifest.json"
        atomic_write_text(
            manifest_path,
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        self._layer_manifest_cache[manifest.layer_idx] = manifest

    def _validate_data_path(self, path: Path) -> Path:
        resolved_path = path.resolve()
        cache_root = self.cache_folder.resolve()
        if not resolved_path.is_relative_to(cache_root):
            raise ValueError(f"KV cache data path escapes cache root: {path}")
        if not path_exists(path):
            raise ValueError(f"KV cache data file is missing: {path}")
        return path

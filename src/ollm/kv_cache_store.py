import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from ollm.async_io import (
    path_exists,
    path_mkdir,
    path_read_bytes,
    path_read_text,
    path_replace,
    path_write_bytes,
    path_write_text,
)

_CACHE_SCHEMA_VERSION = 1
_CACHE_FORMAT = "ollm-kv-chunked"
_CACHE_LAYOUT = "contiguous"
_PERSISTED_DEVICE = "cpu"
_SEQUENCE_AXIS = -2

_DTYPE_TO_NAME = {
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


def _sequence_length(shape: tuple[int, ...]) -> int:
    if len(shape) < 2:
        raise ValueError(f"KV cache tensors must have rank >= 2, got shape {shape}")
    return shape[_SEQUENCE_AXIS]


def _shape_prefix(shape: tuple[int, ...]) -> tuple[int, ...]:
    sequence_axis = len(shape) + _SEQUENCE_AXIS
    return shape[:sequence_axis] + shape[sequence_axis + 1 :]


def _dtype_name(dtype: torch.dtype) -> str:
    dtype_name = _DTYPE_TO_NAME.get(dtype)
    if dtype_name is None:
        raise ValueError(f"Unsupported KV cache dtype: {dtype}")
    return dtype_name


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = _NAME_TO_DTYPE.get(name)
    if dtype is None:
        raise ValueError(f"Unsupported KV cache dtype name: {name}")
    return dtype


def _encode_tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu_tensor = tensor.detach().cpu().contiguous()
    return cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _decode_tensor_bytes(
    raw_bytes: bytes, *, dtype: torch.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    element_size = torch.empty((), dtype=dtype).element_size()
    expected_bytes = element_size
    for dimension in shape:
        expected_bytes *= dimension
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Chunk payload size mismatch for dtype={dtype} shape={shape}: "
            f"expected {expected_bytes} bytes, got {len(raw_bytes)}"
        )
    writable_bytes = bytearray(raw_bytes)
    return torch.frombuffer(writable_bytes, dtype=dtype).clone().reshape(shape)


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path_read_text(path, encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    path_write_text(temp_path, content, encoding="utf-8")
    path_replace(temp_path, path)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    path_write_bytes(temp_path, content)
    path_replace(temp_path, path)


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
        start_token = _require_int(payload, "start_token")
        end_token = _require_int(payload, "end_token")
        if end_token <= start_token:
            raise ValueError(
                f"Invalid KV chunk token range: start={start_token} end={end_token}"
            )
        return cls(
            start_token=start_token,
            end_token=end_token,
            key_dtype=_require_str(payload, "key_dtype"),
            value_dtype=_require_str(payload, "value_dtype"),
            key_shape=_require_shape(payload, "key_shape"),
            value_shape=_require_shape(payload, "value_shape"),
            key_path=_require_chunk_path(payload, "key_path"),
            value_path=_require_chunk_path(payload, "value_path"),
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
            sequence_axis=_SEQUENCE_AXIS,
            persisted_tokens=0,
            chunks=(),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        manifest = cls(
            layer_idx=_require_int(payload, "layer_idx"),
            layout=_require_str(payload, "layout"),
            sequence_axis=_require_int(payload, "sequence_axis"),
            persisted_tokens=_require_int(payload, "persisted_tokens"),
            chunks=tuple(
                KVChunkMetadata.from_dict(chunk_payload)
                for chunk_payload in _require_object_list(
                    payload.get("chunks"), "chunks"
                )
            ),
        )
        _validate_layer_manifest(manifest)
        return manifest


def _require_object_list(
    payload: object, field_name: str
) -> tuple[dict[str, object], ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{field_name} must be a JSON list")
    validated: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"{field_name} entries must be JSON objects")
        validated.append(cast(dict[str, object], item))
    return tuple(validated)


def _require_int(payload: dict[str, object], field_name: str) -> int:
    return _require_int_value(payload.get(field_name), field_name)


def _require_int_value(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_str(payload: dict[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_chunk_path(payload: dict[str, object], field_name: str) -> str:
    value = _require_str(payload, field_name)
    chunk_path = Path(value)
    if chunk_path.is_absolute() or ".." in chunk_path.parts:
        raise ValueError(f"{field_name} must stay within the KV cache root")
    return value


def _require_shape(payload: dict[str, object], field_name: str) -> tuple[int, ...]:
    raw_shape = payload.get(field_name)
    if not isinstance(raw_shape, list) or not raw_shape:
        raise ValueError(f"{field_name} must be a non-empty integer list")
    shape = tuple(_require_int_value(value, field_name) for value in raw_shape)
    for value in shape:
        if value <= 0:
            raise ValueError(f"{field_name} values must be positive integers")
    return shape


def _validate_layer_manifest(manifest: KVLayerManifest) -> None:
    if manifest.layout != _CACHE_LAYOUT:
        raise ValueError(f"Unsupported KV cache layout: {manifest.layout!r}")
    if manifest.sequence_axis != _SEQUENCE_AXIS:
        raise ValueError(
            f"Unsupported KV cache sequence axis: {manifest.sequence_axis}"
        )
    if not manifest.chunks:
        raise ValueError(f"KV layer manifest {manifest.layer_idx} has no chunks")
    next_expected_start = 0
    reference_key_dtype = manifest.chunks[0].key_dtype
    reference_value_dtype = manifest.chunks[0].value_dtype
    reference_key_prefix = _shape_prefix(manifest.chunks[0].key_shape)
    reference_value_prefix = _shape_prefix(manifest.chunks[0].value_shape)
    for chunk in manifest.chunks:
        if chunk.start_token != next_expected_start:
            raise ValueError(
                f"KV layer {manifest.layer_idx} has non-contiguous chunk ranges"
            )
        if _sequence_length(chunk.key_shape) != chunk.token_count:
            raise ValueError(
                f"KV key chunk shape does not match token range for layer {manifest.layer_idx}"
            )
        if _sequence_length(chunk.value_shape) != chunk.token_count:
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
        if _shape_prefix(chunk.key_shape) != reference_key_prefix:
            raise ValueError(
                f"KV layer {manifest.layer_idx} key shape prefix changed across chunks"
            )
        if _shape_prefix(chunk.value_shape) != reference_value_prefix:
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

    def initialize(self) -> None:
        path_mkdir(self.layers_folder, parents=True, exist_ok=True)
        self._write_root_manifest(())

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
        key_tensor = torch.cat(key_chunks, dim=_SEQUENCE_AXIS).to(device)
        value_tensor = torch.cat(value_chunks, dim=_SEQUENCE_AXIS).to(device)
        return key_tensor, value_tensor

    def append_layer_chunk(
        self, layer_idx: int, tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        key_tensor = tensors[0].detach().cpu().contiguous()
        value_tensor = tensors[1].detach().cpu().contiguous()
        self._validate_chunk_pair(layer_idx, key_tensor, value_tensor)
        token_count = _sequence_length(tuple(key_tensor.shape))
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
        _atomic_write_bytes(key_path, _encode_tensor_bytes(key_tensor))
        _atomic_write_bytes(value_path, _encode_tensor_bytes(value_tensor))

        chunk = KVChunkMetadata(
            start_token=start_token,
            end_token=end_token,
            key_dtype=_dtype_name(key_tensor.dtype),
            value_dtype=_dtype_name(value_tensor.dtype),
            key_shape=tuple(key_tensor.shape),
            value_shape=tuple(value_tensor.shape),
            key_path=str(key_path.relative_to(self.cache_folder)),
            value_path=str(value_path.relative_to(self.cache_folder)),
        )
        updated_manifest = KVLayerManifest(
            layer_idx=layer_idx,
            layout=_CACHE_LAYOUT,
            sequence_axis=_SEQUENCE_AXIS,
            persisted_tokens=end_token,
            chunks=layer_manifest.chunks + (chunk,),
        )
        _validate_layer_manifest(updated_manifest)
        self._write_layer_manifest(updated_manifest)

        root_layers = self._read_root_manifest()
        if layer_idx not in root_layers:
            self._write_root_manifest(tuple(sorted(root_layers + (layer_idx,))))

    def _validate_chunk_pair(
        self, layer_idx: int, key_tensor: torch.Tensor, value_tensor: torch.Tensor
    ) -> None:
        key_shape = tuple(key_tensor.shape)
        value_shape = tuple(value_tensor.shape)
        if len(key_shape) < 2 or len(value_shape) < 2:
            raise ValueError(
                f"KV cache chunk tensors must have rank >= 2 for layer {layer_idx}"
            )
        if _sequence_length(key_shape) != _sequence_length(value_shape):
            raise ValueError(
                f"KV cache key/value token counts differ for layer {layer_idx}"
            )
        if _shape_prefix(key_shape) != _shape_prefix(value_shape):
            raise ValueError(
                f"KV cache key/value non-sequence dimensions differ for layer {layer_idx}"
            )
        if _sequence_length(key_shape) <= 0:
            raise ValueError(
                f"KV cache chunk token count must be positive for layer {layer_idx}"
            )

    def _read_root_manifest(self) -> tuple[int, ...]:
        if not path_exists(self.root_manifest_path):
            raise ValueError(
                f"KV cache root manifest is missing: {self.root_manifest_path}"
            )
        payload = _read_json_object(self.root_manifest_path)
        if _require_int(payload, "schema_version") != _CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported KV cache schema version: {payload['schema_version']}"
            )
        if _require_str(payload, "format") != _CACHE_FORMAT:
            raise ValueError(f"Unsupported KV cache format: {payload['format']!r}")
        if _require_int(payload, "chunk_axis") != _SEQUENCE_AXIS:
            raise ValueError(
                f"Unsupported KV cache chunk axis: {payload['chunk_axis']}"
            )
        if _require_str(payload, "persisted_device") != _PERSISTED_DEVICE:
            raise ValueError(
                f"Unsupported KV cache persisted device: {payload['persisted_device']!r}"
            )
        layers_payload = payload.get("layers")
        if not isinstance(layers_payload, list):
            raise ValueError("KV root manifest layers must be a JSON list")
        return tuple(_require_int_value(value, "layers[]") for value in layers_payload)

    def _write_root_manifest(self, layers: tuple[int, ...]) -> None:
        _atomic_write_text(
            self.root_manifest_path,
            json.dumps(
                {
                    "schema_version": _CACHE_SCHEMA_VERSION,
                    "format": _CACHE_FORMAT,
                    "chunk_axis": _SEQUENCE_AXIS,
                    "persisted_device": _PERSISTED_DEVICE,
                    "layers": list(layers),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def _read_layer_manifest(self, layer_idx: int) -> KVLayerManifest | None:
        layer_folder = self.layers_folder / str(layer_idx)
        if not path_exists(layer_folder):
            return None
        manifest_path = layer_folder / "manifest.json"
        if not path_exists(manifest_path):
            raise ValueError(f"KV layer manifest is missing: {manifest_path}")
        return KVLayerManifest.from_dict(_read_json_object(manifest_path))

    def _write_layer_manifest(self, manifest: KVLayerManifest) -> None:
        layer_folder = self.layers_folder / str(manifest.layer_idx)
        path_mkdir(layer_folder, parents=True, exist_ok=True)
        manifest_path = layer_folder / "manifest.json"
        _atomic_write_text(
            manifest_path,
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        )

    def _read_chunk_tensor(
        self, path: Path, *, dtype_name: str, shape: tuple[int, ...]
    ) -> torch.Tensor:
        resolved_path = path.resolve()
        cache_root = self.cache_folder.resolve()
        if not resolved_path.is_relative_to(cache_root):
            raise ValueError(f"KV cache chunk path escapes cache root: {path}")
        if not path_exists(path):
            raise ValueError(f"KV cache chunk file is missing: {path}")
        return _decode_tensor_bytes(
            path_read_bytes(path),
            dtype=_dtype_from_name(dtype_name),
            shape=shape,
        )

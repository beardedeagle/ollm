"""Shared helpers for disk-backed KV cache stores."""

import json
from pathlib import Path
from typing import cast
from uuid import uuid4

import torch

from ollm.async_io import (
    path_read_bytes,
    path_read_text,
    path_replace,
    path_write_bytes,
    path_write_text,
)

CACHE_SCHEMA_VERSION = 1
PERSISTED_DEVICE = "cpu"
SEQUENCE_AXIS = -2

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


def sequence_length(shape: tuple[int, ...]) -> int:
    """Return the token dimension length for a KV tensor shape."""

    if len(shape) < 2:
        raise ValueError(f"KV cache tensors must have rank >= 2, got shape {shape}")
    return shape[SEQUENCE_AXIS]


def shape_prefix(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the non-sequence dimensions for a KV tensor shape."""

    sequence_axis = len(shape) + SEQUENCE_AXIS
    return shape[:sequence_axis] + shape[sequence_axis + 1 :]


def dtype_name(dtype: torch.dtype) -> str:
    """Return the serialized dtype name for a supported tensor dtype."""

    serialized_name = _DTYPE_TO_NAME.get(dtype)
    if serialized_name is None:
        raise ValueError(f"Unsupported KV cache dtype: {dtype}")
    return serialized_name


def dtype_from_name(name: str) -> torch.dtype:
    """Return the torch dtype for a serialized KV dtype name."""

    dtype = _NAME_TO_DTYPE.get(name)
    if dtype is None:
        raise ValueError(f"Unsupported KV cache dtype name: {name}")
    return dtype


def encode_tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor into a raw contiguous CPU byte buffer."""

    cpu_tensor = tensor.detach().cpu().contiguous()
    return cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def decode_tensor_bytes(
    raw_bytes: bytes, *, dtype: torch.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    """Decode a raw byte buffer into a tensor with the expected dtype and shape."""

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


def read_json_object(path: Path) -> dict[str, object]:
    """Read a JSON object from disk and validate the top-level type."""

    payload = json.loads(path_read_text(path, encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a text file."""

    temp_path = _atomic_temp_path(path)
    path_write_text(temp_path, content, encoding="utf-8")
    path_replace(temp_path, path)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically replace a binary file."""

    temp_path = _atomic_temp_path(path)
    path_write_bytes(temp_path, content)
    path_replace(temp_path, path)


def _atomic_temp_path(path: Path) -> Path:
    """Return a unique same-directory temp path for atomic replacement."""

    return path.with_name(f"{path.name}.{uuid4().hex}.tmp")


def require_object_list(
    payload: object, field_name: str
) -> tuple[dict[str, object], ...]:
    """Validate a JSON list of object entries."""

    if not isinstance(payload, list):
        raise ValueError(f"{field_name} must be a JSON list")
    validated: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"{field_name} entries must be JSON objects")
        validated.append(cast(dict[str, object], item))
    return tuple(validated)


def require_int(payload: dict[str, object], field_name: str) -> int:
    """Read a required integer field from a JSON object."""

    return require_int_value(payload.get(field_name), field_name)


def require_int_value(value: object, field_name: str) -> int:
    """Validate a JSON integer value."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def require_str(payload: dict[str, object], field_name: str) -> str:
    """Read a required non-empty string field from a JSON object."""

    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def require_relative_path(payload: dict[str, object], field_name: str) -> str:
    """Read a required relative path field that stays under the cache root."""

    value = require_str(payload, field_name)
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{field_name} must stay within the KV cache root")
    return value


def require_shape(payload: dict[str, object], field_name: str) -> tuple[int, ...]:
    """Read a required positive integer shape list from a JSON object."""

    raw_shape = payload.get(field_name)
    if not isinstance(raw_shape, list) or not raw_shape:
        raise ValueError(f"{field_name} must be a non-empty integer list")
    shape = tuple(require_int_value(value, field_name) for value in raw_shape)
    for value in shape:
        if value <= 0:
            raise ValueError(f"{field_name} values must be positive integers")
    return shape


def read_and_decode_tensor(
    path: Path, *, dtype_name_value: str, shape: tuple[int, ...]
) -> torch.Tensor:
    """Read and decode a whole persisted tensor file."""

    return decode_tensor_bytes(
        path_read_bytes(path),
        dtype=dtype_from_name(dtype_name_value),
        shape=shape,
    )

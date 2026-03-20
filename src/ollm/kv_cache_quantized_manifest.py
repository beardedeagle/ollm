"""Manifest types and validation for the quantized journal-backed KV store."""

import math
from dataclasses import dataclass

import torch

from ollm.kv_cache_store_common import (
    SEQUENCE_AXIS,
    require_int,
    require_object_list,
    require_relative_path,
    require_shape,
    require_str,
    sequence_length,
    shape_prefix,
)

QUANTIZED_JOURNAL_CACHE_LAYOUT = "quantized-journal-append"
QUANTIZED_JOURNAL_FILE_NAME = "journal.bin"
DEFAULT_QUANTIZED_JOURNAL_COMPACTION_ENTRY_THRESHOLD = 8
QUANTIZED_COLD_TIER_REPRESENTATION = "int8-symmetric-per-tensor"
QUANTIZED_TENSOR_DTYPE = torch.int8
QUANTIZED_TENSOR_DTYPE_NAME = "int8"


@dataclass(slots=True, frozen=True)
class KVQuantizedJournalEntryMetadata:
    start_token: int
    end_token: int
    key_original_dtype: str
    value_original_dtype: str
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_offset: int
    value_offset: int
    key_nbytes: int
    value_nbytes: int
    key_scale: float
    value_scale: float

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    def to_dict(self) -> dict[str, object]:
        return {
            "start_token": self.start_token,
            "end_token": self.end_token,
            "key_original_dtype": self.key_original_dtype,
            "value_original_dtype": self.value_original_dtype,
            "key_shape": list(self.key_shape),
            "value_shape": list(self.value_shape),
            "key_offset": self.key_offset,
            "value_offset": self.value_offset,
            "key_nbytes": self.key_nbytes,
            "value_nbytes": self.value_nbytes,
            "key_scale": self.key_scale,
            "value_scale": self.value_scale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        start_token = require_int(payload, "start_token")
        end_token = require_int(payload, "end_token")
        if end_token <= start_token:
            raise ValueError(
                f"Invalid quantized KV journal token range: start={start_token} end={end_token}"
            )
        return cls(
            start_token=start_token,
            end_token=end_token,
            key_original_dtype=require_str(payload, "key_original_dtype"),
            value_original_dtype=require_str(payload, "value_original_dtype"),
            key_shape=require_shape(payload, "key_shape"),
            value_shape=require_shape(payload, "value_shape"),
            key_offset=require_int(payload, "key_offset"),
            value_offset=require_int(payload, "value_offset"),
            key_nbytes=require_int(payload, "key_nbytes"),
            value_nbytes=require_int(payload, "value_nbytes"),
            key_scale=_require_positive_float(payload, "key_scale"),
            value_scale=_require_positive_float(payload, "value_scale"),
        )


@dataclass(slots=True, frozen=True)
class KVQuantizedJournalLayerManifest:
    layer_idx: int
    layout: str
    sequence_axis: int
    persisted_tokens: int
    compaction_count: int
    key_journal_path: str
    value_journal_path: str
    entries: tuple[KVQuantizedJournalEntryMetadata, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_idx": self.layer_idx,
            "layout": self.layout,
            "sequence_axis": self.sequence_axis,
            "persisted_tokens": self.persisted_tokens,
            "compaction_count": self.compaction_count,
            "key_journal_path": self.key_journal_path,
            "value_journal_path": self.value_journal_path,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def new(cls, layer_idx: int, *, key_journal_path: str, value_journal_path: str):
        return cls(
            layer_idx=layer_idx,
            layout=QUANTIZED_JOURNAL_CACHE_LAYOUT,
            sequence_axis=SEQUENCE_AXIS,
            persisted_tokens=0,
            compaction_count=0,
            key_journal_path=key_journal_path,
            value_journal_path=value_journal_path,
            entries=(),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        manifest = cls(
            layer_idx=require_int(payload, "layer_idx"),
            layout=require_str(payload, "layout"),
            sequence_axis=require_int(payload, "sequence_axis"),
            persisted_tokens=require_int(payload, "persisted_tokens"),
            compaction_count=require_int(payload, "compaction_count"),
            key_journal_path=require_relative_path(payload, "key_journal_path"),
            value_journal_path=require_relative_path(payload, "value_journal_path"),
            entries=tuple(
                KVQuantizedJournalEntryMetadata.from_dict(entry_payload)
                for entry_payload in require_object_list(
                    payload.get("entries"), "entries"
                )
            ),
        )
        validate_quantized_journal_layer_manifest(manifest)
        return manifest


def validate_quantized_journal_layer_manifest(
    manifest: KVQuantizedJournalLayerManifest,
) -> None:
    if manifest.layout != QUANTIZED_JOURNAL_CACHE_LAYOUT:
        raise ValueError(f"Unsupported KV cache layout: {manifest.layout!r}")
    if manifest.sequence_axis != SEQUENCE_AXIS:
        raise ValueError(
            f"Unsupported KV cache sequence axis: {manifest.sequence_axis}"
        )
    if manifest.compaction_count < 0:
        raise ValueError("compaction_count must be zero or greater")
    if not manifest.entries:
        raise ValueError(f"KV layer manifest {manifest.layer_idx} has no entries")
    next_expected_start = 0
    next_expected_key_offset = 0
    next_expected_value_offset = 0
    reference_key_dtype = manifest.entries[0].key_original_dtype
    reference_value_dtype = manifest.entries[0].value_original_dtype
    reference_key_prefix = shape_prefix(manifest.entries[0].key_shape)
    reference_value_prefix = shape_prefix(manifest.entries[0].value_shape)
    for entry in manifest.entries:
        if entry.start_token != next_expected_start:
            raise ValueError(
                f"KV layer {manifest.layer_idx} has non-contiguous journal token ranges"
            )
        if entry.key_offset != next_expected_key_offset:
            raise ValueError(
                f"KV layer {manifest.layer_idx} has non-contiguous key journal offsets"
            )
        if entry.value_offset != next_expected_value_offset:
            raise ValueError(
                f"KV layer {manifest.layer_idx} has non-contiguous value journal offsets"
            )
        if sequence_length(entry.key_shape) != entry.token_count:
            raise ValueError(
                f"KV key journal shape does not match token range for layer {manifest.layer_idx}"
            )
        if sequence_length(entry.value_shape) != entry.token_count:
            raise ValueError(
                f"KV value journal shape does not match token range for layer {manifest.layer_idx}"
            )
        if entry.key_original_dtype != reference_key_dtype:
            raise ValueError(
                f"KV layer {manifest.layer_idx} key dtype changed across journal entries"
            )
        if entry.value_original_dtype != reference_value_dtype:
            raise ValueError(
                f"KV layer {manifest.layer_idx} value dtype changed across journal entries"
            )
        if shape_prefix(entry.key_shape) != reference_key_prefix:
            raise ValueError(
                f"KV layer {manifest.layer_idx} key shape prefix changed across journal entries"
            )
        if shape_prefix(entry.value_shape) != reference_value_prefix:
            raise ValueError(
                f"KV layer {manifest.layer_idx} value shape prefix changed across journal entries"
            )
        if entry.key_nbytes != _expected_quantized_nbytes(entry.key_shape):
            raise ValueError(
                f"KV layer {manifest.layer_idx} key journal entry size does not match quantized shape"
            )
        if entry.value_nbytes != _expected_quantized_nbytes(entry.value_shape):
            raise ValueError(
                f"KV layer {manifest.layer_idx} value journal entry size does not match quantized shape"
            )
        if not math.isfinite(entry.key_scale) or entry.key_scale <= 0:
            raise ValueError(
                f"KV layer {manifest.layer_idx} key quantization scale must be finite and positive"
            )
        if not math.isfinite(entry.value_scale) or entry.value_scale <= 0:
            raise ValueError(
                f"KV layer {manifest.layer_idx} value quantization scale must be finite and positive"
            )
        next_expected_start = entry.end_token
        next_expected_key_offset += entry.key_nbytes
        next_expected_value_offset += entry.value_nbytes
    if manifest.persisted_tokens != next_expected_start:
        raise ValueError(
            f"KV layer {manifest.layer_idx} persisted_tokens does not match journal coverage"
        )


def _expected_quantized_nbytes(shape: tuple[int, ...]) -> int:
    total_bytes = torch.empty((), dtype=QUANTIZED_TENSOR_DTYPE).element_size()
    for dimension in shape:
        total_bytes *= dimension
    return total_bytes


def _require_positive_float(payload: dict[str, object], field_name: str) -> float:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite positive number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{field_name} must be a finite positive number")
    return numeric_value

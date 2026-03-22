"""Manifest types and validation for the journal-backed KV cold store."""

from dataclasses import dataclass

import torch

from ollm.kv_cache.store_common import (
    SEQUENCE_AXIS,
    dtype_from_name,
    require_int,
    require_object_list,
    require_relative_path,
    require_shape,
    require_str,
    sequence_length,
    shape_prefix,
)

JOURNAL_CACHE_LAYOUT = "journal-append"
JOURNAL_FILE_NAME = "journal.bin"
DEFAULT_JOURNAL_COMPACTION_ENTRY_THRESHOLD = 8


@dataclass(slots=True, frozen=True)
class KVJournalEntryMetadata:
    start_token: int
    end_token: int
    key_dtype: str
    value_dtype: str
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
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
                f"Invalid KV journal token range: start={start_token} end={end_token}"
            )
        return cls(
            start_token=start_token,
            end_token=end_token,
            key_dtype=require_str(payload, "key_dtype"),
            value_dtype=require_str(payload, "value_dtype"),
            key_shape=require_shape(payload, "key_shape"),
            value_shape=require_shape(payload, "value_shape"),
            key_offset=require_int(payload, "key_offset"),
            value_offset=require_int(payload, "value_offset"),
            key_nbytes=require_int(payload, "key_nbytes"),
            value_nbytes=require_int(payload, "value_nbytes"),
        )


@dataclass(slots=True, frozen=True)
class KVJournalLayerManifest:
    layer_idx: int
    layout: str
    sequence_axis: int
    persisted_tokens: int
    compaction_count: int
    key_journal_path: str
    value_journal_path: str
    entries: tuple[KVJournalEntryMetadata, ...]

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
            layout=JOURNAL_CACHE_LAYOUT,
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
                KVJournalEntryMetadata.from_dict(entry_payload)
                for entry_payload in require_object_list(
                    payload.get("entries"), "entries"
                )
            ),
        )
        validate_journal_layer_manifest(manifest)
        return manifest


def validate_journal_layer_manifest(manifest: KVJournalLayerManifest) -> None:
    if manifest.layout != JOURNAL_CACHE_LAYOUT:
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
    reference_key_dtype = manifest.entries[0].key_dtype
    reference_value_dtype = manifest.entries[0].value_dtype
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
        if entry.key_dtype != reference_key_dtype:
            raise ValueError(
                f"KV layer {manifest.layer_idx} key dtype changed across journal entries"
            )
        if entry.value_dtype != reference_value_dtype:
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
        if entry.key_nbytes != _expected_nbytes(entry.key_dtype, entry.key_shape):
            raise ValueError(
                f"KV layer {manifest.layer_idx} key journal entry size does not match shape"
            )
        if entry.value_nbytes != _expected_nbytes(entry.value_dtype, entry.value_shape):
            raise ValueError(
                f"KV layer {manifest.layer_idx} value journal entry size does not match shape"
            )
        next_expected_start = entry.end_token
        next_expected_key_offset += entry.key_nbytes
        next_expected_value_offset += entry.value_nbytes
    if manifest.persisted_tokens != next_expected_start:
        raise ValueError(
            f"KV layer {manifest.layer_idx} persisted_tokens does not match journal coverage"
        )


def _expected_nbytes(dtype_name_value: str, shape: tuple[int, ...]) -> int:
    dtype = dtype_from_name(dtype_name_value)
    element_size = torch.empty((), dtype=dtype).element_size()
    total_bytes = element_size
    for dimension in shape:
        total_bytes *= dimension
    return total_bytes

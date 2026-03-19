"""Append-only journal-backed KV store for tiered cold spill."""

import json
import time
from pathlib import Path

import torch

from ollm.async_io import (
    path_append_bytes,
    path_exists,
    path_mkdir,
    path_read_bytes_range,
)
from ollm.kv_cache_journal_manifest import (
    DEFAULT_JOURNAL_COMPACTION_ENTRY_THRESHOLD,
    JOURNAL_FILE_NAME,
    KVJournalEntryMetadata,
    KVJournalLayerManifest,
    validate_journal_layer_manifest,
)
from ollm.kv_cache_store_common import (
    CACHE_SCHEMA_VERSION,
    PERSISTED_DEVICE,
    SEQUENCE_AXIS,
    atomic_write_bytes,
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

_CACHE_FORMAT = "ollm-kv-journal"


class JournaledKVStore:
    def __init__(
        self,
        cache_folder: Path,
        *,
        compaction_entry_threshold: int = DEFAULT_JOURNAL_COMPACTION_ENTRY_THRESHOLD,
    ) -> None:
        if compaction_entry_threshold < 0:
            raise ValueError("compaction_entry_threshold must be zero or greater")
        self.cache_folder = cache_folder
        self.layers_folder = cache_folder / "layers"
        self.root_manifest_path = cache_folder / "manifest.json"
        self.compaction_entry_threshold = compaction_entry_threshold
        self._root_manifest_cache: tuple[tuple[int, ...], str] | None = None
        self._layer_manifest_cache: dict[int, KVJournalLayerManifest | None] = {}
        self._last_compaction_elapsed_seconds: float | None = None

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
        key_journal_path = self.cache_folder / layer_manifest.key_journal_path
        value_journal_path = self.cache_folder / layer_manifest.value_journal_path
        key_chunks = self._read_journal_chunks(
            key_journal_path,
            layer_manifest.entries,
            kind="key",
        )
        value_chunks = self._read_journal_chunks(
            value_journal_path,
            layer_manifest.entries,
            kind="value",
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
        self._last_compaction_elapsed_seconds = None
        layer_manifest = self._read_layer_manifest(layer_idx)
        if layer_manifest is None:
            layer_folder = self.layers_folder / str(layer_idx)
            key_journal_path = layer_folder / "key" / JOURNAL_FILE_NAME
            value_journal_path = layer_folder / "value" / JOURNAL_FILE_NAME
            layer_manifest = KVJournalLayerManifest.new(
                layer_idx,
                key_journal_path=str(key_journal_path.relative_to(self.cache_folder)),
                value_journal_path=str(
                    value_journal_path.relative_to(self.cache_folder)
                ),
            )

        key_journal_path = self.cache_folder / layer_manifest.key_journal_path
        value_journal_path = self.cache_folder / layer_manifest.value_journal_path
        path_mkdir(key_journal_path.parent, parents=True, exist_ok=True)
        path_mkdir(value_journal_path.parent, parents=True, exist_ok=True)

        start_token = layer_manifest.persisted_tokens
        end_token = start_token + token_count
        key_offset = path_append_bytes(key_journal_path, key_bytes)
        value_offset = path_append_bytes(value_journal_path, value_bytes)
        entry = KVJournalEntryMetadata(
            start_token=start_token,
            end_token=end_token,
            key_dtype=dtype_name(key_tensor.dtype),
            value_dtype=dtype_name(value_tensor.dtype),
            key_shape=tuple(key_tensor.shape),
            value_shape=tuple(value_tensor.shape),
            key_offset=key_offset,
            value_offset=value_offset,
            key_nbytes=len(key_bytes),
            value_nbytes=len(value_bytes),
        )
        updated_manifest = KVJournalLayerManifest(
            layer_idx=layer_idx,
            layout=layer_manifest.layout,
            sequence_axis=SEQUENCE_AXIS,
            persisted_tokens=end_token,
            compaction_count=layer_manifest.compaction_count,
            key_journal_path=layer_manifest.key_journal_path,
            value_journal_path=layer_manifest.value_journal_path,
            entries=layer_manifest.entries + (entry,),
        )
        validate_journal_layer_manifest(updated_manifest)
        self._write_layer_manifest(updated_manifest)
        if self._should_compact(updated_manifest):
            compact_started_at = time.perf_counter()
            updated_manifest = self._compact_layer(updated_manifest)
            self._last_compaction_elapsed_seconds = (
                time.perf_counter() - compact_started_at
            )

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
        total_entries = 0
        for layer_idx in self.persisted_layer_ids():
            manifest = self._read_layer_manifest(layer_idx)
            if manifest is None:
                continue
            total_entries += len(manifest.entries)
        return total_entries

    def cold_store_format_id(self) -> str | None:
        return None

    def compaction_count(self) -> int:
        total_compactions = 0
        for layer_idx in self.persisted_layer_ids():
            manifest = self._read_layer_manifest(layer_idx)
            if manifest is None:
                continue
            total_compactions += manifest.compaction_count
        return total_compactions

    def consume_last_compaction_elapsed_seconds(self) -> float | None:
        elapsed_seconds = self._last_compaction_elapsed_seconds
        self._last_compaction_elapsed_seconds = None
        return elapsed_seconds

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
        compaction_entry_threshold = require_int(payload, "compaction_entry_threshold")
        if compaction_entry_threshold != self.compaction_entry_threshold:
            raise ValueError(
                "Unsupported KV journal compaction entry threshold: "
                f"{compaction_entry_threshold}"
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
                    "compaction_entry_threshold": self.compaction_entry_threshold,
                    "layers": list(layers),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        self._root_manifest_cache = (layers, policy_id)

    def _read_layer_manifest(self, layer_idx: int) -> KVJournalLayerManifest | None:
        if layer_idx in self._layer_manifest_cache:
            return self._layer_manifest_cache[layer_idx]
        layer_folder = self.layers_folder / str(layer_idx)
        if not path_exists(layer_folder):
            self._layer_manifest_cache[layer_idx] = None
            return None
        manifest_path = layer_folder / "manifest.json"
        if not path_exists(manifest_path):
            raise ValueError(f"KV layer manifest is missing: {manifest_path}")
        manifest = KVJournalLayerManifest.from_dict(read_json_object(manifest_path))
        self._layer_manifest_cache[layer_idx] = manifest
        return manifest

    def _write_layer_manifest(self, manifest: KVJournalLayerManifest) -> None:
        layer_folder = self.layers_folder / str(manifest.layer_idx)
        path_mkdir(layer_folder, parents=True, exist_ok=True)
        manifest_path = layer_folder / "manifest.json"
        atomic_write_text(
            manifest_path,
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        self._layer_manifest_cache[manifest.layer_idx] = manifest

    def _read_journal_chunks(
        self,
        journal_path: Path,
        entries: tuple[KVJournalEntryMetadata, ...],
        *,
        kind: str,
    ) -> list[torch.Tensor]:
        self._validate_journal_path(journal_path)
        total_bytes = sum(self._entry_length(entry, kind=kind) for entry in entries)
        raw_buffer = path_read_bytes_range(journal_path, offset=0, length=total_bytes)
        return [
            decode_tensor_bytes(
                raw_buffer[
                    self._entry_offset(entry, kind=kind) : self._entry_offset(
                        entry, kind=kind
                    )
                    + self._entry_length(entry, kind=kind)
                ],
                dtype=dtype_from_name(self._entry_dtype(entry, kind=kind)),
                shape=self._entry_shape(entry, kind=kind),
            )
            for entry in entries
        ]

    def _entry_dtype(self, entry: KVJournalEntryMetadata, *, kind: str) -> str:
        if kind == "key":
            return entry.key_dtype
        if kind == "value":
            return entry.value_dtype
        raise ValueError(f"Unsupported journal entry kind: {kind}")

    def _entry_shape(
        self, entry: KVJournalEntryMetadata, *, kind: str
    ) -> tuple[int, ...]:
        if kind == "key":
            return entry.key_shape
        if kind == "value":
            return entry.value_shape
        raise ValueError(f"Unsupported journal entry kind: {kind}")

    def _entry_offset(self, entry: KVJournalEntryMetadata, *, kind: str) -> int:
        if kind == "key":
            return entry.key_offset
        if kind == "value":
            return entry.value_offset
        raise ValueError(f"Unsupported journal entry kind: {kind}")

    def _entry_length(self, entry: KVJournalEntryMetadata, *, kind: str) -> int:
        if kind == "key":
            return entry.key_nbytes
        if kind == "value":
            return entry.value_nbytes
        raise ValueError(f"Unsupported journal entry kind: {kind}")

    def _validate_journal_path(self, path: Path) -> None:
        resolved_path = path.resolve()
        cache_root = self.cache_folder.resolve()
        if not resolved_path.is_relative_to(cache_root):
            raise ValueError(f"KV cache chunk path escapes cache root: {path}")
        if not path_exists(path):
            raise ValueError(f"KV cache chunk file is missing: {path}")

    def _should_compact(self, manifest: KVJournalLayerManifest) -> bool:
        if self.compaction_entry_threshold <= 0:
            return False
        return len(manifest.entries) >= self.compaction_entry_threshold

    def _compact_layer(
        self, manifest: KVJournalLayerManifest
    ) -> KVJournalLayerManifest:
        key_tensor, value_tensor = self._load_layer_cpu_from_manifest(manifest)
        key_journal_path = self.cache_folder / manifest.key_journal_path
        value_journal_path = self.cache_folder / manifest.value_journal_path
        key_bytes = encode_tensor_bytes(key_tensor)
        value_bytes = encode_tensor_bytes(value_tensor)
        atomic_write_bytes(key_journal_path, key_bytes)
        atomic_write_bytes(value_journal_path, value_bytes)
        compacted_manifest = KVJournalLayerManifest(
            layer_idx=manifest.layer_idx,
            layout=manifest.layout,
            sequence_axis=manifest.sequence_axis,
            persisted_tokens=manifest.persisted_tokens,
            compaction_count=manifest.compaction_count + 1,
            key_journal_path=manifest.key_journal_path,
            value_journal_path=manifest.value_journal_path,
            entries=(
                KVJournalEntryMetadata(
                    start_token=0,
                    end_token=manifest.persisted_tokens,
                    key_dtype=dtype_name(key_tensor.dtype),
                    value_dtype=dtype_name(value_tensor.dtype),
                    key_shape=tuple(key_tensor.shape),
                    value_shape=tuple(value_tensor.shape),
                    key_offset=0,
                    value_offset=0,
                    key_nbytes=len(key_bytes),
                    value_nbytes=len(value_bytes),
                ),
            ),
        )
        validate_journal_layer_manifest(compacted_manifest)
        self._write_layer_manifest(compacted_manifest)
        return compacted_manifest

    def _load_layer_cpu_from_manifest(
        self, manifest: KVJournalLayerManifest
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_journal_path = self.cache_folder / manifest.key_journal_path
        value_journal_path = self.cache_folder / manifest.value_journal_path
        key_chunks = self._read_journal_chunks(
            key_journal_path,
            manifest.entries,
            kind="key",
        )
        value_chunks = self._read_journal_chunks(
            value_journal_path,
            manifest.entries,
            kind="value",
        )
        return (
            torch.cat(key_chunks, dim=SEQUENCE_AXIS),
            torch.cat(value_chunks, dim=SEQUENCE_AXIS),
        )

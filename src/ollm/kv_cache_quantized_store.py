"""Quantized journal-backed KV store for cold-tier persistence."""

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
from ollm.kv_cache_quantized_manifest import (
    DEFAULT_QUANTIZED_JOURNAL_COMPACTION_ENTRY_THRESHOLD,
    QUANTIZED_COLD_TIER_REPRESENTATION,
    QUANTIZED_JOURNAL_FILE_NAME,
    QUANTIZED_TENSOR_DTYPE,
    QUANTIZED_TENSOR_DTYPE_NAME,
    KVQuantizedJournalEntryMetadata,
    KVQuantizedJournalLayerManifest,
    validate_quantized_journal_layer_manifest,
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

_CACHE_FORMAT = "ollm-kv-journal-quantized"


class QuantizedJournaledKVStore:
    def __init__(
        self,
        cache_folder: Path,
        *,
        compaction_entry_threshold: int = DEFAULT_QUANTIZED_JOURNAL_COMPACTION_ENTRY_THRESHOLD,
    ) -> None:
        if compaction_entry_threshold < 0:
            raise ValueError("compaction_entry_threshold must be zero or greater")
        self.cache_folder = cache_folder
        self.layers_folder = cache_folder / "layers"
        self.root_manifest_path = cache_folder / "manifest.json"
        self.compaction_entry_threshold = compaction_entry_threshold
        self._root_manifest_cache: tuple[tuple[int, ...], str] | None = None
        self._layer_manifest_cache: dict[
            int, KVQuantizedJournalLayerManifest | None
        ] = {}
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
        key_chunks = self._read_quantized_journal_chunks(
            key_journal_path,
            layer_manifest.entries,
            kind="key",
        )
        value_chunks = self._read_quantized_journal_chunks(
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

        quantized_key, key_scale = _quantize_tensor(key_tensor)
        quantized_value, value_scale = _quantize_tensor(value_tensor)
        key_bytes = encode_tensor_bytes(quantized_key)
        value_bytes = encode_tensor_bytes(quantized_value)
        self._last_compaction_elapsed_seconds = None
        layer_manifest = self._read_layer_manifest(layer_idx)
        if layer_manifest is None:
            layer_folder = self.layers_folder / str(layer_idx)
            key_journal_path = layer_folder / "key" / QUANTIZED_JOURNAL_FILE_NAME
            value_journal_path = layer_folder / "value" / QUANTIZED_JOURNAL_FILE_NAME
            layer_manifest = KVQuantizedJournalLayerManifest.new(
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
        entry = KVQuantizedJournalEntryMetadata(
            start_token=start_token,
            end_token=end_token,
            key_original_dtype=dtype_name(key_tensor.dtype),
            value_original_dtype=dtype_name(value_tensor.dtype),
            key_shape=tuple(key_tensor.shape),
            value_shape=tuple(value_tensor.shape),
            key_offset=key_offset,
            value_offset=value_offset,
            key_nbytes=len(key_bytes),
            value_nbytes=len(value_bytes),
            key_scale=key_scale,
            value_scale=value_scale,
        )
        updated_manifest = KVQuantizedJournalLayerManifest(
            layer_idx=layer_idx,
            layout=layer_manifest.layout,
            sequence_axis=SEQUENCE_AXIS,
            persisted_tokens=end_token,
            compaction_count=layer_manifest.compaction_count,
            key_journal_path=layer_manifest.key_journal_path,
            value_journal_path=layer_manifest.value_journal_path,
            entries=layer_manifest.entries + (entry,),
        )
        validate_quantized_journal_layer_manifest(updated_manifest)
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
        return _CACHE_FORMAT

    def cold_tier_representation_id(self) -> str | None:
        return QUANTIZED_COLD_TIER_REPRESENTATION

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
        if not key_tensor.is_floating_point() or not value_tensor.is_floating_point():
            raise ValueError(
                f"Quantized cold-tier KV only supports floating-point tensors for layer {layer_idx}"
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
        if require_str(payload, "quantized_dtype") != QUANTIZED_TENSOR_DTYPE_NAME:
            raise ValueError(
                f"Unsupported quantized KV tensor dtype: {payload['quantized_dtype']!r}"
            )
        if (
            require_str(payload, "cold_tier_representation")
            != QUANTIZED_COLD_TIER_REPRESENTATION
        ):
            raise ValueError(
                "Unsupported quantized cold-tier representation: "
                f"{payload['cold_tier_representation']!r}"
            )
        policy_id = require_str(payload, "policy_id")
        compaction_entry_threshold = require_int(payload, "compaction_entry_threshold")
        if compaction_entry_threshold != self.compaction_entry_threshold:
            raise ValueError(
                "Unsupported quantized KV journal compaction entry threshold: "
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
                    "quantized_dtype": QUANTIZED_TENSOR_DTYPE_NAME,
                    "cold_tier_representation": QUANTIZED_COLD_TIER_REPRESENTATION,
                    "compaction_entry_threshold": self.compaction_entry_threshold,
                    "layers": list(layers),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        self._root_manifest_cache = (layers, policy_id)

    def _read_layer_manifest(
        self, layer_idx: int
    ) -> KVQuantizedJournalLayerManifest | None:
        if layer_idx in self._layer_manifest_cache:
            return self._layer_manifest_cache[layer_idx]
        manifest_path = self.layers_folder / str(layer_idx) / "manifest.json"
        if not path_exists(manifest_path):
            self._layer_manifest_cache[layer_idx] = None
            return None
        manifest = KVQuantizedJournalLayerManifest.from_dict(
            read_json_object(manifest_path)
        )
        self._layer_manifest_cache[layer_idx] = manifest
        return manifest

    def _write_layer_manifest(self, manifest: KVQuantizedJournalLayerManifest) -> None:
        manifest_path = self.layers_folder / str(manifest.layer_idx) / "manifest.json"
        path_mkdir(manifest_path.parent, parents=True, exist_ok=True)
        atomic_write_text(
            manifest_path,
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        self._layer_manifest_cache[manifest.layer_idx] = manifest

    def _should_compact(self, manifest: KVQuantizedJournalLayerManifest) -> bool:
        threshold = self.compaction_entry_threshold
        return threshold > 0 and len(manifest.entries) >= threshold

    def _read_quantized_journal_chunks(
        self,
        journal_path: Path,
        entries: tuple[KVQuantizedJournalEntryMetadata, ...],
        *,
        kind: str,
    ) -> list[torch.Tensor]:
        self._validate_journal_path(journal_path)
        total_bytes = sum(
            entry.key_nbytes if kind == "key" else entry.value_nbytes
            for entry in entries
        )
        raw_buffer = path_read_bytes_range(journal_path, offset=0, length=total_bytes)
        chunks: list[torch.Tensor] = []
        for entry in entries:
            if kind == "key":
                raw_bytes = raw_buffer[
                    entry.key_offset : entry.key_offset + entry.key_nbytes
                ]
                chunks.append(
                    _decode_quantized_tensor(
                        raw_bytes=raw_bytes,
                        scale=entry.key_scale,
                        original_dtype=dtype_from_name(entry.key_original_dtype),
                        shape=entry.key_shape,
                    )
                )
                continue
            if kind == "value":
                raw_bytes = raw_buffer[
                    entry.value_offset : entry.value_offset + entry.value_nbytes
                ]
                chunks.append(
                    _decode_quantized_tensor(
                        raw_bytes=raw_bytes,
                        scale=entry.value_scale,
                        original_dtype=dtype_from_name(entry.value_original_dtype),
                        shape=entry.value_shape,
                    )
                )
                continue
            raise ValueError(f"Unsupported journal entry kind: {kind}")
        return chunks

    def _validate_journal_path(self, path: Path) -> None:
        resolved_root = self.cache_folder.resolve()
        resolved_path = path.resolve()
        if (
            resolved_root not in resolved_path.parents
            and resolved_path != resolved_root
        ):
            raise ValueError("journal path must stay within the KV cache root")

    def _compact_layer(
        self, manifest: KVQuantizedJournalLayerManifest
    ) -> KVQuantizedJournalLayerManifest:
        key_journal_path = self.cache_folder / manifest.key_journal_path
        value_journal_path = self.cache_folder / manifest.value_journal_path
        key_chunks = self._read_quantized_journal_chunks(
            key_journal_path, manifest.entries, kind="key"
        )
        value_chunks = self._read_quantized_journal_chunks(
            value_journal_path, manifest.entries, kind="value"
        )
        key_tensor = torch.cat(key_chunks, dim=SEQUENCE_AXIS).cpu().contiguous()
        value_tensor = torch.cat(value_chunks, dim=SEQUENCE_AXIS).cpu().contiguous()
        quantized_key, key_scale = _quantize_tensor(key_tensor)
        quantized_value, value_scale = _quantize_tensor(value_tensor)
        key_bytes = encode_tensor_bytes(quantized_key)
        value_bytes = encode_tensor_bytes(quantized_value)
        atomic_write_bytes(key_journal_path, key_bytes)
        atomic_write_bytes(value_journal_path, value_bytes)
        compacted_manifest = KVQuantizedJournalLayerManifest(
            layer_idx=manifest.layer_idx,
            layout=manifest.layout,
            sequence_axis=manifest.sequence_axis,
            persisted_tokens=manifest.persisted_tokens,
            compaction_count=manifest.compaction_count + 1,
            key_journal_path=manifest.key_journal_path,
            value_journal_path=manifest.value_journal_path,
            entries=(
                KVQuantizedJournalEntryMetadata(
                    start_token=0,
                    end_token=manifest.persisted_tokens,
                    key_original_dtype=dtype_name(key_tensor.dtype),
                    value_original_dtype=dtype_name(value_tensor.dtype),
                    key_shape=tuple(key_tensor.shape),
                    value_shape=tuple(value_tensor.shape),
                    key_offset=0,
                    value_offset=0,
                    key_nbytes=len(key_bytes),
                    value_nbytes=len(value_bytes),
                    key_scale=key_scale,
                    value_scale=value_scale,
                ),
            ),
        )
        validate_quantized_journal_layer_manifest(compacted_manifest)
        self._write_layer_manifest(compacted_manifest)
        return compacted_manifest


def _quantize_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    float_tensor = tensor.detach().cpu().to(torch.float32).contiguous()
    if not bool(torch.isfinite(float_tensor).all().item()):
        raise ValueError("Quantized KV cache tensors must be finite")
    max_abs = float(float_tensor.abs().amax().item())
    scale = 1.0 if max_abs == 0.0 else max_abs / 127.0
    quantized = torch.clamp(torch.round(float_tensor / scale), min=-127, max=127).to(
        QUANTIZED_TENSOR_DTYPE
    )
    return quantized.contiguous(), scale


def _decode_quantized_tensor(
    *,
    raw_bytes: bytes,
    scale: float,
    original_dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    quantized = decode_tensor_bytes(
        raw_bytes,
        dtype=QUANTIZED_TENSOR_DTYPE,
        shape=shape,
    )
    return (quantized.to(torch.float32) * scale).to(original_dtype)

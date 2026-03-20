"""Tiered write-back disk KV store with explicit cold-tier metadata."""

import json
from pathlib import Path

import torch

from ollm.async_io import path_exists, path_mkdir
from ollm.kv_cache_journal_store import JournaledKVStore
from ollm.kv_cache_store_common import (
    CACHE_SCHEMA_VERSION,
    PERSISTED_DEVICE,
    SEQUENCE_AXIS,
    atomic_write_text,
    read_json_object,
    require_int,
    require_int_value,
    require_str,
)

_CACHE_FORMAT = "ollm-kv-tiered-write-back"
_COLD_STORE_ROOT = "cold"
_COLD_STORE_FORMAT = "ollm-kv-journal"


class TieredWriteBackKVStore:
    """Persist only the cold tier while hot KV remains resident in memory."""

    def __init__(self, cache_folder: Path) -> None:
        self.cache_folder = cache_folder
        self.root_manifest_path = cache_folder / "manifest.json"
        self.cold_root = cache_folder / _COLD_STORE_ROOT
        self._cold_store = JournaledKVStore(
            self.cold_root, compaction_entry_threshold=0
        )
        self._root_manifest_cache: tuple[tuple[int, ...], str] | None = None

    def initialize(self, policy_id: str) -> None:
        path_mkdir(self.cache_folder, parents=True, exist_ok=True)
        self._cold_store.initialize(policy_id)
        self._write_root_manifest((), policy_id)

    def load_layer(
        self, layer_idx: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        _ = self._read_root_manifest()
        return self._cold_store.load_layer(layer_idx, device=device)

    def append_layer_chunk(
        self, layer_idx: int, tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        self._cold_store.append_layer_chunk(layer_idx, tensors)
        layers, policy_id = self._read_root_manifest()
        if layer_idx not in layers:
            self._write_root_manifest(tuple(sorted(layers + (layer_idx,))), policy_id)

    def persisted_layer_ids(self) -> tuple[int, ...]:
        if not path_exists(self.root_manifest_path):
            return ()
        return self._read_root_manifest()[0]

    def persisted_token_count(self) -> int:
        return self._cold_store.persisted_token_count()

    def persisted_artifact_count(self) -> int:
        return self._cold_store.persisted_artifact_count()

    def cold_store_format_id(self) -> str | None:
        return _COLD_STORE_FORMAT

    def cold_tier_representation_id(self) -> str | None:
        return None

    def compaction_count(self) -> int:
        return self._cold_store.compaction_count()

    def eviction_count(self) -> int:
        return self._cold_store.eviction_count()

    def evicted_token_count(self) -> int:
        return self._cold_store.evicted_token_count()

    def consume_last_compaction_elapsed_seconds(self) -> float | None:
        return self._cold_store.consume_last_compaction_elapsed_seconds()

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
        if require_str(payload, "cold_store_root") != _COLD_STORE_ROOT:
            raise ValueError(
                f"Unsupported tiered cold-store root: {payload['cold_store_root']!r}"
            )
        if require_str(payload, "cold_store_format") != _COLD_STORE_FORMAT:
            raise ValueError(
                f"Unsupported tiered cold-store format: {payload['cold_store_format']!r}"
            )
        layers_payload = payload.get("layers")
        if not isinstance(layers_payload, list):
            raise ValueError("KV root manifest layers must be a JSON list")
        manifest = (
            tuple(require_int_value(value, "layers[]") for value in layers_payload),
            require_str(payload, "policy_id"),
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
                    "cold_store_root": _COLD_STORE_ROOT,
                    "cold_store_format": _COLD_STORE_FORMAT,
                    "layers": list(layers),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        self._root_manifest_cache = (layers, policy_id)

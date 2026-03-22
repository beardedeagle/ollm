"""Manifest types and validation for the streamed KV cache store."""

from dataclasses import dataclass

from ollm.kv_cache.store_common import (
    SEQUENCE_AXIS,
    require_int,
    require_object_list,
    require_relative_path,
    require_shape,
    require_str,
    sequence_length,
    shape_prefix,
)

STREAMED_CACHE_LAYOUT = "streamed-segmented"
DEFAULT_SEGMENT_BYTES_TARGET = 8 * 1024 * 1024


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
            layout=STREAMED_CACHE_LAYOUT,
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
        validate_stream_layer_manifest(manifest)
        return manifest


def validate_stream_layer_manifest(manifest: KVStreamLayerManifest) -> None:
    if manifest.layout != STREAMED_CACHE_LAYOUT:
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

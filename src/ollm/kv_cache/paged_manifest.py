"""Manifest models and validation for the paged KV cache store."""

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

PAGED_CACHE_LAYOUT = "paged"
PAGED_BLOB_FILE_NAME = "pages.bin"


@dataclass(slots=True, frozen=True)
class KVPageMetadata:
    page_index: int
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
            "page_index": self.page_index,
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
        page_index = require_int(payload, "page_index")
        start_token = require_int(payload, "start_token")
        end_token = require_int(payload, "end_token")
        if page_index < 0:
            raise ValueError(f"Invalid KV page index: {page_index}")
        if end_token <= start_token:
            raise ValueError(
                f"Invalid KV page token range: start={start_token} end={end_token}"
            )
        return cls(
            page_index=page_index,
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
class KVPagedLayerManifest:
    layer_idx: int
    layout: str
    sequence_axis: int
    persisted_tokens: int
    page_token_capacity: int
    key_pages_path: str
    value_pages_path: str
    pages: tuple[KVPageMetadata, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_idx": self.layer_idx,
            "layout": self.layout,
            "sequence_axis": self.sequence_axis,
            "persisted_tokens": self.persisted_tokens,
            "page_token_capacity": self.page_token_capacity,
            "key_pages_path": self.key_pages_path,
            "value_pages_path": self.value_pages_path,
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def new(
        cls,
        layer_idx: int,
        *,
        page_token_capacity: int,
        key_pages_path: str,
        value_pages_path: str,
    ):
        return cls(
            layer_idx=layer_idx,
            layout=PAGED_CACHE_LAYOUT,
            sequence_axis=SEQUENCE_AXIS,
            persisted_tokens=0,
            page_token_capacity=page_token_capacity,
            key_pages_path=key_pages_path,
            value_pages_path=value_pages_path,
            pages=(),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        manifest = cls(
            layer_idx=require_int(payload, "layer_idx"),
            layout=require_str(payload, "layout"),
            sequence_axis=require_int(payload, "sequence_axis"),
            persisted_tokens=require_int(payload, "persisted_tokens"),
            page_token_capacity=require_int(payload, "page_token_capacity"),
            key_pages_path=require_relative_path(payload, "key_pages_path"),
            value_pages_path=require_relative_path(payload, "value_pages_path"),
            pages=tuple(
                KVPageMetadata.from_dict(page_payload)
                for page_payload in require_object_list(payload.get("pages"), "pages")
            ),
        )
        validate_paged_layer_manifest(manifest)
        return manifest


def validate_paged_layer_manifest(manifest: KVPagedLayerManifest) -> None:
    if manifest.layout != PAGED_CACHE_LAYOUT:
        raise ValueError(f"Unsupported KV paged cache layout: {manifest.layout!r}")
    if manifest.sequence_axis != SEQUENCE_AXIS:
        raise ValueError(
            f"Unsupported KV paged cache sequence axis: {manifest.sequence_axis}"
        )
    if manifest.page_token_capacity <= 0:
        raise ValueError("KV paged cache page_token_capacity must be positive")
    if not manifest.pages:
        if manifest.persisted_tokens != 0:
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} has persisted tokens without pages"
            )
        return
    reference_key_dtype = manifest.pages[0].key_dtype
    reference_value_dtype = manifest.pages[0].value_dtype
    reference_key_prefix = shape_prefix(manifest.pages[0].key_shape)
    reference_value_prefix = shape_prefix(manifest.pages[0].value_shape)
    next_expected_start = 0
    for page_index, page in enumerate(manifest.pages):
        if page.page_index != page_index:
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} has non-sequential page indices"
            )
        expected_start = page_index * manifest.page_token_capacity
        if page.start_token != expected_start:
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} page {page_index} has an unexpected start token"
            )
        if page.start_token != next_expected_start:
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} has non-contiguous page ranges"
            )
        if page.key_offset < 0 or page.value_offset < 0:
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} page {page_index} uses a negative offset"
            )
        if page.key_nbytes <= 0 or page.value_nbytes <= 0:
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} page {page_index} uses a non-positive payload size"
            )
        if sequence_length(page.key_shape) != page.token_count:
            raise ValueError(
                f"KV paged key page shape does not match token range for layer {manifest.layer_idx}"
            )
        if sequence_length(page.value_shape) != page.token_count:
            raise ValueError(
                f"KV paged value page shape does not match token range for layer {manifest.layer_idx}"
            )
        if page.key_dtype != reference_key_dtype:
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} key dtype changed across pages"
            )
        if page.value_dtype != reference_value_dtype:
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} value dtype changed across pages"
            )
        if shape_prefix(page.key_shape) != reference_key_prefix:
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} key shape prefix changed across pages"
            )
        if shape_prefix(page.value_shape) != reference_value_prefix:
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} value shape prefix changed across pages"
            )
        if page.token_count > manifest.page_token_capacity:
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} page {page_index} exceeds page_token_capacity"
            )
        if page_index < len(manifest.pages) - 1 and (
            page.token_count != manifest.page_token_capacity
        ):
            raise ValueError(
                f"KV paged layer {manifest.layer_idx} has a non-final partial page"
            )
        next_expected_start = page.end_token
    if manifest.persisted_tokens != next_expected_start:
        raise ValueError(
            f"KV paged layer {manifest.layer_idx} persisted_tokens does not match page coverage"
        )

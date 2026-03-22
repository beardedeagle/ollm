"""Orthogonal KV strategy, lifecycle, and adaptation helpers."""

import hashlib
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from ollm.kv_cache.strategy import (
    DEFAULT_KV_CACHE_STRATEGY,
    normalize_kv_cache_strategy,
)


class KVCachePersistenceFormat(StrEnum):
    """Describe how cold KV is persisted."""

    RESIDENT_ONLY = "resident-only"
    CHUNKED_MANIFEST = "chunked-manifest"
    PAGED_MANIFEST = "paged-manifest"
    STREAMED_SEGMENTED = "streamed-segmented"
    LOG_STRUCTURED_JOURNAL = "log-structured-journal"
    SLIDING_WINDOW_RING_BUFFER = "sliding-window-ring-buffer"


class KVCacheResidencyMode(StrEnum):
    """Describe how active KV is buffered in-process."""

    BUFFERED_TAIL = "buffered-tail"
    TIERED_WRITE_BACK = "tiered-write-back"
    FULLY_RESIDENT = "fully-resident"


class KVCacheWindowPolicy(StrEnum):
    """Describe how much history a strategy intends to preserve."""

    FULL_HISTORY = "full-history"
    SLIDING_WINDOW = "sliding-window"


class KVCacheColdTierEncoding(StrEnum):
    """Describe how colder persisted KV is encoded."""

    FULL_PRECISION = "full-precision"
    QUANTIZED = "quantized"


class KVCacheLifecycle(StrEnum):
    """Describe cache ownership across process boundaries."""

    RUNTIME_SCOPED = "runtime-scoped"
    PERSISTENT = "persistent"


class KVCacheAdaptationMode(StrEnum):
    """Describe how runtime telemetry can influence cache planning."""

    DISABLED = "disabled"
    OBSERVE_ONLY = "observe-only"
    AUTOMATIC = "automatic"


DEFAULT_KV_CACHE_LIFECYCLE = KVCacheLifecycle.RUNTIME_SCOPED.value
DEFAULT_KV_CACHE_ADAPTATION_MODE = KVCacheAdaptationMode.OBSERVE_ONLY.value
DEFAULT_KV_CACHE_WINDOW_TOKENS = 256
DEFAULT_KV_CACHE_EVICTION_POLICY = "drop-oldest"
_SAFE_PATH_FRAGMENT = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(frozen=True, slots=True)
class KVCacheStrategyAxes:
    """Describe the orthogonal axes behind one strategy preset."""

    strategy_id: str
    persistence_format: str
    residency_mode: str
    window_policy: str
    cold_tier_encoding: str
    compaction_capable: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary view."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class KVCacheAdaptationSurface:
    """Describe the current adaptation surface truthfully."""

    adaptation_mode: str
    recommendation_available: bool
    recommended_strategy_id: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary view."""

        return asdict(self)


def normalize_kv_cache_lifecycle(lifecycle: str | None) -> str | None:
    """Validate and normalize a cache lifecycle identifier."""

    if lifecycle is None:
        return None
    normalized = lifecycle.strip().lower()
    if not normalized:
        raise ValueError("kv_cache_lifecycle cannot be empty")
    try:
        return KVCacheLifecycle(normalized).value
    except ValueError as exc:
        allowed = ", ".join(item.value for item in KVCacheLifecycle)
        raise ValueError(f"kv_cache_lifecycle must be one of: {allowed}") from exc


def resolve_kv_cache_lifecycle(
    strategy: str | None,
    lifecycle: str | None,
) -> str:
    """Resolve lifecycle semantics for one strategy preset."""

    normalized_lifecycle = normalize_kv_cache_lifecycle(lifecycle)
    resolved_lifecycle = (
        DEFAULT_KV_CACHE_LIFECYCLE
        if normalized_lifecycle is None
        else normalized_lifecycle
    )
    normalized_strategy = normalize_kv_cache_strategy(strategy)
    strategy_id = (
        DEFAULT_KV_CACHE_STRATEGY
        if normalized_strategy is None
        else normalized_strategy
    )
    if (
        strategy_id == "resident"
        and resolved_lifecycle != KVCacheLifecycle.RUNTIME_SCOPED.value
    ):
        raise ValueError(
            "--kv-cache-strategy resident requires --kv-cache-lifecycle runtime-scoped"
        )
    return resolved_lifecycle


def normalize_kv_cache_adaptation_mode(mode: str | None) -> str | None:
    """Validate and normalize an adaptation-mode identifier."""

    if mode is None:
        return None
    normalized = mode.strip().lower()
    if not normalized:
        raise ValueError("kv_cache_adaptation_mode cannot be empty")
    try:
        return KVCacheAdaptationMode(normalized).value
    except ValueError as exc:
        allowed = ", ".join(item.value for item in KVCacheAdaptationMode)
        raise ValueError(f"kv_cache_adaptation_mode must be one of: {allowed}") from exc


def normalize_kv_cache_window_tokens(window_tokens: int | None) -> int | None:
    """Validate and normalize a sliding-window token budget."""

    if window_tokens is None:
        return None
    if window_tokens <= 0:
        raise ValueError("kv_cache_window_tokens must be greater than zero")
    return int(window_tokens)


def resolve_kv_cache_window_tokens(
    strategy: str | None,
    window_tokens: int | None,
) -> int | None:
    """Resolve the active sliding-window token budget for one strategy."""

    normalized_strategy = normalize_kv_cache_strategy(strategy)
    strategy_id = (
        DEFAULT_KV_CACHE_STRATEGY
        if normalized_strategy is None
        else normalized_strategy
    )
    normalized_window_tokens = normalize_kv_cache_window_tokens(window_tokens)
    if strategy_id == "sliding-window-ring-buffer":
        if normalized_window_tokens is None:
            return DEFAULT_KV_CACHE_WINDOW_TOKENS
        return normalized_window_tokens
    if normalized_window_tokens is not None:
        raise ValueError(
            "--kv-cache-window-tokens requires --kv-cache-strategy "
            "sliding-window-ring-buffer"
        )
    return None


def resolve_kv_cache_eviction_policy(strategy: str | None) -> str | None:
    """Resolve the eviction policy identifier for one strategy."""

    normalized_strategy = normalize_kv_cache_strategy(strategy)
    strategy_id = (
        DEFAULT_KV_CACHE_STRATEGY
        if normalized_strategy is None
        else normalized_strategy
    )
    if strategy_id == "sliding-window-ring-buffer":
        return DEFAULT_KV_CACHE_EVICTION_POLICY
    return None


def describe_kv_cache_strategy(strategy: str | None) -> KVCacheStrategyAxes:
    """Return the orthogonal axes behind a strategy preset."""

    normalized = normalize_kv_cache_strategy(strategy)
    strategy_id = DEFAULT_KV_CACHE_STRATEGY if normalized is None else normalized
    if strategy_id == "resident":
        return KVCacheStrategyAxes(
            strategy_id=strategy_id,
            persistence_format=KVCachePersistenceFormat.RESIDENT_ONLY.value,
            residency_mode=KVCacheResidencyMode.FULLY_RESIDENT.value,
            window_policy=KVCacheWindowPolicy.FULL_HISTORY.value,
            cold_tier_encoding=KVCacheColdTierEncoding.FULL_PRECISION.value,
            compaction_capable=False,
        )
    if strategy_id == "chunked":
        return KVCacheStrategyAxes(
            strategy_id=strategy_id,
            persistence_format=KVCachePersistenceFormat.CHUNKED_MANIFEST.value,
            residency_mode=KVCacheResidencyMode.BUFFERED_TAIL.value,
            window_policy=KVCacheWindowPolicy.FULL_HISTORY.value,
            cold_tier_encoding=KVCacheColdTierEncoding.FULL_PRECISION.value,
            compaction_capable=False,
        )
    if strategy_id == "paged":
        return KVCacheStrategyAxes(
            strategy_id=strategy_id,
            persistence_format=KVCachePersistenceFormat.PAGED_MANIFEST.value,
            residency_mode=KVCacheResidencyMode.BUFFERED_TAIL.value,
            window_policy=KVCacheWindowPolicy.FULL_HISTORY.value,
            cold_tier_encoding=KVCacheColdTierEncoding.FULL_PRECISION.value,
            compaction_capable=False,
        )
    if strategy_id == "streamed-segmented":
        return KVCacheStrategyAxes(
            strategy_id=strategy_id,
            persistence_format=KVCachePersistenceFormat.STREAMED_SEGMENTED.value,
            residency_mode=KVCacheResidencyMode.BUFFERED_TAIL.value,
            window_policy=KVCacheWindowPolicy.FULL_HISTORY.value,
            cold_tier_encoding=KVCacheColdTierEncoding.FULL_PRECISION.value,
            compaction_capable=False,
        )
    if strategy_id == "log-structured-journal":
        return KVCacheStrategyAxes(
            strategy_id=strategy_id,
            persistence_format=KVCachePersistenceFormat.LOG_STRUCTURED_JOURNAL.value,
            residency_mode=KVCacheResidencyMode.BUFFERED_TAIL.value,
            window_policy=KVCacheWindowPolicy.FULL_HISTORY.value,
            cold_tier_encoding=KVCacheColdTierEncoding.FULL_PRECISION.value,
            compaction_capable=True,
        )
    if strategy_id == "sliding-window-ring-buffer":
        return KVCacheStrategyAxes(
            strategy_id=strategy_id,
            persistence_format=KVCachePersistenceFormat.SLIDING_WINDOW_RING_BUFFER.value,
            residency_mode=KVCacheResidencyMode.BUFFERED_TAIL.value,
            window_policy=KVCacheWindowPolicy.SLIDING_WINDOW.value,
            cold_tier_encoding=KVCacheColdTierEncoding.FULL_PRECISION.value,
            compaction_capable=False,
        )
    if strategy_id == "quantized-cold-tier":
        return KVCacheStrategyAxes(
            strategy_id=strategy_id,
            persistence_format=KVCachePersistenceFormat.LOG_STRUCTURED_JOURNAL.value,
            residency_mode=KVCacheResidencyMode.BUFFERED_TAIL.value,
            window_policy=KVCacheWindowPolicy.FULL_HISTORY.value,
            cold_tier_encoding=KVCacheColdTierEncoding.QUANTIZED.value,
            compaction_capable=True,
        )
    if strategy_id == "tiered-write-back":
        return KVCacheStrategyAxes(
            strategy_id=strategy_id,
            persistence_format=KVCachePersistenceFormat.LOG_STRUCTURED_JOURNAL.value,
            residency_mode=KVCacheResidencyMode.TIERED_WRITE_BACK.value,
            window_policy=KVCacheWindowPolicy.FULL_HISTORY.value,
            cold_tier_encoding=KVCacheColdTierEncoding.FULL_PRECISION.value,
            compaction_capable=True,
        )
    raise ValueError(f"Unsupported kv cache strategy: {strategy_id}")


def build_kv_cache_adaptation_surface(
    *,
    adaptation_mode: str | None,
    current_strategy: str,
    persisted_artifact_count: int | None = None,
    spill_count: int | None = None,
    resident_bytes: int | None = None,
    hot_bytes: int | None = None,
) -> KVCacheAdaptationSurface:
    """Describe the current adaptation surface using observe-only rules."""

    normalized_mode = normalize_kv_cache_adaptation_mode(adaptation_mode)
    resolved_mode = (
        DEFAULT_KV_CACHE_ADAPTATION_MODE if normalized_mode is None else normalized_mode
    )
    if resolved_mode == KVCacheAdaptationMode.DISABLED.value:
        return KVCacheAdaptationSurface(
            adaptation_mode=resolved_mode,
            recommendation_available=False,
            recommended_strategy_id=None,
            reason="KV adaptation is disabled.",
        )
    normalized_strategy = normalize_kv_cache_strategy(current_strategy)
    strategy_id = (
        DEFAULT_KV_CACHE_STRATEGY
        if normalized_strategy is None
        else normalized_strategy
    )
    if resolved_mode == KVCacheAdaptationMode.OBSERVE_ONLY.value:
        if strategy_id == "resident":
            return KVCacheAdaptationSurface(
                adaptation_mode=resolved_mode,
                recommendation_available=True,
                recommended_strategy_id=strategy_id,
                reason="Resident mode is already the lowest-overhead full-history baseline.",
            )
        if (
            strategy_id in {"chunked", "streamed-segmented"}
            and persisted_artifact_count is not None
            and persisted_artifact_count >= 64
        ):
            return KVCacheAdaptationSurface(
                adaptation_mode=resolved_mode,
                recommendation_available=True,
                recommended_strategy_id="log-structured-journal",
                reason="High persisted artifact pressure favors a compaction-capable journal layout.",
            )
        if (
            strategy_id == "tiered-write-back"
            and spill_count is not None
            and spill_count == 0
        ):
            return KVCacheAdaptationSurface(
                adaptation_mode=resolved_mode,
                recommendation_available=True,
                recommended_strategy_id="log-structured-journal",
                reason="Tiered write-back has not spilled, so a non-tiered journal preset is likely a better fit.",
            )
        if (
            resident_bytes is not None
            and hot_bytes is not None
            and hot_bytes > 0
            and resident_bytes >= hot_bytes * 4
        ):
            return KVCacheAdaptationSurface(
                adaptation_mode=resolved_mode,
                recommendation_available=True,
                recommended_strategy_id=strategy_id,
                reason="Resident in-process KV dominates the active hot tail, so the current preset remains the best observe-only choice.",
            )
        return KVCacheAdaptationSurface(
            adaptation_mode=resolved_mode,
            recommendation_available=True,
            recommended_strategy_id=strategy_id,
            reason="No migration pressure was detected from the current KV state.",
        )
    return KVCacheAdaptationSurface(
        adaptation_mode=resolved_mode,
        recommendation_available=True,
        recommended_strategy_id=strategy_id,
        reason="Automatic switching is not enabled yet, so the current strategy remains pinned.",
    )


def resolve_kv_cache_base_dir(
    *,
    cache_dir: Path,
    lifecycle: str | None,
    model_reference: str,
    normalized_name: str,
    backend_id: str,
    specialization_provider_id: str | None,
) -> Path:
    """Return the lifecycle-aware base cache directory before strategy suffixing."""

    normalized_lifecycle = normalize_kv_cache_lifecycle(lifecycle)
    resolved_lifecycle = (
        DEFAULT_KV_CACHE_LIFECYCLE
        if normalized_lifecycle is None
        else normalized_lifecycle
    )
    base_dir = cache_dir.expanduser().resolve()
    if resolved_lifecycle == KVCacheLifecycle.RUNTIME_SCOPED.value:
        return base_dir
    safe_name = _SAFE_PATH_FRAGMENT.sub("-", normalized_name).strip("-")
    if not safe_name:
        safe_name = "model"
    identity_material = "|".join(
        (
            model_reference,
            normalized_name,
            backend_id,
            "" if specialization_provider_id is None else specialization_provider_id,
        )
    )
    digest = hashlib.sha256(identity_material.encode("utf-8")).hexdigest()[:12]
    return base_dir / "persistent" / f"{safe_name}-{digest}"

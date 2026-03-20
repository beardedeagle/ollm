"""Orthogonal KV strategy, lifecycle, and adaptation scaffolding."""

from dataclasses import asdict, dataclass
from enum import StrEnum

from ollm.kv_cache_strategy import (
    DEFAULT_KV_CACHE_STRATEGY,
    normalize_kv_cache_strategy,
)


class KVCachePersistenceFormat(StrEnum):
    """Describe how cold KV is persisted."""

    CHUNKED_MANIFEST = "chunked-manifest"
    STREAMED_SEGMENTED = "streamed-segmented"
    LOG_STRUCTURED_JOURNAL = "log-structured-journal"


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


def describe_kv_cache_strategy(strategy: str | None) -> KVCacheStrategyAxes:
    """Return the orthogonal axes behind a strategy preset."""

    normalized = normalize_kv_cache_strategy(strategy)
    strategy_id = DEFAULT_KV_CACHE_STRATEGY if normalized is None else normalized
    if strategy_id == "chunked":
        return KVCacheStrategyAxes(
            strategy_id=strategy_id,
            persistence_format=KVCachePersistenceFormat.CHUNKED_MANIFEST.value,
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
) -> KVCacheAdaptationSurface:
    """Describe the current adaptation surface without fabricating live switching."""

    del current_strategy
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
    if resolved_mode == KVCacheAdaptationMode.OBSERVE_ONLY.value:
        return KVCacheAdaptationSurface(
            adaptation_mode=resolved_mode,
            recommendation_available=False,
            recommended_strategy_id=None,
            reason="Observe-only scaffolding is enabled, but live recommendation rules are not implemented yet.",
        )
    return KVCacheAdaptationSurface(
        adaptation_mode=resolved_mode,
        recommendation_available=False,
        recommended_strategy_id=None,
        reason="Automatic KV adaptation is scaffolded, but no switching rules are active yet.",
    )

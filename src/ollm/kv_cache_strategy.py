"""Explicit disk-KV strategy selection and cache-root helpers."""

from pathlib import Path

DEFAULT_KV_CACHE_STRATEGY = "chunked"
KNOWN_KV_CACHE_STRATEGY_IDS = (
    "chunked",
    "streamed-segmented",
    "log-structured-journal",
    "tiered-write-back",
)

_CACHE_ROOT_BY_STRATEGY = {
    "chunked": "kv_cache_chunked",
    "streamed-segmented": "kv_cache_streamed_segmented",
    "log-structured-journal": "kv_cache_log_structured_journal",
    "tiered-write-back": "kv_cache_tiered_write_back",
}


def normalize_kv_cache_strategy(strategy: str | None) -> str | None:
    """Validate and normalize an explicit disk-KV strategy identifier."""

    if strategy is None:
        return None
    normalized_strategy = strategy.strip().lower()
    if not normalized_strategy:
        raise ValueError("--kv-cache-strategy cannot be empty")
    if normalized_strategy not in KNOWN_KV_CACHE_STRATEGY_IDS:
        allowed_strategies = ", ".join(KNOWN_KV_CACHE_STRATEGY_IDS)
        raise ValueError(f"--kv-cache-strategy must be one of: {allowed_strategies}")
    return normalized_strategy


def kv_cache_root(cache_dir: Path | str, strategy: str) -> Path:
    """Return the strategy-specific cache root under the configured cache dir."""

    normalized_strategy = normalize_kv_cache_strategy(strategy)
    if normalized_strategy is None:
        raise ValueError("kv_cache_root requires an explicit strategy")
    return Path(cache_dir) / _CACHE_ROOT_BY_STRATEGY[normalized_strategy]

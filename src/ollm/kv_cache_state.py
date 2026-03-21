"""Typed runtime state snapshots for resident and disk-backed KV strategies."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class KVCacheStateSnapshot:
    """Summarize resident, hot, and cold KV state for runtime reporting."""

    strategy_id: str
    policy_id: str
    persistence_format: str
    residency_mode: str
    window_policy: str
    cold_tier_encoding: str
    cold_tier_representation: str | None
    persisted_layer_count: int
    persisted_tokens: int
    persisted_artifact_count: int
    resident_layer_count: int
    resident_tokens: int
    resident_bytes: int
    hot_layer_count: int
    hot_tokens: int
    hot_bytes: int
    compaction_count: int
    spill_count: int
    spilled_tokens: int
    window_max_tokens: int | None = None
    eviction_policy: str | None = None
    eviction_count: int = 0
    evicted_tokens: int = 0
    cold_store_format: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary view of the snapshot."""

        return asdict(self)

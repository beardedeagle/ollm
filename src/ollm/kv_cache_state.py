"""Typed runtime state snapshots for disk-backed KV strategies."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class KVCacheStateSnapshot:
    """Summarize hot and cold KV state for runtime and benchmark reporting."""

    strategy_id: str
    policy_id: str
    persisted_layer_count: int
    persisted_tokens: int
    hot_layer_count: int
    hot_tokens: int
    hot_bytes: int
    spill_count: int
    spilled_tokens: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary view of the snapshot."""

        return asdict(self)

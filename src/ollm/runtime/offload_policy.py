"""Deterministic CPU offload policy helpers."""

from dataclasses import asdict, dataclass
from enum import StrEnum


class CpuOffloadPolicy(StrEnum):
    """Supported CPU offload placement policies."""

    AUTO = "auto"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    MIDDLE_BAND = "middle-band"


DEFAULT_CPU_OFFLOAD_POLICY = CpuOffloadPolicy.AUTO.value


@dataclass(frozen=True, slots=True)
class CpuOffloadPlacement:
    """Describe one resolved CPU offload placement."""

    requested_policy_id: str
    resolved_policy_id: str
    requested_layers: int
    applied_layers: int
    total_layers: int
    layer_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def normalize_cpu_offload_policy(policy: str | None) -> str | None:
    """Validate and normalize a CPU offload policy identifier."""

    if policy is None:
        return None
    normalized_policy = policy.strip().lower()
    if not normalized_policy:
        raise ValueError("--offload-cpu-policy cannot be empty")
    try:
        return CpuOffloadPolicy(normalized_policy).value
    except ValueError as exc:
        allowed_policies = ", ".join(policy_id.value for policy_id in CpuOffloadPolicy)
        raise ValueError(
            f"--offload-cpu-policy must be one of: {allowed_policies}"
        ) from exc


def resolve_cpu_offload_policy(policy: str | None) -> str:
    """Resolve a CPU offload policy, applying the default when omitted."""

    normalized_policy = normalize_cpu_offload_policy(policy)
    if normalized_policy is None:
        return DEFAULT_CPU_OFFLOAD_POLICY
    return normalized_policy


def plan_cpu_offload_placement(
    *,
    requested_layers: int,
    total_layers: int,
    policy: str | None,
) -> CpuOffloadPlacement:
    """Resolve concrete layer indices for a CPU offload request."""

    if requested_layers < 0:
        raise ValueError("requested_layers must be zero or greater")
    if total_layers < 0:
        raise ValueError("total_layers must be zero or greater")

    requested_policy_id = resolve_cpu_offload_policy(policy)
    resolved_policy_id = (
        CpuOffloadPolicy.MIDDLE_BAND.value
        if requested_policy_id == CpuOffloadPolicy.AUTO.value
        else requested_policy_id
    )
    applied_layers = min(requested_layers, total_layers)
    if applied_layers == 0:
        layer_indices: tuple[int, ...] = ()
    elif resolved_policy_id == CpuOffloadPolicy.PREFIX.value:
        layer_indices = tuple(range(applied_layers))
    elif resolved_policy_id == CpuOffloadPolicy.SUFFIX.value:
        layer_indices = tuple(range(total_layers - applied_layers, total_layers))
    elif resolved_policy_id == CpuOffloadPolicy.MIDDLE_BAND.value:
        start_idx = max(0, (total_layers - applied_layers) // 2)
        layer_indices = tuple(range(start_idx, start_idx + applied_layers))
    else:
        raise ValueError(f"Unsupported CPU offload policy: {resolved_policy_id}")

    return CpuOffloadPlacement(
        requested_policy_id=requested_policy_id,
        resolved_policy_id=resolved_policy_id,
        requested_layers=requested_layers,
        applied_layers=applied_layers,
        total_layers=total_layers,
        layer_indices=layer_indices,
    )


def format_layer_indices(layer_indices: tuple[int, ...]) -> str:
    """Serialize layer indices for plan and metadata output."""

    return ",".join(str(layer_idx) for layer_idx in layer_indices)


def require_hidden_layer_count(model: object) -> int:
    """Read the hidden-layer count from one loaded native model."""

    total_layers = getattr(model, "num_hidden_layers", None)
    if not isinstance(total_layers, int) or total_layers <= 0:
        raise ValueError(
            "The selected optimized runtime does not expose a valid hidden-layer count"
        )
    return total_layers

"""Deterministic runtime strategy selector for KV strategy defaults."""

import platform
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import psutil
import torch

from ollm.kv_cache_strategy import normalize_kv_cache_strategy
from ollm.runtime.catalog import ModelModality
from ollm.runtime.resolver import ResolvedModel


class StrategySelectorProfile(StrEnum):
    """Supported deterministic selector profiles."""

    BALANCED = "balanced"
    LATENCY = "latency"
    CAPACITY = "capacity"
    BOUNDED_WINDOW = "bounded-window"


DEFAULT_STRATEGY_SELECTOR_PROFILE = StrategySelectorProfile.BALANCED.value
_MODEL_SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)b", re.IGNORECASE)


class _RuntimePlanProtocol(Protocol):
    backend_id: str | None
    supports_disk_cache: bool
    specialization_enabled: bool
    details: dict[str, str]


@dataclass(frozen=True, slots=True)
class RuntimeStrategySelectorDimensions:
    """Describe the coarse dimensions used by selector rules."""

    model_family: str
    modality_bucket: str
    platform: str
    accelerator_kind: str
    host_ram_tier: str
    accelerator_memory_tier: str
    required_runtime_features: tuple[str, ...]
    model_size_tier: str

    def to_details(self) -> dict[str, str]:
        """Return string details suitable for runtime-plan reporting."""

        return {
            "strategy_selector_model_family": self.model_family,
            "strategy_selector_modality_bucket": self.modality_bucket,
            "strategy_selector_platform": self.platform,
            "strategy_selector_accelerator_kind": self.accelerator_kind,
            "strategy_selector_host_ram_tier": self.host_ram_tier,
            "strategy_selector_accelerator_memory_tier": self.accelerator_memory_tier,
            "strategy_selector_required_runtime_features": ",".join(
                self.required_runtime_features
            ),
            "strategy_selector_model_size_tier": self.model_size_tier,
        }


@dataclass(frozen=True, slots=True)
class RuntimeStrategySelection:
    """Describe one deterministic selector result."""

    profile_id: str
    rule_id: str
    requested_strategy_override: str | None
    selected_kv_cache_strategy: str
    applied_kv_cache_strategy: str
    fallback_chain: tuple[str, ...]
    reason: str
    dimensions: RuntimeStrategySelectorDimensions

    def to_details(self) -> dict[str, str]:
        """Return string details suitable for runtime-plan reporting."""

        details = self.dimensions.to_details()
        details.update(
            {
                "strategy_selector_profile": self.profile_id,
                "strategy_selector_rule_id": self.rule_id,
                "strategy_selector_requested_override": (
                    ""
                    if self.requested_strategy_override is None
                    else self.requested_strategy_override
                ),
                "strategy_selector_selected_kv_cache_strategy": (
                    self.selected_kv_cache_strategy
                ),
                "strategy_selector_applied_kv_cache_strategy": (
                    self.applied_kv_cache_strategy
                ),
                "strategy_selector_fallback_chain": ",".join(self.fallback_chain),
                "strategy_selector_reason": self.reason,
            }
        )
        return details


def normalize_strategy_selector_profile(profile: str | None) -> str | None:
    """Validate and normalize a selector-profile identifier."""

    if profile is None:
        return None
    normalized = profile.strip().lower()
    if not normalized:
        raise ValueError("strategy_selector_profile cannot be empty")
    try:
        return StrategySelectorProfile(normalized).value
    except ValueError as exc:
        allowed = ", ".join(item.value for item in StrategySelectorProfile)
        raise ValueError(
            f"strategy_selector_profile must be one of: {allowed}"
        ) from exc


def resolve_strategy_selector_profile(profile: str | None) -> str:
    """Resolve a selector profile, applying the default when omitted."""

    normalized = normalize_strategy_selector_profile(profile)
    if normalized is None:
        return DEFAULT_STRATEGY_SELECTOR_PROFILE
    return normalized


def select_runtime_strategy(
    *,
    resolved_model: ResolvedModel,
    runtime_plan: _RuntimePlanProtocol,
    requested_strategy_override: str | None,
    strategy_selector_profile: str | None,
    requested_window_tokens: int | None,
) -> RuntimeStrategySelection:
    """Resolve the deterministic KV strategy choice for one runtime plan."""

    dimensions = build_runtime_strategy_dimensions(
        resolved_model=resolved_model,
        runtime_plan=runtime_plan,
    )
    explicit_override = normalize_kv_cache_strategy(requested_strategy_override)
    if explicit_override is not None:
        if explicit_override != "resident" and not runtime_plan.supports_disk_cache:
            raise ValueError(
                f"Requested kv_cache_strategy '{explicit_override}' requires disk-cache "
                f"support on backend '{runtime_plan.backend_id}'."
            )
        return RuntimeStrategySelection(
            profile_id=resolve_strategy_selector_profile(strategy_selector_profile),
            rule_id="explicit-override",
            requested_strategy_override=explicit_override,
            selected_kv_cache_strategy=explicit_override,
            applied_kv_cache_strategy=explicit_override,
            fallback_chain=(explicit_override,),
            reason=(
                f"Using explicit kv_cache_strategy override '{explicit_override}'."
            ),
            dimensions=dimensions,
        )

    profile_id = resolve_strategy_selector_profile(strategy_selector_profile)
    if profile_id == StrategySelectorProfile.BOUNDED_WINDOW.value:
        if not runtime_plan.supports_disk_cache:
            raise ValueError(
                "--strategy-selector-profile bounded-window requires a backend with "
                "disk-cache support."
            )
        return RuntimeStrategySelection(
            profile_id=profile_id,
            rule_id="bounded-window-profile",
            requested_strategy_override=None,
            selected_kv_cache_strategy="sliding-window-ring-buffer",
            applied_kv_cache_strategy="sliding-window-ring-buffer",
            fallback_chain=("sliding-window-ring-buffer",),
            reason=(
                "The bounded-window selector profile keeps sliding-window semantics "
                "explicit instead of silently falling back to full-history presets."
            ),
            dimensions=dimensions,
        )

    if requested_window_tokens is not None:
        if not runtime_plan.supports_disk_cache:
            raise ValueError(
                "--kv-cache-window-tokens requires a backend with disk-cache support."
            )
        return RuntimeStrategySelection(
            profile_id=profile_id,
            rule_id="explicit-window-budget",
            requested_strategy_override=None,
            selected_kv_cache_strategy="sliding-window-ring-buffer",
            applied_kv_cache_strategy="sliding-window-ring-buffer",
            fallback_chain=("sliding-window-ring-buffer",),
            reason=(
                "An explicit kv_cache_window_tokens value opts into bounded "
                "sliding-window semantics."
            ),
            dimensions=dimensions,
        )

    if not runtime_plan.supports_disk_cache:
        return RuntimeStrategySelection(
            profile_id=profile_id,
            rule_id="no-disk-cache-support",
            requested_strategy_override=None,
            selected_kv_cache_strategy="resident",
            applied_kv_cache_strategy="resident",
            fallback_chain=("resident",),
            reason=(
                f"Backend '{runtime_plan.backend_id}' does not support disk-backed KV, "
                "so resident mode is the truthful fallback."
            ),
            dimensions=dimensions,
        )

    if profile_id == StrategySelectorProfile.CAPACITY.value:
        if dimensions.modality_bucket == "text-only":
            return RuntimeStrategySelection(
                profile_id=profile_id,
                rule_id="capacity-text-full-history",
                requested_strategy_override=None,
                selected_kv_cache_strategy="quantized-cold-tier",
                applied_kv_cache_strategy="quantized-cold-tier",
                fallback_chain=("quantized-cold-tier", "paged", "chunked", "resident"),
                reason=(
                    "Capacity profile prefers the selector-eligible quantized cold tier "
                    "for full-history text workloads."
                ),
                dimensions=dimensions,
            )
        return _paged_selection(
            profile_id=profile_id,
            rule_id="capacity-multimodal-conservative",
            reason=(
                "Capacity profile stays on paged full-history storage for multimodal "
                "workloads because quantized cold-tier evidence is still text-only."
            ),
            dimensions=dimensions,
        )

    if profile_id == StrategySelectorProfile.LATENCY.value:
        if (
            dimensions.modality_bucket == "text-only"
            and dimensions.host_ram_tier == "large"
            and dimensions.model_size_tier in {"small", "medium"}
        ):
            return RuntimeStrategySelection(
                profile_id=profile_id,
                rule_id="latency-high-headroom-resident",
                requested_strategy_override=None,
                selected_kv_cache_strategy="resident",
                applied_kv_cache_strategy="resident",
                fallback_chain=("resident", "paged", "chunked"),
                reason=(
                    "Latency profile prefers resident KV when the model size and host "
                    "memory tier indicate enough headroom."
                ),
                dimensions=dimensions,
            )
        return _paged_selection(
            profile_id=profile_id,
            rule_id="latency-conservative-paged",
            reason=(
                "Latency profile falls back to paged full-history KV when resident "
                "headroom is not clearly available."
            ),
            dimensions=dimensions,
        )

    if (
        dimensions.modality_bucket == "text-only"
        and dimensions.model_size_tier == "small"
        and (
            dimensions.host_ram_tier == "large"
            or dimensions.accelerator_memory_tier == "large"
        )
    ):
        return RuntimeStrategySelection(
            profile_id=profile_id,
            rule_id="balanced-small-model-resident",
            requested_strategy_override=None,
            selected_kv_cache_strategy="resident",
            applied_kv_cache_strategy="resident",
            fallback_chain=("resident", "paged", "chunked"),
            reason=(
                "Balanced profile uses resident KV for small text models when the "
                "host or accelerator memory tier is clearly high."
            ),
            dimensions=dimensions,
        )

    return _paged_selection(
        profile_id=profile_id,
        rule_id="balanced-paged-default",
        reason=(
            "Balanced profile uses paged KV as the canonical full-history default "
            "with deterministic page boundaries and a known-good fallback chain."
        ),
        dimensions=dimensions,
    )


def build_runtime_strategy_dimensions(
    *,
    resolved_model: ResolvedModel,
    runtime_plan: _RuntimePlanProtocol,
) -> RuntimeStrategySelectorDimensions:
    """Capture the coarse dimensions used by selector rules."""

    platform_id = platform.system().lower() or "unknown"
    accelerator_kind = _accelerator_kind(
        runtime_plan.details.get("execution_device_type")
    )
    host_ram_tier = _host_ram_tier()
    accelerator_memory_tier = _accelerator_memory_tier(accelerator_kind)
    required_runtime_features = _required_runtime_features(
        resolved_model=resolved_model,
        runtime_plan=runtime_plan,
    )
    return RuntimeStrategySelectorDimensions(
        model_family=(
            "generic"
            if resolved_model.native_family is None
            else resolved_model.native_family.value
        ),
        modality_bucket=(
            "text-only"
            if resolved_model.capabilities.modalities == (ModelModality.TEXT,)
            else "multimodal"
        ),
        platform=platform_id,
        accelerator_kind=accelerator_kind,
        host_ram_tier=host_ram_tier,
        accelerator_memory_tier=accelerator_memory_tier,
        required_runtime_features=required_runtime_features,
        model_size_tier=_model_size_tier(resolved_model),
    )


def _required_runtime_features(
    *,
    resolved_model: ResolvedModel,
    runtime_plan: _RuntimePlanProtocol,
) -> tuple[str, ...]:
    features: list[str] = []
    if runtime_plan.supports_disk_cache:
        features.append("disk-cache-support")
    else:
        features.append("no-disk-cache-support")
    if resolved_model.capabilities.requires_processor:
        features.append("processor-required")
    if runtime_plan.specialization_enabled:
        features.append("specialization-enabled")
    return tuple(features)


def _paged_selection(
    *,
    profile_id: str,
    rule_id: str,
    reason: str,
    dimensions: RuntimeStrategySelectorDimensions,
) -> RuntimeStrategySelection:
    return RuntimeStrategySelection(
        profile_id=profile_id,
        rule_id=rule_id,
        requested_strategy_override=None,
        selected_kv_cache_strategy="paged",
        applied_kv_cache_strategy="paged",
        fallback_chain=("paged", "chunked", "resident"),
        reason=reason,
        dimensions=dimensions,
    )


def _accelerator_kind(execution_device_type: str | None) -> str:
    if execution_device_type is not None:
        return execution_device_type
    return "unknown"


def _host_ram_tier() -> str:
    try:
        total_gib = psutil.virtual_memory().total / (1024**3)
    except Exception:
        return "unknown"
    if total_gib < 16:
        return "small"
    if total_gib < 64:
        return "medium"
    return "large"


def _accelerator_memory_tier(accelerator_kind: str) -> str:
    total_gib = None
    if accelerator_kind == "cuda":
        try:
            if torch.cuda.is_available():
                total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            total_gib = None
    elif accelerator_kind == "mps":
        try:
            total_gib = psutil.virtual_memory().total / (1024**3)
        except Exception:
            total_gib = None
    if total_gib is None:
        return "unknown" if accelerator_kind != "cpu" else "none"
    if total_gib < 8:
        return "small"
    if total_gib < 24:
        return "medium"
    return "large"


def _model_size_tier(resolved_model: ResolvedModel) -> str:
    candidates = (
        resolved_model.normalized_name,
        resolved_model.reference.raw,
        "" if resolved_model.repo_id is None else resolved_model.repo_id,
    )
    for candidate in candidates:
        match = _MODEL_SIZE_PATTERN.search(candidate)
        if match is None:
            continue
        size_in_billions = float(match.group(1))
        if size_in_billions <= 4:
            return "small"
        if size_in_billions <= 16:
            return "medium"
        return "large"
    return "unknown"

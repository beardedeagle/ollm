from pathlib import Path

from ollm.kv_cache.matrix import (
    KVCacheAdaptationSurface,
    build_kv_cache_adaptation_surface,
    describe_kv_cache_strategy,
    normalize_kv_cache_adaptation_mode,
    normalize_kv_cache_lifecycle,
    resolve_kv_cache_base_dir,
    resolve_kv_cache_eviction_policy,
    resolve_kv_cache_lifecycle,
    resolve_kv_cache_window_tokens,
)


def test_describe_kv_cache_strategy_maps_resident_axes() -> None:
    axes = describe_kv_cache_strategy("resident")

    assert axes.strategy_id == "resident"
    assert axes.persistence_format == "resident-only"
    assert axes.residency_mode == "fully-resident"
    assert axes.window_policy == "full-history"
    assert axes.cold_tier_encoding == "full-precision"
    assert axes.compaction_capable is False


def test_describe_kv_cache_strategy_maps_chunked_axes() -> None:
    axes = describe_kv_cache_strategy("chunked")

    assert axes.strategy_id == "chunked"
    assert axes.persistence_format == "chunked-manifest"
    assert axes.residency_mode == "buffered-tail"
    assert axes.window_policy == "full-history"
    assert axes.cold_tier_encoding == "full-precision"
    assert axes.compaction_capable is False


def test_describe_kv_cache_strategy_maps_paged_axes() -> None:
    axes = describe_kv_cache_strategy("paged")

    assert axes.strategy_id == "paged"
    assert axes.persistence_format == "paged-manifest"
    assert axes.residency_mode == "buffered-tail"
    assert axes.window_policy == "full-history"
    assert axes.cold_tier_encoding == "full-precision"
    assert axes.compaction_capable is False


def test_describe_kv_cache_strategy_maps_tiered_axes() -> None:
    axes = describe_kv_cache_strategy("tiered-write-back")

    assert axes.strategy_id == "tiered-write-back"
    assert axes.persistence_format == "log-structured-journal"
    assert axes.residency_mode == "tiered-write-back"
    assert axes.window_policy == "full-history"
    assert axes.cold_tier_encoding == "full-precision"
    assert axes.compaction_capable is True


def test_describe_kv_cache_strategy_maps_quantized_cold_tier_axes() -> None:
    axes = describe_kv_cache_strategy("quantized-cold-tier")

    assert axes.strategy_id == "quantized-cold-tier"
    assert axes.persistence_format == "log-structured-journal"
    assert axes.residency_mode == "buffered-tail"
    assert axes.window_policy == "full-history"
    assert axes.cold_tier_encoding == "quantized"
    assert axes.compaction_capable is True


def test_describe_kv_cache_strategy_maps_sliding_window_axes() -> None:
    axes = describe_kv_cache_strategy("sliding-window-ring-buffer")

    assert axes.strategy_id == "sliding-window-ring-buffer"
    assert axes.persistence_format == "sliding-window-ring-buffer"
    assert axes.residency_mode == "buffered-tail"
    assert axes.window_policy == "sliding-window"
    assert axes.cold_tier_encoding == "full-precision"
    assert axes.compaction_capable is False


def test_normalize_kv_cache_lifecycle_accepts_known_values() -> None:
    assert normalize_kv_cache_lifecycle("persistent") == "persistent"
    assert normalize_kv_cache_lifecycle("runtime-scoped") == "runtime-scoped"


def test_resolve_kv_cache_lifecycle_rejects_persistent_resident() -> None:
    try:
        resolve_kv_cache_lifecycle("resident", "persistent")
    except ValueError as exc:
        assert "--kv-cache-strategy resident requires" in str(exc)
    else:
        raise AssertionError("Resident mode should reject persistent lifecycle")


def test_normalize_kv_cache_adaptation_mode_accepts_known_values() -> None:
    assert normalize_kv_cache_adaptation_mode("disabled") == "disabled"
    assert normalize_kv_cache_adaptation_mode("observe-only") == "observe-only"
    assert normalize_kv_cache_adaptation_mode("automatic") == "automatic"


def test_resolve_kv_cache_window_tokens_for_sliding_window_strategy() -> None:
    assert resolve_kv_cache_window_tokens("sliding-window-ring-buffer", None) == 256
    assert resolve_kv_cache_window_tokens("sliding-window-ring-buffer", 96) == 96


def test_resolve_kv_cache_window_tokens_rejects_non_sliding_strategies() -> None:
    try:
        resolve_kv_cache_window_tokens("chunked", 32)
    except ValueError as exc:
        assert "--kv-cache-window-tokens requires --kv-cache-strategy" in str(exc)
    else:
        raise AssertionError(
            "Non-sliding strategies should reject explicit window tokens"
        )


def test_resolve_kv_cache_eviction_policy_for_sliding_window_strategy() -> None:
    assert (
        resolve_kv_cache_eviction_policy("sliding-window-ring-buffer") == "drop-oldest"
    )


def test_resolve_kv_cache_window_tokens_requires_sliding_strategy() -> None:
    assert resolve_kv_cache_window_tokens("sliding-window-ring-buffer", None) == 256
    assert resolve_kv_cache_window_tokens("sliding-window-ring-buffer", 48) == 48
    assert resolve_kv_cache_window_tokens("chunked", None) is None

    try:
        resolve_kv_cache_window_tokens("chunked", 48)
    except ValueError as exc:
        assert "--kv-cache-window-tokens requires" in str(exc)
    else:
        raise AssertionError(
            "resolve_kv_cache_window_tokens should reject non-sliding strategies"
        )


def test_resolve_kv_cache_eviction_policy_is_drop_oldest_for_sliding_window() -> None:
    assert (
        resolve_kv_cache_eviction_policy("sliding-window-ring-buffer") == "drop-oldest"
    )
    assert resolve_kv_cache_eviction_policy("chunked") is None


def test_kv_cache_adaptation_surface_serializes() -> None:
    surface = KVCacheAdaptationSurface(
        adaptation_mode="observe-only",
        recommendation_available=False,
        recommended_strategy_id=None,
        reason="Observe-only scaffolding is enabled.",
    )

    assert surface.to_dict()["adaptation_mode"] == "observe-only"


def test_build_kv_cache_adaptation_surface_recommends_journal_for_artifact_pressure() -> (
    None
):
    surface = build_kv_cache_adaptation_surface(
        adaptation_mode="observe-only",
        current_strategy="chunked",
        persisted_artifact_count=96,
        spill_count=0,
        resident_bytes=0,
        hot_bytes=1024,
    )

    assert surface.recommendation_available is True
    assert surface.recommended_strategy_id == "log-structured-journal"


def test_build_kv_cache_adaptation_surface_pins_resident_baseline() -> None:
    surface = build_kv_cache_adaptation_surface(
        adaptation_mode="observe-only",
        current_strategy="resident",
        persisted_artifact_count=0,
        spill_count=0,
        resident_bytes=4096,
        hot_bytes=0,
    )

    assert surface.recommendation_available is True
    assert surface.recommended_strategy_id == "resident"


def test_resolve_kv_cache_base_dir_namespaces_persistent_lifecycle() -> None:
    persistent_dir = resolve_kv_cache_base_dir(
        cache_dir=Path("/tmp/kv-cache"),
        lifecycle="persistent",
        model_reference="Qwen/Qwen2.5-7B-Instruct",
        normalized_name="qwen2.5-7b",
        backend_id="optimized-native",
        specialization_provider_id="qwen3-next-native",
    )

    assert persistent_dir.parts[-2] == "persistent"
    assert persistent_dir.parts[-1].startswith("qwen2.5-7b-")

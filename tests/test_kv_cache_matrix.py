from pathlib import Path

from ollm.kv_cache_matrix import (
    KVCacheAdaptationSurface,
    build_kv_cache_adaptation_surface,
    describe_kv_cache_strategy,
    normalize_kv_cache_adaptation_mode,
    normalize_kv_cache_lifecycle,
    resolve_kv_cache_base_dir,
)


def test_describe_kv_cache_strategy_maps_chunked_axes() -> None:
    axes = describe_kv_cache_strategy("chunked")

    assert axes.strategy_id == "chunked"
    assert axes.persistence_format == "chunked-manifest"
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


def test_normalize_kv_cache_lifecycle_accepts_known_values() -> None:
    assert normalize_kv_cache_lifecycle("persistent") == "persistent"
    assert normalize_kv_cache_lifecycle("runtime-scoped") == "runtime-scoped"


def test_normalize_kv_cache_adaptation_mode_accepts_known_values() -> None:
    assert normalize_kv_cache_adaptation_mode("disabled") == "disabled"
    assert normalize_kv_cache_adaptation_mode("observe-only") == "observe-only"
    assert normalize_kv_cache_adaptation_mode("automatic") == "automatic"


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

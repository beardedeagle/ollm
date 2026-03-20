from ollm.kv_cache_matrix import KVCacheAdaptationSurface
from ollm.kv_cache_state import KVCacheStateSnapshot
from ollm.runtime.benchmark_probes import (
    EventTimingSummary,
    NativeRuntimeProfile,
    RequestProbeMetrics,
)
from ollm.runtime.benchmark_resources import StageResourceSnapshot


def build_stage_resources() -> StageResourceSnapshot:
    return StageResourceSnapshot(
        current_rss_mb=128.0,
        peak_rss_mb=132.0,
        peak_rss_source="native",
        accelerator_kind="cuda",
        accelerator_current_mb=256.0,
        accelerator_peak_mb=300.0,
        accelerator_reserved_mb=280.0,
        accelerator_peak_reserved_mb=320.0,
        accelerator_peak_source="native",
        process_cpu_utilization_percent=88.0,
        accelerator_utilization=None,
    )


def build_native_runtime_profile() -> NativeRuntimeProfile:
    return NativeRuntimeProfile(
        storage_paths=("disk-kv-cache", "safetensor-io"),
        events={
            "kvload": EventTimingSummary(
                count=2,
                total_ms=8.0,
                min_ms=3.0,
                median_ms=4.0,
                p95_ms=5.0,
                max_ms=5.0,
                mean_ms=4.0,
            ),
            "layer_load": EventTimingSummary(
                count=4,
                total_ms=20.0,
                min_ms=4.0,
                median_ms=5.0,
                p95_ms=6.0,
                max_ms=6.0,
                mean_ms=5.0,
            ),
        },
    )


def build_request_probe_metrics() -> RequestProbeMetrics:
    return RequestProbeMetrics(
        total_ms=30.0,
        generation_ms=20.0,
        time_to_first_token_ms=5.0,
        inter_token_latencies_ms=(2.0, 2.5, 3.0),
        prompt_tokens=16,
        prompt_tokens_per_second=3200.0,
        output_tokens=4,
        output_tokens_per_second=200.0,
        cache_mode="disk-kv",
        kv_cache_strategy="chunked",
        kv_cache_adaptation=KVCacheAdaptationSurface(
            adaptation_mode="observe-only",
            recommendation_available=True,
            recommended_strategy_id="chunked",
            reason="No migration pressure was detected from the current KV state.",
        ),
        cache_dir_size_mb=12.0,
        cache_state=KVCacheStateSnapshot(
            strategy_id="chunked",
            policy_id="test-policy",
            persistence_format="chunked-manifest",
            residency_mode="buffered-tail",
            window_policy="full-history",
            cold_tier_encoding="full-precision",
            cold_tier_representation=None,
            persisted_layer_count=2,
            persisted_tokens=128,
            persisted_artifact_count=2,
            resident_layer_count=2,
            resident_tokens=128,
            resident_bytes=4096,
            hot_layer_count=1,
            hot_tokens=16,
            hot_bytes=8192,
            compaction_count=0,
            spill_count=0,
            spilled_tokens=0,
            cold_store_format=None,
        ),
        allocator_gap_mb=20.0,
        allocator_gap_ratio=0.066667,
        native_runtime_profile=build_native_runtime_profile(),
        resources=build_stage_resources(),
        text_excerpt="Hello",
    )

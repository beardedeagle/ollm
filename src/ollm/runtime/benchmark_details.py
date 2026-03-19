"""Shared benchmark detail summarization helpers."""

import statistics

from ollm.runtime.benchmark_probe_types import NativeRuntimeProfile
from ollm.runtime.benchmark_probes import RequestProbeMetrics, RuntimeProbeResult
from ollm.runtime.benchmark_resources import (
    StageResourceSnapshot,
    summarize_optional_numeric_values,
)
from ollm.runtime.benchmark_types import (
    BenchmarkMeasurement,
    CommandBenchmarkSpec,
)


def build_cold_probe_details(
    spec: CommandBenchmarkSpec,
    samples: list[RuntimeProbeResult],
) -> dict[str, object]:
    """Build the details payload for cold runtime probe measurements."""

    return {
        "command": list(spec.command),
        "load": {
            "latency_ms": summarize_numeric_values(
                [sample.load_ms for sample in samples]
            ),
            "resources": summarize_stage_resources(
                [sample.load_resources for sample in samples]
            ),
        },
        "metrics": summarize_request_metrics([sample.request for sample in samples]),
        "text_excerpt": samples[-1].request.text_excerpt,
    }


def summarize_request_metrics(
    samples: list[RequestProbeMetrics],
) -> dict[str, object]:
    """Summarize request-level runtime probe metrics."""

    return {
        "latency_ms": {
            "total": summarize_numeric_values([sample.total_ms for sample in samples]),
            "generation": summarize_numeric_values(
                [sample.generation_ms for sample in samples]
            ),
            "time_to_first_token": optional_summary_dict(
                [
                    sample.time_to_first_token_ms
                    for sample in samples
                    if sample.time_to_first_token_ms is not None
                ]
            ),
            "inter_token_latency": optional_summary_dict(
                [
                    latency
                    for sample in samples
                    for latency in sample.inter_token_latencies_ms
                ]
            ),
        },
        "throughput": {
            "prompt_tokens": summarize_numeric_values(
                [float(sample.prompt_tokens) for sample in samples]
            ),
            "prompt_tokens_per_second": optional_summary_dict(
                [
                    sample.prompt_tokens_per_second
                    for sample in samples
                    if sample.prompt_tokens_per_second is not None
                ]
            ),
            "output_tokens": summarize_numeric_values(
                [float(sample.output_tokens) for sample in samples]
            ),
            "output_tokens_per_second": optional_summary_dict(
                [
                    sample.output_tokens_per_second
                    for sample in samples
                    if sample.output_tokens_per_second is not None
                ]
            ),
        },
        "memory": summarize_stage_resources([sample.resources for sample in samples]),
        "cache": {
            "cache_mode": single_optional_string(
                [sample.cache_mode for sample in samples]
            ),
            "kv_cache_strategy": single_optional_string(
                [
                    sample.kv_cache_strategy
                    for sample in samples
                    if sample.kv_cache_strategy is not None
                ]
            ),
            "cache_dir_size_mb": optional_summary_dict(
                [
                    sample.cache_dir_size_mb
                    for sample in samples
                    if sample.cache_dir_size_mb is not None
                ]
            ),
            "cache_state": summarize_cache_states(samples),
        },
        "allocator": {
            "allocator_gap_mb": optional_summary_dict(
                [
                    sample.allocator_gap_mb
                    for sample in samples
                    if sample.allocator_gap_mb is not None
                ]
            ),
            "allocator_gap_ratio": optional_summary_dict(
                [
                    sample.allocator_gap_ratio
                    for sample in samples
                    if sample.allocator_gap_ratio is not None
                ]
            ),
        },
        "native_runtime_profile": summarize_native_runtime_profiles(samples),
    }


def summarize_stage_resources(
    snapshots: list[StageResourceSnapshot],
) -> dict[str, object]:
    """Summarize stage-local resource snapshots."""

    accelerator_utilizations = [
        snapshot.accelerator_utilization
        for snapshot in snapshots
        if snapshot.accelerator_utilization is not None
    ]
    gpu_utilization_means = [
        utilization.gpu_utilization_percent.mean
        for utilization in accelerator_utilizations
        if utilization.gpu_utilization_percent is not None
    ]
    memory_utilization_means = [
        utilization.memory_utilization_percent.mean
        for utilization in accelerator_utilizations
        if utilization.memory_utilization_percent is not None
    ]
    return {
        "current_rss_mb": optional_summary_dict(
            [
                snapshot.current_rss_mb
                for snapshot in snapshots
                if snapshot.current_rss_mb is not None
            ]
        ),
        "peak_rss_mb": optional_summary_dict(
            [
                snapshot.peak_rss_mb
                for snapshot in snapshots
                if snapshot.peak_rss_mb is not None
            ]
        ),
        "peak_rss_source": single_optional_string(
            [
                snapshot.peak_rss_source
                for snapshot in snapshots
                if snapshot.peak_rss_source
            ]
        ),
        "accelerator_kind": single_optional_string(
            [
                snapshot.accelerator_kind
                for snapshot in snapshots
                if snapshot.accelerator_kind
            ]
        ),
        "accelerator_current_mb": optional_summary_dict(
            [
                snapshot.accelerator_current_mb
                for snapshot in snapshots
                if snapshot.accelerator_current_mb is not None
            ]
        ),
        "accelerator_peak_mb": optional_summary_dict(
            [
                snapshot.accelerator_peak_mb
                for snapshot in snapshots
                if snapshot.accelerator_peak_mb is not None
            ]
        ),
        "accelerator_reserved_mb": optional_summary_dict(
            [
                snapshot.accelerator_reserved_mb
                for snapshot in snapshots
                if snapshot.accelerator_reserved_mb is not None
            ]
        ),
        "accelerator_peak_reserved_mb": optional_summary_dict(
            [
                snapshot.accelerator_peak_reserved_mb
                for snapshot in snapshots
                if snapshot.accelerator_peak_reserved_mb is not None
            ]
        ),
        "accelerator_peak_source": single_optional_string(
            [
                snapshot.accelerator_peak_source
                for snapshot in snapshots
                if snapshot.accelerator_peak_source
            ]
        ),
        "process_cpu_utilization_percent": optional_summary_dict(
            [
                snapshot.process_cpu_utilization_percent
                for snapshot in snapshots
                if snapshot.process_cpu_utilization_percent is not None
            ]
        ),
        "accelerator_utilization_percent": optional_summary_dict(gpu_utilization_means),
        "accelerator_memory_utilization_percent": optional_summary_dict(
            memory_utilization_means
        ),
    }


def summarize_native_runtime_profiles(
    samples: list[RequestProbeMetrics],
) -> dict[str, object] | None:
    profiles = [
        sample.native_runtime_profile
        for sample in samples
        if sample.native_runtime_profile is not None
    ]
    if not profiles:
        return None
    typed_profiles = [profile for profile in profiles if profile is not None]
    event_names = sorted(
        {event_name for profile in typed_profiles for event_name in profile.events}
    )
    return {
        "storage_paths": sorted(
            {path for profile in typed_profiles for path in profile.storage_paths}
        ),
        "events": {
            event_name: _summarize_native_runtime_event(typed_profiles, event_name)
            for event_name in event_names
        },
    }


def summarize_cache_states(
    samples: list[RequestProbeMetrics],
) -> dict[str, object] | None:
    snapshots = [
        sample.cache_state for sample in samples if sample.cache_state is not None
    ]
    if not snapshots:
        return None
    return {
        "strategy_id": single_optional_string(
            [snapshot.strategy_id for snapshot in snapshots]
        ),
        "policy_id": single_optional_string(
            [snapshot.policy_id for snapshot in snapshots]
        ),
        "persisted_layer_count": summarize_numeric_values(
            [float(snapshot.persisted_layer_count) for snapshot in snapshots]
        ),
        "persisted_tokens": summarize_numeric_values(
            [float(snapshot.persisted_tokens) for snapshot in snapshots]
        ),
        "persisted_artifact_count": summarize_numeric_values(
            [float(snapshot.persisted_artifact_count) for snapshot in snapshots]
        ),
        "hot_layer_count": summarize_numeric_values(
            [float(snapshot.hot_layer_count) for snapshot in snapshots]
        ),
        "hot_tokens": summarize_numeric_values(
            [float(snapshot.hot_tokens) for snapshot in snapshots]
        ),
        "hot_bytes": summarize_numeric_values(
            [float(snapshot.hot_bytes) for snapshot in snapshots]
        ),
        "compaction_count": summarize_numeric_values(
            [float(snapshot.compaction_count) for snapshot in snapshots]
        ),
        "cold_store_format": single_optional_string(
            [
                snapshot.cold_store_format
                for snapshot in snapshots
                if snapshot.cold_store_format is not None
            ]
        ),
        "spill_count": summarize_numeric_values(
            [float(snapshot.spill_count) for snapshot in snapshots]
        ),
        "spilled_tokens": summarize_numeric_values(
            [float(snapshot.spilled_tokens) for snapshot in snapshots]
        ),
    }


def _summarize_native_runtime_event(
    profiles: list[NativeRuntimeProfile],
    event_name: str,
) -> dict[str, object]:
    summaries = [
        profile.events[event_name]
        for profile in profiles
        if event_name in profile.events
    ]
    return {
        "sample_count": len(summaries),
        "event_count": sum(summary.count for summary in summaries),
        "total_ms": optional_summary_dict([summary.total_ms for summary in summaries]),
        "mean_ms": optional_summary_dict([summary.mean_ms for summary in summaries]),
        "max_ms": optional_summary_dict([summary.max_ms for summary in summaries]),
    }


def backend_pair_payload(
    generic: BenchmarkMeasurement,
    optimized: BenchmarkMeasurement,
) -> dict[str, object]:
    """Build the generic-vs-optimized comparison payload."""

    comparison_available = (
        generic.status == "measured" and optimized.status == "measured"
    )
    speedup_ratio = None
    reason = None
    if comparison_available:
        assert generic.stats is not None
        assert optimized.stats is not None
        speedup_ratio = round(generic.stats.mean_ms / optimized.stats.mean_ms, 6)
    else:
        reason = measurement_reason(generic)
        if reason is None:
            reason = measurement_reason(optimized)
        if reason is None:
            reason = runtime_comparison_unavailable_reason(generic, optimized)
    return {
        "generic": generic.to_dict(),
        "optimized_native": optimized.to_dict(),
        "comparison_available": comparison_available,
        "speedup_ratio": speedup_ratio,
        "reason": reason,
    }


def summarize_numeric_values(values: list[float]) -> dict[str, float]:
    """Summarize numeric values with latency-friendly percentiles."""

    sorted_values = sorted(values)
    p95_index = max(0, int(round((len(sorted_values) - 1) * 0.95)))
    return {
        "min": round(sorted_values[0], 6),
        "median": round(statistics.median(sorted_values), 6),
        "p95": round(sorted_values[p95_index], 6),
        "max": round(sorted_values[-1], 6),
        "mean": round(statistics.fmean(sorted_values), 6),
    }


def single_optional_string(values: list[str]) -> str | None:
    """Return a shared string when every value agrees."""

    if not values:
        return None
    if len(set(values)) != 1:
        return None
    return values[0]


def optional_summary_dict(values: list[float]) -> dict[str, float] | None:
    """Return an optional numeric summary dictionary."""

    summary = summarize_optional_numeric_values(values)
    if summary is None:
        return None
    return summary.to_dict()


def mean_delta_ms(
    left: BenchmarkMeasurement, right: BenchmarkMeasurement
) -> float | None:
    """Return the mean delta between two measurements when both are measured."""

    if left.stats is None or right.stats is None:
        return None
    return round(left.stats.mean_ms - right.stats.mean_ms, 6)


def clip_text(text: str, max_chars: int = 400) -> str:
    """Collapse whitespace and clip long subprocess output."""

    normalized_text = " ".join(text.split())
    if len(normalized_text) <= max_chars:
        return normalized_text
    return normalized_text[: max_chars - 3] + "..."


def runtime_comparison_unavailable_reason(
    generic: BenchmarkMeasurement,
    optimized: BenchmarkMeasurement,
) -> str:
    """Explain why a generic-vs-optimized comparison is unavailable."""

    if generic.status != "measured" and optimized.status != "measured":
        return "Neither runtime benchmark completed successfully on this host"
    if generic.status != "measured":
        return (
            "The generic runtime benchmark did not complete successfully on this host"
        )
    return "The optimized-native runtime benchmark did not complete successfully on this host"


def measurement_reason(measurement: BenchmarkMeasurement) -> str | None:
    """Return the explicit reason stored on a measurement payload."""

    reason = measurement.details.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    return None

import json
from dataclasses import replace
from typing import cast

from ollm.runtime.benchmark import render_runtime_probe_json
from ollm.runtime.benchmark.details import summarize_request_metrics
from ollm.runtime.benchmark.probes import RuntimeProbeResult
from tests.benchmark_support import (
    build_request_probe_metrics,
    build_stage_resources,
)


def test_summarize_request_metrics_reports_sliding_window_cache_state() -> None:
    metrics = build_request_probe_metrics()
    cache_state = metrics.cache_state
    assert cache_state is not None
    sliding_metrics = replace(
        metrics,
        kv_cache_strategy="sliding-window-ring-buffer",
        cache_state=replace(
            cache_state,
            strategy_id="sliding-window-ring-buffer",
            persistence_format="sliding-window-ring-buffer",
            window_policy="sliding-window",
            window_max_tokens=64,
            eviction_policy="drop-oldest",
            eviction_count=3,
            evicted_tokens=48,
            cold_store_format="ollm-kv-sliding-window",
        ),
    )

    summary = summarize_request_metrics([sliding_metrics])
    cache = cast(dict[str, object], summary["cache"])
    cache_state_summary = cast(dict[str, object], cache["cache_state"])

    assert cache["kv_cache_strategy"] == "sliding-window-ring-buffer"
    assert cache_state_summary["window_policy"] == "sliding-window"
    assert (
        cast(dict[str, float], cache_state_summary["window_max_tokens"])["mean"] == 64.0
    )
    assert cast(dict[str, float], cache_state_summary["eviction_count"])["mean"] == 3.0
    assert cast(dict[str, float], cache_state_summary["evicted_tokens"])["mean"] == 48.0


def test_summarize_request_metrics_reports_paged_cache_state() -> None:
    metrics = build_request_probe_metrics()
    cache_state = metrics.cache_state
    assert cache_state is not None
    paged_metrics = replace(
        metrics,
        kv_cache_strategy="paged",
        cache_state=replace(
            cache_state,
            strategy_id="paged",
            persistence_format="paged-manifest",
            persisted_artifact_count=3,
            cold_store_format="ollm-kv-paged",
        ),
    )

    summary = summarize_request_metrics([paged_metrics])
    cache = cast(dict[str, object], summary["cache"])
    cache_state_summary = cast(dict[str, object], cache["cache_state"])

    assert cache["kv_cache_strategy"] == "paged"
    assert cache_state_summary["strategy_id"] == "paged"
    assert cache_state_summary["persistence_format"] == "paged-manifest"
    assert cache_state_summary["cold_store_format"] == "ollm-kv-paged"
    assert (
        cast(dict[str, float], cache_state_summary["persisted_artifact_count"])["mean"]
        == 3.0
    )


def test_summarize_request_metrics_reports_resident_cache_state() -> None:
    metrics = build_request_probe_metrics()
    cache_state = metrics.cache_state
    assert cache_state is not None
    resident_metrics = replace(
        metrics,
        cache_mode="resident-kv",
        kv_cache_strategy="resident",
        cache_dir_size_mb=None,
        native_runtime_profile=None,
        cache_state=replace(
            cache_state,
            strategy_id="resident",
            policy_id="resident-baseline",
            persistence_format="resident-only",
            residency_mode="fully-resident",
            persisted_layer_count=0,
            persisted_tokens=0,
            persisted_artifact_count=0,
            hot_layer_count=0,
            hot_tokens=0,
            hot_bytes=0,
            cold_store_format=None,
        ),
    )

    summary = summarize_request_metrics([resident_metrics])
    cache = cast(dict[str, object], summary["cache"])
    cache_state_summary = cast(dict[str, object], cache["cache_state"])

    assert cache["cache_mode"] == "resident-kv"
    assert cache["kv_cache_strategy"] == "resident"
    assert cache["cache_dir_size_mb"] is None
    assert cache_state_summary["strategy_id"] == "resident"
    assert cache_state_summary["persistence_format"] == "resident-only"
    assert (
        cast(dict[str, float], cache_state_summary["persisted_artifact_count"])["mean"]
        == 0.0
    )


def test_summarize_request_metrics_reports_cpu_offload_policy() -> None:
    metrics = replace(
        build_request_probe_metrics(),
        offload_cpu_policy="middle-band",
        offload_cpu_requested_layers=4,
        offload_cpu_applied_layers=4,
        offload_cpu_applied_indices=(10, 11, 12, 13),
    )

    summary = summarize_request_metrics([metrics])
    offload = cast(dict[str, object], summary["offload"])

    assert offload["cpu_policy"] == "middle-band"
    assert cast(dict[str, float], offload["cpu_requested_layers"])["mean"] == 4.0
    assert cast(dict[str, float], offload["cpu_applied_layers"])["mean"] == 4.0
    assert offload["cpu_applied_indices"] == "10,11,12,13"


def test_render_runtime_probe_json_round_trips_sliding_window_cache_state() -> None:
    base_metrics = build_request_probe_metrics()
    cache_state = base_metrics.cache_state
    assert cache_state is not None
    probe = RuntimeProbeResult(
        load_ms=10.0,
        load_resources=build_stage_resources(),
        request=replace(
            base_metrics,
            kv_cache_strategy="sliding-window-ring-buffer",
            cache_state=replace(
                cache_state,
                strategy_id="sliding-window-ring-buffer",
                persistence_format="sliding-window-ring-buffer",
                window_policy="sliding-window",
                window_max_tokens=32,
                eviction_policy="drop-oldest",
                persisted_artifact_count=1,
                eviction_count=3,
                evicted_tokens=24,
                cold_store_format="ollm-kv-sliding-window",
            ),
        ),
    )

    payload = json.loads(render_runtime_probe_json(probe))
    request = cast(dict[str, object], payload["request"])
    cache_state_payload = cast(dict[str, object], request["cache_state"])

    assert request["kv_cache_strategy"] == "sliding-window-ring-buffer"
    assert cache_state_payload["strategy_id"] == "sliding-window-ring-buffer"
    assert cache_state_payload["window_policy"] == "sliding-window"
    assert cache_state_payload["window_max_tokens"] == 32
    assert cache_state_payload["eviction_policy"] == "drop-oldest"
    assert cache_state_payload["eviction_count"] == 3
    assert cache_state_payload["evicted_tokens"] == 24
    assert cache_state_payload["cold_store_format"] == "ollm-kv-sliding-window"
    assert cache_state_payload["resident_layer_count"] == 2
    assert cache_state_payload["compaction_count"] == 0


def test_render_runtime_probe_json_round_trips_paged_cache_state() -> None:
    base_metrics = build_request_probe_metrics()
    cache_state = base_metrics.cache_state
    assert cache_state is not None
    probe = RuntimeProbeResult(
        load_ms=10.0,
        load_resources=build_stage_resources(),
        request=replace(
            base_metrics,
            kv_cache_strategy="paged",
            cache_state=replace(
                cache_state,
                strategy_id="paged",
                persistence_format="paged-manifest",
                persisted_artifact_count=3,
                cold_store_format="ollm-kv-paged",
            ),
        ),
    )

    payload = json.loads(render_runtime_probe_json(probe))
    request = cast(dict[str, object], payload["request"])
    cache_state_payload = cast(dict[str, object], request["cache_state"])

    assert request["kv_cache_strategy"] == "paged"
    assert cache_state_payload["strategy_id"] == "paged"
    assert cache_state_payload["persistence_format"] == "paged-manifest"
    assert cache_state_payload["persisted_artifact_count"] == 3
    assert cache_state_payload["cold_store_format"] == "ollm-kv-paged"


def test_render_runtime_probe_json_round_trips_resident_cache_state() -> None:
    base_metrics = build_request_probe_metrics()
    cache_state = base_metrics.cache_state
    assert cache_state is not None
    probe = RuntimeProbeResult(
        load_ms=10.0,
        load_resources=build_stage_resources(),
        request=replace(
            base_metrics,
            cache_mode="resident-kv",
            kv_cache_strategy="resident",
            cache_dir_size_mb=None,
            native_runtime_profile=None,
            cache_state=replace(
                cache_state,
                strategy_id="resident",
                policy_id="resident-baseline",
                persistence_format="resident-only",
                residency_mode="fully-resident",
                persisted_layer_count=0,
                persisted_tokens=0,
                persisted_artifact_count=0,
                hot_layer_count=0,
                hot_tokens=0,
                hot_bytes=0,
                cold_store_format=None,
            ),
        ),
    )

    payload = json.loads(render_runtime_probe_json(probe))
    request = cast(dict[str, object], payload["request"])
    cache_state_payload = cast(dict[str, object], request["cache_state"])

    assert request["cache_mode"] == "resident-kv"
    assert request["kv_cache_strategy"] == "resident"
    assert request["cache_dir_size_mb"] is None
    assert cache_state_payload["strategy_id"] == "resident"
    assert cache_state_payload["policy_id"] == "resident-baseline"
    assert cache_state_payload["persistence_format"] == "resident-only"
    assert cache_state_payload["persisted_artifact_count"] == 0

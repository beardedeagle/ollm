import json
from dataclasses import replace
from typing import cast

from ollm.runtime.benchmark_details import summarize_request_metrics
from ollm.runtime.benchmark_probes import RuntimeProbeResult
from ollm.runtime.benchmarks import render_runtime_probe_json
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

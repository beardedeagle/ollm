"""Benchmark probe JSON rendering and parsing helpers."""

import json
from collections.abc import Mapping
from typing import cast

from ollm.kv_cache.matrix import KVCacheAdaptationSurface
from ollm.kv_cache.state import KVCacheStateSnapshot
from ollm.runtime.benchmark.probe_types import (
    EventTimingSummary,
    NativeRuntimeProfile,
    OutputScalingCase,
    OutputScalingProbeResult,
    PromptScalingCase,
    PromptScalingProbeResult,
    ReopenSessionGrowthProbeResult,
    ReopenSessionGrowthTurn,
    RequestProbeMetrics,
    RuntimeProbeResult,
    SessionGrowthProbeResult,
    SessionGrowthTurn,
    WarmRuntimeProbeResult,
)
from ollm.runtime.benchmark.resources import (
    AcceleratorUtilizationSnapshot,
    NumericSummary,
    StageResourceSnapshot,
)


def render_runtime_probe_json(probe: RuntimeProbeResult) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def render_warm_runtime_probe_json(probe: WarmRuntimeProbeResult) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def render_prompt_scaling_probe_json(probe: PromptScalingProbeResult) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def render_output_scaling_probe_json(probe: OutputScalingProbeResult) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def render_session_growth_probe_json(probe: SessionGrowthProbeResult) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def render_reopen_session_growth_probe_json(
    probe: ReopenSessionGrowthProbeResult,
) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def parse_runtime_probe_result(stdout: str) -> RuntimeProbeResult:
    payload = _load_probe_payload(stdout)
    load_resources = _parse_stage_resources(_require_mapping(payload, "load_resources"))
    request = _parse_request_probe_metrics(_require_mapping(payload, "request"))
    return RuntimeProbeResult(
        load_ms=_require_float(payload, "load_ms"),
        load_resources=load_resources,
        request=request,
    )


def parse_warm_runtime_probe_result(stdout: str) -> WarmRuntimeProbeResult:
    payload = _load_probe_payload(stdout)
    request_items = _require_sequence(payload, "measured_iterations")
    return WarmRuntimeProbeResult(
        runtime_load_ms=_require_float(payload, "runtime_load_ms"),
        runtime_load_resources=_parse_stage_resources(
            _require_mapping(payload, "runtime_load_resources")
        ),
        warmup_iterations=_require_int(payload, "warmup_iterations"),
        measured_iterations=tuple(
            _parse_request_probe_metrics(
                _require_object_mapping(item, f"measured_iterations[{index}]")
            )
            for index, item in enumerate(request_items)
        ),
    )


def parse_prompt_scaling_probe_result(stdout: str) -> PromptScalingProbeResult:
    payload = _load_probe_payload(stdout)
    case_items = _require_sequence(payload, "cases")
    return PromptScalingProbeResult(
        runtime_load_ms=_require_float(payload, "runtime_load_ms"),
        runtime_load_resources=_parse_stage_resources(
            _require_mapping(payload, "runtime_load_resources")
        ),
        cases=tuple(
            PromptScalingCase(
                requested_prompt_tokens=_require_int(
                    case_payload, "requested_prompt_tokens"
                ),
                request=_parse_request_probe_metrics(
                    _require_mapping(case_payload, "request")
                ),
            )
            for case_payload in (
                _require_object_mapping(item, f"cases[{index}]")
                for index, item in enumerate(case_items)
            )
        ),
    )


def parse_output_scaling_probe_result(stdout: str) -> OutputScalingProbeResult:
    payload = _load_probe_payload(stdout)
    case_items = _require_sequence(payload, "cases")
    return OutputScalingProbeResult(
        runtime_load_ms=_require_float(payload, "runtime_load_ms"),
        runtime_load_resources=_parse_stage_resources(
            _require_mapping(payload, "runtime_load_resources")
        ),
        cases=tuple(
            OutputScalingCase(
                requested_max_new_tokens=_require_int(
                    case_payload, "requested_max_new_tokens"
                ),
                request=_parse_request_probe_metrics(
                    _require_mapping(case_payload, "request")
                ),
            )
            for case_payload in (
                _require_object_mapping(item, f"cases[{index}]")
                for index, item in enumerate(case_items)
            )
        ),
    )


def parse_session_growth_probe_result(stdout: str) -> SessionGrowthProbeResult:
    payload = _load_probe_payload(stdout)
    turn_items = _require_sequence(payload, "turns")
    return SessionGrowthProbeResult(
        runtime_load_ms=_require_float(payload, "runtime_load_ms"),
        runtime_load_resources=_parse_stage_resources(
            _require_mapping(payload, "runtime_load_resources")
        ),
        turns=tuple(
            SessionGrowthTurn(
                turn_index=_require_int(turn_payload, "turn_index"),
                request=_parse_request_probe_metrics(
                    _require_mapping(turn_payload, "request")
                ),
            )
            for turn_payload in (
                _require_object_mapping(item, f"turns[{index}]")
                for index, item in enumerate(turn_items)
            )
        ),
    )


def parse_reopen_session_growth_probe_result(
    stdout: str,
) -> ReopenSessionGrowthProbeResult:
    payload = _load_probe_payload(stdout)
    turn_items = _require_sequence(payload, "turns")
    return ReopenSessionGrowthProbeResult(
        turns=tuple(
            ReopenSessionGrowthTurn(
                turn_index=_require_int(turn_payload, "turn_index"),
                runtime_load_ms=_require_float(turn_payload, "runtime_load_ms"),
                runtime_load_resources=_parse_stage_resources(
                    _require_mapping(turn_payload, "runtime_load_resources")
                ),
                request=_parse_request_probe_metrics(
                    _require_mapping(turn_payload, "request")
                ),
            )
            for turn_payload in (
                _require_object_mapping(item, f"turns[{index}]")
                for index, item in enumerate(turn_items)
            )
        ),
    )


def _load_probe_payload(stdout: str) -> Mapping[str, object]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("runtime probe output was not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("runtime probe output must be a JSON object")
    return cast(Mapping[str, object], payload)


def _parse_request_probe_metrics(payload: Mapping[str, object]) -> RequestProbeMetrics:
    inter_token_values = _require_sequence(payload, "inter_token_latencies_ms")
    return RequestProbeMetrics(
        total_ms=_require_float(payload, "total_ms"),
        generation_ms=_require_float(payload, "generation_ms"),
        time_to_first_token_ms=_optional_float(payload, "time_to_first_token_ms"),
        inter_token_latencies_ms=tuple(
            _require_numeric_value(value, "inter_token_latencies_ms[]")
            for value in inter_token_values
        ),
        prompt_tokens=_require_int(payload, "prompt_tokens"),
        prompt_tokens_per_second=_optional_float(payload, "prompt_tokens_per_second"),
        output_tokens=_require_int(payload, "output_tokens"),
        output_tokens_per_second=_optional_float(payload, "output_tokens_per_second"),
        cache_mode=_require_string(payload, "cache_mode"),
        kv_cache_strategy=_optional_string(payload, "kv_cache_strategy"),
        strategy_selector_profile=_optional_string(
            payload, "strategy_selector_profile"
        ),
        strategy_selector_rule_id=_optional_string(
            payload, "strategy_selector_rule_id"
        ),
        strategy_selector_requested_override=_optional_string(
            payload, "strategy_selector_requested_override"
        ),
        strategy_selector_selected_kv_cache_strategy=_optional_string(
            payload, "strategy_selector_selected_kv_cache_strategy"
        ),
        strategy_selector_applied_kv_cache_strategy=_optional_string(
            payload, "strategy_selector_applied_kv_cache_strategy"
        ),
        strategy_selector_fallback_chain=tuple(
            _require_string({"value": value}, "value")
            for value in _require_optional_sequence(
                payload, "strategy_selector_fallback_chain"
            )
        ),
        offload_cpu_policy=_optional_string(payload, "offload_cpu_policy"),
        offload_cpu_requested_layers=_optional_int(
            payload, "offload_cpu_requested_layers"
        ),
        offload_cpu_applied_layers=_optional_int(payload, "offload_cpu_applied_layers"),
        offload_cpu_applied_indices=tuple(
            _require_int({"value": value}, "value")
            for value in _require_optional_sequence(
                payload, "offload_cpu_applied_indices"
            )
        ),
        kv_cache_adaptation=_parse_kv_cache_adaptation(
            payload.get("kv_cache_adaptation")
        ),
        cache_dir_size_mb=_optional_float(payload, "cache_dir_size_mb"),
        cache_state=_parse_cache_state(payload.get("cache_state")),
        allocator_gap_mb=_optional_float(payload, "allocator_gap_mb"),
        allocator_gap_ratio=_optional_float(payload, "allocator_gap_ratio"),
        native_runtime_profile=_parse_native_runtime_profile(
            payload.get("native_runtime_profile")
        ),
        resources=_parse_stage_resources(_require_mapping(payload, "resources")),
        text_excerpt=_require_string(payload, "text_excerpt"),
    )


def _parse_stage_resources(payload: Mapping[str, object]) -> StageResourceSnapshot:
    utilization_payload = payload.get("accelerator_utilization")
    accelerator_utilization = None
    if isinstance(utilization_payload, Mapping):
        utilization_mapping = cast(Mapping[str, object], utilization_payload)
        gpu_utilization_payload = utilization_mapping.get("gpu_utilization_percent")
        memory_utilization_payload = utilization_mapping.get(
            "memory_utilization_percent"
        )
        accelerator_utilization = AcceleratorUtilizationSnapshot(
            gpu_utilization_percent=_parse_optional_summary(gpu_utilization_payload),
            memory_utilization_percent=_parse_optional_summary(
                memory_utilization_payload
            ),
        )
    return StageResourceSnapshot(
        current_rss_mb=_optional_float(payload, "current_rss_mb"),
        peak_rss_mb=_optional_float(payload, "peak_rss_mb"),
        peak_rss_source=_optional_string(payload, "peak_rss_source"),
        accelerator_kind=_optional_string(payload, "accelerator_kind"),
        accelerator_current_mb=_optional_float(payload, "accelerator_current_mb"),
        accelerator_peak_mb=_optional_float(payload, "accelerator_peak_mb"),
        accelerator_reserved_mb=_optional_float(payload, "accelerator_reserved_mb"),
        accelerator_peak_reserved_mb=_optional_float(
            payload, "accelerator_peak_reserved_mb"
        ),
        accelerator_peak_source=_optional_string(payload, "accelerator_peak_source"),
        process_cpu_utilization_percent=_optional_float(
            payload, "process_cpu_utilization_percent"
        ),
        accelerator_utilization=accelerator_utilization,
    )


def _parse_optional_summary(value: object) -> NumericSummary | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("numeric summary must be an object or null")
    payload = cast(Mapping[str, object], value)
    return NumericSummary(
        min=_require_float(payload, "min"),
        median=_require_float(payload, "median"),
        p95=_require_float(payload, "p95"),
        max=_require_float(payload, "max"),
        mean=_require_float(payload, "mean"),
    )


def _parse_native_runtime_profile(value: object) -> NativeRuntimeProfile | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("native runtime profile must be an object or null")
    payload = cast(Mapping[str, object], value)
    storage_paths = tuple(
        _require_string({"value": item}, "value")
        for item in _require_sequence(payload, "storage_paths")
    )
    raw_events = _require_mapping(payload, "events")
    events = {
        event_name: _parse_event_timing_summary(event_payload)
        for event_name, event_payload in raw_events.items()
    }
    return NativeRuntimeProfile(storage_paths=storage_paths, events=events)


def _parse_cache_state(value: object) -> KVCacheStateSnapshot | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("cache_state must be an object or null")
    payload = cast(Mapping[str, object], value)
    return KVCacheStateSnapshot(
        strategy_id=_require_string(payload, "strategy_id"),
        policy_id=_require_string(payload, "policy_id"),
        persistence_format=_require_string(payload, "persistence_format"),
        residency_mode=_require_string(payload, "residency_mode"),
        window_policy=_require_string(payload, "window_policy"),
        window_max_tokens=_optional_int(payload, "window_max_tokens"),
        eviction_policy=_optional_string(payload, "eviction_policy"),
        cold_tier_encoding=_require_string(payload, "cold_tier_encoding"),
        cold_tier_representation=_optional_string(payload, "cold_tier_representation"),
        persisted_layer_count=_require_int(payload, "persisted_layer_count"),
        persisted_tokens=_require_int(payload, "persisted_tokens"),
        persisted_artifact_count=_require_int(payload, "persisted_artifact_count"),
        resident_layer_count=_require_int(payload, "resident_layer_count"),
        resident_tokens=_require_int(payload, "resident_tokens"),
        resident_bytes=_require_int(payload, "resident_bytes"),
        hot_layer_count=_require_int(payload, "hot_layer_count"),
        hot_tokens=_require_int(payload, "hot_tokens"),
        hot_bytes=_require_int(payload, "hot_bytes"),
        compaction_count=_require_int(payload, "compaction_count"),
        spill_count=_require_int(payload, "spill_count"),
        spilled_tokens=_require_int(payload, "spilled_tokens"),
        eviction_count=_require_int(payload, "eviction_count"),
        evicted_tokens=_require_int(payload, "evicted_tokens"),
        cold_store_format=_optional_string(payload, "cold_store_format"),
    )


def _parse_kv_cache_adaptation(value: object) -> KVCacheAdaptationSurface | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("kv_cache_adaptation must be an object or null")
    payload = cast(Mapping[str, object], value)
    return KVCacheAdaptationSurface(
        adaptation_mode=_require_string(payload, "adaptation_mode"),
        recommendation_available=bool(payload.get("recommendation_available")),
        recommended_strategy_id=_optional_string(payload, "recommended_strategy_id"),
        reason=_require_string(payload, "reason"),
    )


def _parse_event_timing_summary(value: object) -> EventTimingSummary:
    if not isinstance(value, Mapping):
        raise ValueError("event timing summary must be an object")
    payload = cast(Mapping[str, object], value)
    return EventTimingSummary(
        count=_require_int(payload, "count"),
        total_ms=_require_float(payload, "total_ms"),
        min_ms=_require_float(payload, "min_ms"),
        median_ms=_require_float(payload, "median_ms"),
        p95_ms=_require_float(payload, "p95_ms"),
        max_ms=_require_float(payload, "max_ms"),
        mean_ms=_require_float(payload, "mean_ms"),
    )


def _require_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"probe field '{key}' must be an object")
    return cast(Mapping[str, object], value)


def _require_object_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"probe field '{field_name}' must be an object")
    return cast(Mapping[str, object], value)


def _require_sequence(payload: Mapping[str, object], key: str) -> tuple[object, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"probe field '{key}' must be a list")
    return tuple(value)


def _require_optional_sequence(
    payload: Mapping[str, object], key: str
) -> tuple[object, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"probe field '{key}' must be a list")
    return tuple(value)


def _require_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"probe field '{key}' must be numeric")


def _optional_float(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"probe field '{key}' must be numeric or null")


def _require_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int):
        return value
    raise ValueError(f"probe field '{key}' must be an integer")


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise ValueError(f"probe field '{key}' must be an integer or null")


def _require_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    raise ValueError(f"probe field '{key}' must be a string")


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"probe field '{key}' must be a string or null")


def _require_numeric_value(value: object, key: str) -> float:
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"probe field '{key}' must contain only numeric values")

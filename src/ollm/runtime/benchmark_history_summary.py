"""Benchmark-history summary normalization and regression comparison helpers."""

from statistics import mean
from typing import cast

_LATENCY_REGRESSION_THRESHOLD = 0.05
_THROUGHPUT_REGRESSION_THRESHOLD = 0.05
_MEMORY_REGRESSION_THRESHOLD = 0.05
_HIGHER_IS_WORSE_METRICS = {
    "load_ms",
    "request_total_ms",
    "ttft_ms",
    "cache_dir_size_mb",
    "allocator_gap_mb",
    "peak_rss_mb",
    "accelerator_peak_mb",
    "accelerator_peak_reserved_mb",
    "kvload_ms",
    "kvsave_ms",
    "kvcompact_ms",
    "persisted_tokens",
    "persisted_artifact_count",
    "hot_tokens",
    "compaction_count",
    "spill_count",
    "spilled_tokens",
}
_HIGHER_IS_BETTER_METRICS = {
    "prompt_tokens_per_second",
    "output_tokens_per_second",
}


def summarize_benchmark_payload(
    payload: dict[str, object], *, run_kind: str
) -> dict[str, object]:
    """Normalize a probe or report payload into comparable summary metrics."""

    if run_kind == "probe-cold":
        request = require_object(payload.get("request"), "request")
        resources = require_object(request.get("resources"), "resources")
        cache_state = optional_object(request.get("cache_state"))
        return {
            "load_ms": optional_float(payload.get("load_ms")),
            "request_total_ms": optional_float(request.get("total_ms")),
            "ttft_ms": optional_float(request.get("time_to_first_token_ms")),
            "prompt_tokens_per_second": optional_float(
                request.get("prompt_tokens_per_second")
            ),
            "output_tokens_per_second": optional_float(
                request.get("output_tokens_per_second")
            ),
            "cache_dir_size_mb": optional_float(request.get("cache_dir_size_mb")),
            "allocator_gap_mb": optional_float(request.get("allocator_gap_mb")),
            "peak_rss_mb": optional_float(resources.get("peak_rss_mb")),
            "accelerator_peak_mb": optional_float(resources.get("accelerator_peak_mb")),
            "accelerator_peak_reserved_mb": optional_float(
                resources.get("accelerator_peak_reserved_mb")
            ),
            "kvload_ms": event_total_ms(request, "kvload"),
            "kvsave_ms": event_total_ms(request, "kvsave"),
            "kvcompact_ms": event_total_ms(request, "kvcompact"),
            "persisted_tokens": optional_int(
                None if cache_state is None else cache_state.get("persisted_tokens")
            ),
            "persisted_artifact_count": optional_int(
                None
                if cache_state is None
                else cache_state.get("persisted_artifact_count")
            ),
            "hot_tokens": optional_int(
                None if cache_state is None else cache_state.get("hot_tokens")
            ),
            "compaction_count": optional_int(
                None if cache_state is None else cache_state.get("compaction_count")
            ),
            "cold_store_format": (
                None if cache_state is None else cache_state.get("cold_store_format")
            ),
            "spill_count": optional_int(
                None if cache_state is None else cache_state.get("spill_count")
            ),
            "spilled_tokens": optional_int(
                None if cache_state is None else cache_state.get("spilled_tokens")
            ),
        }
    if run_kind == "probe-prompt-scaling":
        cases = require_probe_case_requests(payload, "cases")
        final_request = cases[-1]
        return {
            "load_ms": optional_float(payload.get("runtime_load_ms")),
            "request_total_ms": optional_float(final_request.get("total_ms")),
            "max_request_total_ms": max_optional_float(
                request.get("total_ms") for request in cases
            ),
            "ttft_ms": optional_float(final_request.get("time_to_first_token_ms")),
            "prompt_tokens_per_second": optional_float(
                final_request.get("prompt_tokens_per_second")
            ),
            "output_tokens_per_second": optional_float(
                final_request.get("output_tokens_per_second")
            ),
            "cache_dir_size_mb": optional_float(final_request.get("cache_dir_size_mb")),
            "allocator_gap_mb": optional_float(final_request.get("allocator_gap_mb")),
            "peak_rss_mb": optional_float(
                resource_metric(final_request, "peak_rss_mb")
            ),
            "accelerator_peak_mb": optional_float(
                resource_metric(final_request, "accelerator_peak_mb")
            ),
            "accelerator_peak_reserved_mb": optional_float(
                resource_metric(final_request, "accelerator_peak_reserved_mb")
            ),
            "kvload_ms": event_total_ms(final_request, "kvload"),
            "kvsave_ms": event_total_ms(final_request, "kvsave"),
            "kvcompact_ms": event_total_ms(final_request, "kvcompact"),
            "case_count": len(cases),
        }
    if run_kind == "probe-output-scaling":
        cases = require_probe_case_requests(payload, "cases")
        final_request = cases[-1]
        return {
            "load_ms": optional_float(payload.get("runtime_load_ms")),
            "request_total_ms": optional_float(final_request.get("total_ms")),
            "max_request_total_ms": max_optional_float(
                request.get("total_ms") for request in cases
            ),
            "ttft_ms": optional_float(final_request.get("time_to_first_token_ms")),
            "prompt_tokens_per_second": optional_float(
                final_request.get("prompt_tokens_per_second")
            ),
            "output_tokens_per_second": optional_float(
                final_request.get("output_tokens_per_second")
            ),
            "cache_dir_size_mb": optional_float(final_request.get("cache_dir_size_mb")),
            "allocator_gap_mb": optional_float(final_request.get("allocator_gap_mb")),
            "peak_rss_mb": optional_float(
                resource_metric(final_request, "peak_rss_mb")
            ),
            "accelerator_peak_mb": optional_float(
                resource_metric(final_request, "accelerator_peak_mb")
            ),
            "accelerator_peak_reserved_mb": optional_float(
                resource_metric(final_request, "accelerator_peak_reserved_mb")
            ),
            "kvload_ms": event_total_ms(final_request, "kvload"),
            "kvsave_ms": event_total_ms(final_request, "kvsave"),
            "kvcompact_ms": event_total_ms(final_request, "kvcompact"),
            "case_count": len(cases),
        }
    if run_kind == "probe-session-growth":
        turns = require_probe_case_requests(payload, "turns")
        final_request = turns[-1]
        cache_state = optional_object(final_request.get("cache_state"))
        return {
            "load_ms": optional_float(payload.get("runtime_load_ms")),
            "request_total_ms": optional_float(final_request.get("total_ms")),
            "mean_request_total_ms": mean_optional_float(
                request.get("total_ms") for request in turns
            ),
            "ttft_ms": optional_float(final_request.get("time_to_first_token_ms")),
            "prompt_tokens_per_second": optional_float(
                final_request.get("prompt_tokens_per_second")
            ),
            "output_tokens_per_second": optional_float(
                final_request.get("output_tokens_per_second")
            ),
            "cache_dir_size_mb": optional_float(final_request.get("cache_dir_size_mb")),
            "allocator_gap_mb": optional_float(final_request.get("allocator_gap_mb")),
            "peak_rss_mb": optional_float(
                resource_metric(final_request, "peak_rss_mb")
            ),
            "accelerator_peak_mb": optional_float(
                resource_metric(final_request, "accelerator_peak_mb")
            ),
            "accelerator_peak_reserved_mb": optional_float(
                resource_metric(final_request, "accelerator_peak_reserved_mb")
            ),
            "kvload_ms": event_total_ms(final_request, "kvload"),
            "kvsave_ms": event_total_ms(final_request, "kvsave"),
            "kvcompact_ms": event_total_ms(final_request, "kvcompact"),
            "persisted_tokens": optional_int(
                None if cache_state is None else cache_state.get("persisted_tokens")
            ),
            "persisted_artifact_count": optional_int(
                None
                if cache_state is None
                else cache_state.get("persisted_artifact_count")
            ),
            "hot_tokens": optional_int(
                None if cache_state is None else cache_state.get("hot_tokens")
            ),
            "compaction_count": optional_int(
                None if cache_state is None else cache_state.get("compaction_count")
            ),
            "cold_store_format": (
                None if cache_state is None else cache_state.get("cold_store_format")
            ),
            "spill_count": optional_int(
                None if cache_state is None else cache_state.get("spill_count")
            ),
            "spilled_tokens": optional_int(
                None if cache_state is None else cache_state.get("spilled_tokens")
            ),
            "session_turns": len(turns),
        }
    if run_kind == "probe-warm":
        measured_iterations = payload.get("measured_iterations")
        if not isinstance(measured_iterations, list):
            raise ValueError("Warm probe payload must contain measured_iterations")
        requests = tuple(
            require_object(item, "measured_iterations[]")
            for item in measured_iterations
        )
        cache_states = tuple(
            optional_object(request.get("cache_state")) for request in requests
        )
        return {
            "load_ms": optional_float(payload.get("runtime_load_ms")),
            "request_total_ms": mean_optional_float(
                request.get("total_ms") for request in requests
            ),
            "ttft_ms": mean_optional_float(
                request.get("time_to_first_token_ms") for request in requests
            ),
            "prompt_tokens_per_second": mean_optional_float(
                request.get("prompt_tokens_per_second") for request in requests
            ),
            "output_tokens_per_second": mean_optional_float(
                request.get("output_tokens_per_second") for request in requests
            ),
            "cache_dir_size_mb": mean_optional_float(
                request.get("cache_dir_size_mb") for request in requests
            ),
            "allocator_gap_mb": mean_optional_float(
                request.get("allocator_gap_mb") for request in requests
            ),
            "peak_rss_mb": mean_optional_float(
                resource_metric(request, "peak_rss_mb") for request in requests
            ),
            "accelerator_peak_mb": mean_optional_float(
                resource_metric(request, "accelerator_peak_mb") for request in requests
            ),
            "accelerator_peak_reserved_mb": mean_optional_float(
                resource_metric(request, "accelerator_peak_reserved_mb")
                for request in requests
            ),
            "kvload_ms": mean_optional_float(
                event_total_ms(request, "kvload") for request in requests
            ),
            "kvsave_ms": mean_optional_float(
                event_total_ms(request, "kvsave") for request in requests
            ),
            "kvcompact_ms": mean_optional_float(
                event_total_ms(request, "kvcompact") for request in requests
            ),
            "persisted_tokens": mean_optional_int(
                None if cache_state is None else cache_state.get("persisted_tokens")
                for cache_state in cache_states
            ),
            "persisted_artifact_count": mean_optional_int(
                None
                if cache_state is None
                else cache_state.get("persisted_artifact_count")
                for cache_state in cache_states
            ),
            "hot_tokens": mean_optional_int(
                None if cache_state is None else cache_state.get("hot_tokens")
                for cache_state in cache_states
            ),
            "compaction_count": mean_optional_int(
                None if cache_state is None else cache_state.get("compaction_count")
                for cache_state in cache_states
            ),
            "cold_store_format": single_optional_string(
                None if cache_state is None else cache_state.get("cold_store_format")
                for cache_state in cache_states
            ),
            "spill_count": mean_optional_int(
                None if cache_state is None else cache_state.get("spill_count")
                for cache_state in cache_states
            ),
            "spilled_tokens": mean_optional_int(
                None if cache_state is None else cache_state.get("spilled_tokens")
                for cache_state in cache_states
            ),
        }
    if run_kind == "report":
        runtime_comparison = require_object(
            payload.get("runtime_comparison"), "runtime_comparison"
        )
        primary_target = require_object(
            runtime_comparison.get("primary_target"), "primary_target"
        )
        cold_start = require_object(primary_target.get("cold_start"), "cold_start")
        warm_runtime = require_object(
            primary_target.get("warm_runtime"), "warm_runtime"
        )
        optimized_cold = require_object(
            cold_start.get("optimized_native"), "optimized_native"
        )
        optimized_warm = require_object(
            warm_runtime.get("optimized_native"), "optimized_native"
        )
        optimized_cold_details = require_object(
            optimized_cold.get("details"), "details"
        )
        optimized_warm_details = require_object(
            optimized_warm.get("details"), "details"
        )
        return {
            "load_ms": optional_float(optimized_cold_details.get("load_ms")),
            "request_total_ms": optional_float(
                optimized_cold_details.get("request_total_ms")
            ),
            "ttft_ms": optional_float(
                optimized_cold_details.get("request_time_to_first_token_ms")
            ),
            "prompt_tokens_per_second": optional_float(
                optimized_cold_details.get("request_prompt_tokens_per_second")
            ),
            "output_tokens_per_second": optional_float(
                optimized_cold_details.get("request_output_tokens_per_second")
            ),
            "cache_dir_size_mb": optional_float(
                optimized_cold_details.get("request_cache_dir_size_mb")
            ),
            "allocator_gap_mb": optional_float(
                optimized_cold_details.get("request_allocator_gap_mb")
            ),
            "peak_rss_mb": optional_float(
                optimized_cold_details.get("request_peak_rss_mb")
            ),
            "accelerator_peak_mb": optional_float(
                optimized_cold_details.get("request_accelerator_peak_mb")
            ),
            "accelerator_peak_reserved_mb": optional_float(
                optimized_cold_details.get("request_accelerator_peak_reserved_mb")
            ),
            "kvload_ms": optional_float(
                optimized_cold_details.get("request_native_kvload_ms")
            ),
            "kvsave_ms": optional_float(
                optimized_cold_details.get("request_native_kvsave_ms")
            ),
            "warm_request_total_ms": optional_float(
                optimized_warm_details.get("request_total_ms_mean")
            ),
        }
    raise ValueError(f"Unsupported benchmark history run kind: {run_kind}")


def compare_metric_summaries(
    *,
    current: dict[str, object],
    previous: dict[str, object],
) -> dict[str, object]:
    """Compare normalized summaries and flag obvious regressions."""

    deltas: dict[str, dict[str, float | None]] = {}
    regressions: list[str] = []
    for metric_name, current_value in current.items():
        previous_value = previous.get(metric_name)
        if not isinstance(current_value, (int, float)) or not isinstance(
            previous_value, (int, float)
        ):
            continue
        current_float = float(current_value)
        previous_float = float(previous_value)
        absolute_delta = current_float - previous_float
        percent_delta = None
        if previous_float != 0:
            percent_delta = (absolute_delta / previous_float) * 100.0
        deltas[metric_name] = {
            "current": current_float,
            "previous": previous_float,
            "absolute_delta": absolute_delta,
            "percent_delta": percent_delta,
        }
        if is_regression(
            metric_name=metric_name,
            current=current_float,
            previous=previous_float,
        ):
            regressions.append(metric_name)
    return {
        "metrics": deltas,
        "potential_regressions": sorted(regressions),
    }


def event_total_ms(request: dict[str, object], event_name: str) -> float | None:
    native_runtime_profile = optional_object(request.get("native_runtime_profile"))
    if native_runtime_profile is None:
        return None
    events = optional_object(native_runtime_profile.get("events"))
    if events is None:
        return None
    event_payload = optional_object(events.get(event_name))
    if event_payload is None:
        return None
    return optional_float(event_payload.get("total_ms"))


def resource_metric(request: dict[str, object], field_name: str) -> object:
    resources = optional_object(request.get("resources"))
    if resources is None:
        return None
    return resources.get(field_name)


def require_probe_case_requests(
    payload: dict[str, object],
    field_name: str,
) -> tuple[dict[str, object], ...]:
    cases = payload.get(field_name)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{field_name} must contain at least one case")
    requests: list[dict[str, object]] = []
    for case in cases:
        case_payload = require_object(case, field_name)
        requests.append(require_object(case_payload.get("request"), "request"))
    return tuple(requests)


def require_object(payload: object, field_name: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be an object")
    return cast(dict[str, object], payload)


def optional_object(payload: object) -> dict[str, object] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("Expected object or null")
    return cast(dict[str, object], payload)


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected numeric value or null")
    return float(value)


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected integer value or null")
    return value


def mean_optional_float(values) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return mean(numeric_values)


def mean_optional_int(values) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return mean(numeric_values)


def max_optional_float(values) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return max(numeric_values)


def single_optional_string(values) -> str | None:
    string_values = [value for value in values if isinstance(value, str) and value]
    if not string_values:
        return None
    return string_values[-1]


def is_regression(*, metric_name: str, current: float, previous: float) -> bool:
    if metric_name in _HIGHER_IS_WORSE_METRICS:
        if previous == 0:
            return current > 0
        threshold = (
            _MEMORY_REGRESSION_THRESHOLD
            if "accelerator" in metric_name
            or metric_name in {"peak_rss_mb", "cache_dir_size_mb"}
            else _LATENCY_REGRESSION_THRESHOLD
        )
        return current > previous * (1.0 + threshold)
    if metric_name in _HIGHER_IS_BETTER_METRICS:
        if previous == 0:
            return False
        return current < previous * (1.0 - _THROUGHPUT_REGRESSION_THRESHOLD)
    return False

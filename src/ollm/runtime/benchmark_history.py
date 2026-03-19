"""Persistent benchmark history and regression comparison helpers."""

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import cast

from ollm.async_io import path_mkdir, path_write_text
from ollm.runtime.benchmark_metadata import (
    build_git_summary,
    build_history_codebase_summary,
    build_history_host_summary,
)

_HISTORY_DIR = Path(".omx/logs/benchmark-history")
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
    "persisted_tokens",
    "hot_tokens",
    "spill_count",
    "spilled_tokens",
}
_HIGHER_IS_BETTER_METRICS = {
    "prompt_tokens_per_second",
    "output_tokens_per_second",
}


def record_benchmark_history(
    *,
    repo_root: Path,
    payload: dict[str, object],
    run_kind: str,
    history_dir: Path | None,
    comparison_key: dict[str, object],
    codebase_label: str,
) -> dict[str, object]:
    """Persist one benchmark payload and compare it with the last matching run."""

    resolved_history_dir = (
        (repo_root / _HISTORY_DIR).resolve()
        if history_dir is None
        else history_dir.expanduser().resolve()
    )
    records_dir = resolved_history_dir / "records"
    index_path = resolved_history_dir / "index.jsonl"
    path_mkdir(records_dir, parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    previous = find_previous_record(index_path, comparison_key=comparison_key)
    summary = summarize_benchmark_payload(payload, run_kind=run_kind)
    comparison = (
        None
        if previous is None
        else compare_metric_summaries(
            current=summary,
            previous=_require_object(previous.get("summary"), "summary"),
        )
    )
    record = {
        "generated_at": generated_at,
        "run_kind": run_kind,
        "comparison_key": comparison_key,
        "codebase": build_history_codebase_summary(
            repo_root, codebase_label=codebase_label
        ),
        "host": build_history_host_summary(),
        "git": build_git_summary(repo_root),
        "summary": summary,
        "comparison_to_previous": comparison,
        "payload": payload,
    }
    record_name = (
        f"{generated_at.replace(':', '').replace('+00:00', 'Z')}-{run_kind}.json"
    )
    record_path = records_dir / record_name
    rendered_record = json.dumps(record, indent=2, sort_keys=True) + "\n"
    path_write_text(record_path, rendered_record, encoding="utf-8")
    index_entry: dict[str, object] = {
        "generated_at": generated_at,
        "record_path": str(record_path),
        "run_kind": run_kind,
        "comparison_key": comparison_key,
        "codebase": record["codebase"],
        "summary": summary,
        "comparison_to_previous": comparison,
        "host": record["host"],
        "git": record["git"],
    }
    _append_jsonl_entry(index_path, index_entry)
    return {
        "record_path": str(record_path),
        "codebase_label": codebase_label,
        "comparison_to_previous": comparison,
        "summary": summary,
    }


def find_previous_record(
    index_path: Path, *, comparison_key: dict[str, object]
) -> dict[str, object] | None:
    """Return the last matching benchmark history entry, if any."""

    if not index_path.exists():
        return None
    for line in reversed(index_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            continue
        if entry.get("comparison_key") == comparison_key:
            return entry
    return None


def summarize_benchmark_payload(
    payload: dict[str, object], *, run_kind: str
) -> dict[str, object]:
    """Normalize a probe or report payload into comparable summary metrics."""

    if run_kind == "probe-cold":
        request = _require_object(payload.get("request"), "request")
        resources = _require_object(request.get("resources"), "resources")
        cache_state = _optional_object(request.get("cache_state"))
        return {
            "load_ms": _optional_float(payload.get("load_ms")),
            "request_total_ms": _optional_float(request.get("total_ms")),
            "ttft_ms": _optional_float(request.get("time_to_first_token_ms")),
            "prompt_tokens_per_second": _optional_float(
                request.get("prompt_tokens_per_second")
            ),
            "output_tokens_per_second": _optional_float(
                request.get("output_tokens_per_second")
            ),
            "cache_dir_size_mb": _optional_float(request.get("cache_dir_size_mb")),
            "allocator_gap_mb": _optional_float(request.get("allocator_gap_mb")),
            "peak_rss_mb": _optional_float(resources.get("peak_rss_mb")),
            "accelerator_peak_mb": _optional_float(
                resources.get("accelerator_peak_mb")
            ),
            "accelerator_peak_reserved_mb": _optional_float(
                resources.get("accelerator_peak_reserved_mb")
            ),
            "kvload_ms": _event_total_ms(request, "kvload"),
            "kvsave_ms": _event_total_ms(request, "kvsave"),
            "persisted_tokens": _optional_int(
                None if cache_state is None else cache_state.get("persisted_tokens")
            ),
            "hot_tokens": _optional_int(
                None if cache_state is None else cache_state.get("hot_tokens")
            ),
            "spill_count": _optional_int(
                None if cache_state is None else cache_state.get("spill_count")
            ),
            "spilled_tokens": _optional_int(
                None if cache_state is None else cache_state.get("spilled_tokens")
            ),
        }
    if run_kind == "probe-warm":
        measured_iterations = payload.get("measured_iterations")
        if not isinstance(measured_iterations, list):
            raise ValueError("Warm probe payload must contain measured_iterations")
        requests = tuple(
            _require_object(item, "measured_iterations[]")
            for item in measured_iterations
        )
        cache_states = tuple(
            _optional_object(request.get("cache_state")) for request in requests
        )
        return {
            "load_ms": _optional_float(payload.get("runtime_load_ms")),
            "request_total_ms": _mean_optional_float(
                request.get("total_ms") for request in requests
            ),
            "ttft_ms": _mean_optional_float(
                request.get("time_to_first_token_ms") for request in requests
            ),
            "prompt_tokens_per_second": _mean_optional_float(
                request.get("prompt_tokens_per_second") for request in requests
            ),
            "output_tokens_per_second": _mean_optional_float(
                request.get("output_tokens_per_second") for request in requests
            ),
            "cache_dir_size_mb": _mean_optional_float(
                request.get("cache_dir_size_mb") for request in requests
            ),
            "allocator_gap_mb": _mean_optional_float(
                request.get("allocator_gap_mb") for request in requests
            ),
            "peak_rss_mb": _mean_optional_float(
                _resource_metric(request, "peak_rss_mb") for request in requests
            ),
            "accelerator_peak_mb": _mean_optional_float(
                _resource_metric(request, "accelerator_peak_mb") for request in requests
            ),
            "accelerator_peak_reserved_mb": _mean_optional_float(
                _resource_metric(request, "accelerator_peak_reserved_mb")
                for request in requests
            ),
            "kvload_ms": _mean_optional_float(
                _event_total_ms(request, "kvload") for request in requests
            ),
            "kvsave_ms": _mean_optional_float(
                _event_total_ms(request, "kvsave") for request in requests
            ),
            "persisted_tokens": _mean_optional_int(
                None if cache_state is None else cache_state.get("persisted_tokens")
                for cache_state in cache_states
            ),
            "hot_tokens": _mean_optional_int(
                None if cache_state is None else cache_state.get("hot_tokens")
                for cache_state in cache_states
            ),
            "spill_count": _mean_optional_int(
                None if cache_state is None else cache_state.get("spill_count")
                for cache_state in cache_states
            ),
            "spilled_tokens": _mean_optional_int(
                None if cache_state is None else cache_state.get("spilled_tokens")
                for cache_state in cache_states
            ),
        }
    if run_kind == "report":
        runtime_comparison = _require_object(
            payload.get("runtime_comparison"), "runtime_comparison"
        )
        primary_target = _require_object(
            runtime_comparison.get("primary_target"), "primary_target"
        )
        cold_start = _require_object(primary_target.get("cold_start"), "cold_start")
        warm_runtime = _require_object(
            primary_target.get("warm_runtime"), "warm_runtime"
        )
        optimized_cold = _require_object(
            cold_start.get("optimized_native"), "optimized_native"
        )
        optimized_warm = _require_object(
            warm_runtime.get("optimized_native"), "optimized_native"
        )
        optimized_cold_details = _require_object(
            optimized_cold.get("details"), "details"
        )
        optimized_warm_details = _require_object(
            optimized_warm.get("details"), "details"
        )
        return {
            "load_ms": _optional_float(optimized_cold_details.get("load_ms")),
            "request_total_ms": _optional_float(
                optimized_cold_details.get("request_total_ms")
            ),
            "ttft_ms": _optional_float(
                optimized_cold_details.get("request_time_to_first_token_ms")
            ),
            "prompt_tokens_per_second": _optional_float(
                optimized_cold_details.get("request_prompt_tokens_per_second")
            ),
            "output_tokens_per_second": _optional_float(
                optimized_cold_details.get("request_output_tokens_per_second")
            ),
            "cache_dir_size_mb": _optional_float(
                optimized_cold_details.get("request_cache_dir_size_mb")
            ),
            "allocator_gap_mb": _optional_float(
                optimized_cold_details.get("request_allocator_gap_mb")
            ),
            "peak_rss_mb": _optional_float(
                optimized_cold_details.get("request_peak_rss_mb")
            ),
            "accelerator_peak_mb": _optional_float(
                optimized_cold_details.get("request_accelerator_peak_mb")
            ),
            "accelerator_peak_reserved_mb": _optional_float(
                optimized_cold_details.get("request_accelerator_peak_reserved_mb")
            ),
            "kvload_ms": _optional_float(
                optimized_cold_details.get("request_native_kvload_ms")
            ),
            "kvsave_ms": _optional_float(
                optimized_cold_details.get("request_native_kvsave_ms")
            ),
            "warm_request_total_ms": _optional_float(
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
        if _is_regression(
            metric_name=metric_name,
            current=current_float,
            previous=previous_float,
        ):
            regressions.append(metric_name)
    return {
        "metrics": deltas,
        "potential_regressions": sorted(regressions),
    }


def _append_jsonl_entry(path: Path, payload: dict[str, object]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    rendered_payload = json.dumps(payload, sort_keys=True)
    content = (
        rendered_payload
        if not existing
        else existing.rstrip("\n") + "\n" + rendered_payload
    )
    path_write_text(path, content + "\n", encoding="utf-8")


def _event_total_ms(request: dict[str, object], event_name: str) -> float | None:
    native_runtime_profile = _optional_object(request.get("native_runtime_profile"))
    if native_runtime_profile is None:
        return None
    events = _optional_object(native_runtime_profile.get("events"))
    if events is None:
        return None
    event_payload = _optional_object(events.get(event_name))
    if event_payload is None:
        return None
    return _optional_float(event_payload.get("total_ms"))


def _resource_metric(request: dict[str, object], field_name: str) -> object:
    resources = _optional_object(request.get("resources"))
    if resources is None:
        return None
    return resources.get(field_name)


def _require_object(payload: object, field_name: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be an object")
    return cast(dict[str, object], payload)


def _optional_object(payload: object) -> dict[str, object] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("Expected object or null")
    return cast(dict[str, object], payload)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected numeric value or null")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected integer value or null")
    return value


def _mean_optional_float(values) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return mean(numeric_values)


def _mean_optional_int(values) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return mean(numeric_values)


def _is_regression(*, metric_name: str, current: float, previous: float) -> bool:
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

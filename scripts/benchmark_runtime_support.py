import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from ollm.client import RuntimeClient
from ollm.runtime.benchmark.probe_registry import ProbeMode, get_probe_definition
from ollm.runtime.config import RuntimeConfig


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_positive_int_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        raise SystemExit("expected a comma-separated list of positive integers")
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit(
            "expected a comma-separated list of positive integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise SystemExit("expected a comma-separated list of positive integers")
    return values


def emit_history_status(history_result: dict[str, object]) -> None:
    record_path = history_result.get("record_path")
    if isinstance(record_path, str):
        codebase = history_result.get("codebase_label")
        if isinstance(codebase, str):
            print(
                f"benchmark history recorded [{codebase}]: {record_path}",
                file=sys.stderr,
            )
        else:
            print(f"benchmark history recorded: {record_path}", file=sys.stderr)
    comparison = history_result.get("comparison_to_previous")
    if not isinstance(comparison, dict):
        return
    comparison_payload = cast(Mapping[str, object], comparison)
    comparison_available = comparison_payload.get("comparison_available")
    if comparison_available is False:
        reason = comparison_payload.get("reason")
        if isinstance(reason, str):
            print(f"benchmark comparison unavailable: {reason}", file=sys.stderr)
    regressions = comparison_payload.get("potential_regressions")
    if not isinstance(regressions, list) or not regressions:
        return
    print(
        "potential benchmark regressions: "
        + ", ".join(str(item) for item in regressions),
        file=sys.stderr,
    )


def extract_probe_selector_result(
    payload: Mapping[str, object],
    *,
    probe_mode: ProbeMode,
) -> tuple[str | None, str | None]:
    request_payload = get_probe_definition(probe_mode).history_request_extractor(payload)
    if request_payload is None:
        return None, None
    return (
        _optional_string(request_payload, "strategy_selector_rule_id"),
        _optional_string(
            request_payload, "strategy_selector_applied_kv_cache_strategy"
        ),
    )


def _optional_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, object], value)


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        return None
    return value


def resolve_report_selector_result(
    *,
    model_reference: str,
    models_dir: Path,
    device: str,
    kv_cache_strategy: str | None,
    strategy_selector_profile: str,
    kv_cache_window_tokens: int | None,
    offload_cpu_layers: int,
    offload_cpu_policy: str,
    offload_gpu_layers: int,
) -> tuple[str | None, str | None]:
    selector_plan = RuntimeClient().plan(
        RuntimeConfig(
            model_reference=model_reference,
            models_dir=models_dir,
            device=device,
            kv_cache_strategy=kv_cache_strategy,
            strategy_selector_profile=strategy_selector_profile,
            kv_cache_window_tokens=kv_cache_window_tokens,
            offload_cpu_layers=offload_cpu_layers,
            offload_cpu_policy=offload_cpu_policy,
            offload_gpu_layers=offload_gpu_layers,
        )
    )
    return (
        selector_plan.details.get("strategy_selector_rule_id"),
        selector_plan.details.get("strategy_selector_applied_kv_cache_strategy"),
    )

"""Support helpers for benchmark-history summary normalization."""

from statistics import mean
from typing import cast


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

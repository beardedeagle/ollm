"""Apply selector results to runtime config and plan details."""

from dataclasses import replace

from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.strategy_selector import RuntimeStrategySelection


def apply_strategy_selection_to_config(
    config: RuntimeConfig,
    strategy_selection: RuntimeStrategySelection,
) -> RuntimeConfig:
    """Build the effective runtime config after selector choice."""

    applied_lifecycle = config.resolved_kv_cache_lifecycle()
    if strategy_selection.applied_kv_cache_strategy == "resident":
        applied_lifecycle = "runtime-scoped"
    effective_config = replace(
        config,
        kv_cache_strategy=strategy_selection.applied_kv_cache_strategy,
        kv_cache_lifecycle=applied_lifecycle,
    )
    effective_config.validate()
    return effective_config


def plan_with_strategy_selection(
    runtime_plan: RuntimePlan,
    strategy_selection: RuntimeStrategySelection,
    *,
    requested_config: RuntimeConfig,
    effective_config: RuntimeConfig,
) -> RuntimePlan:
    """Merge selector details into a runtime plan."""

    details = dict(runtime_plan.details)
    for key, value in strategy_selection.to_details().items():
        details[key] = value
    details["strategy_selector_requested_kv_cache_lifecycle"] = (
        requested_config.resolved_kv_cache_lifecycle()
    )
    details["strategy_selector_applied_kv_cache_lifecycle"] = (
        effective_config.resolved_kv_cache_lifecycle()
    )
    if (
        requested_config.resolved_kv_cache_lifecycle()
        != effective_config.resolved_kv_cache_lifecycle()
    ):
        details["strategy_selector_lifecycle_reason"] = (
            "Resident fallback requires runtime-scoped lifecycle."
        )
    return replace(runtime_plan, details=details)

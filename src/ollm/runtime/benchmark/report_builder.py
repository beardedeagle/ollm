"""Runtime benchmark report orchestration."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from ollm.client import RuntimeClient
from ollm.runtime.benchmark.commands import measure_callable
from ollm.runtime.benchmark.details import mean_delta_ms
from ollm.runtime.benchmark.fixtures import create_tiny_t5_fixture
from ollm.runtime.benchmark.host import build_host_summary
from ollm.runtime.benchmark.targets import (
    benchmark_runtime_target,
    build_current_supported_family_targets,
)
from ollm.runtime.benchmark.types import (
    DEFAULT_OUTPUT_TOKEN_TARGETS,
    DEFAULT_PROMPT_TOKEN_TARGETS,
    DEFAULT_SESSION_MAX_NEW_TOKENS,
    DEFAULT_SESSION_TURNS,
    RuntimeBenchmarkReport,
    RuntimeComparisonTarget,
)
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.strategy_selector import DEFAULT_STRATEGY_SELECTOR_PROFILE


def build_runtime_benchmark_report(
    *,
    repo_root: Path,
    benchmark_model_reference: str,
    models_dir: Path,
    device: str,
    kv_cache_strategy: str | None = None,
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE,
    kv_cache_window_tokens: int | None = None,
    offload_cpu_layers: int = 0,
    offload_cpu_policy: str = "auto",
    offload_gpu_layers: int = 0,
    iterations: int = 5,
    warmup_iterations: int = 1,
    prompt_token_targets: tuple[int, ...] = DEFAULT_PROMPT_TOKEN_TARGETS,
    output_token_targets: tuple[int, ...] = DEFAULT_OUTPUT_TOKEN_TARGETS,
    session_turns: int = DEFAULT_SESSION_TURNS,
    session_max_new_tokens: int = DEFAULT_SESSION_MAX_NEW_TOKENS,
    include_family_results: bool = True,
    include_primary_extended_scenarios: bool = True,
    cold_timeout_seconds: float = 240.0,
    warm_timeout_seconds: float = 240.0,
    scaling_timeout_seconds: float = 300.0,
    session_timeout_seconds: float = 300.0,
) -> RuntimeBenchmarkReport:
    """Build the full runtime benchmark report for one primary model reference."""

    client = RuntimeClient()
    models_root = models_dir.expanduser().resolve()
    planner_enabled_config = RuntimeConfig(
        model_reference=benchmark_model_reference,
        models_dir=models_root,
        device=device,
    )
    planner_disabled_config = RuntimeConfig(
        model_reference=benchmark_model_reference,
        models_dir=models_root,
        device=device,
        backend="transformers-generic",
        use_specialization=False,
    )
    planner_enabled = measure_callable(
        "planner-specialization-enabled",
        lambda: client.plan(planner_enabled_config),
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        details={
            "model_reference": benchmark_model_reference,
            "device": device,
            "use_specialization": True,
        },
    )
    planner_disabled = measure_callable(
        "planner-specialization-disabled",
        lambda: client.plan(planner_disabled_config),
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        details={
            "model_reference": benchmark_model_reference,
            "device": device,
            "backend": "transformers-generic",
            "use_specialization": False,
        },
    )
    fallback_measurements = measure_no_specialization_fallback_cost(
        device=device,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )
    runtime_comparison = _measure_runtime_comparison(
        repo_root=repo_root,
        benchmark_model_reference=benchmark_model_reference,
        models_dir=models_root,
        device=device,
        kv_cache_strategy=kv_cache_strategy,
        strategy_selector_profile=strategy_selector_profile,
        kv_cache_window_tokens=kv_cache_window_tokens,
        offload_cpu_layers=offload_cpu_layers,
        offload_cpu_policy=offload_cpu_policy,
        offload_gpu_layers=offload_gpu_layers,
        iterations=max(1, min(iterations, 3)),
        warmup_iterations=min(warmup_iterations, 1),
        prompt_token_targets=prompt_token_targets,
        output_token_targets=output_token_targets,
        session_turns=session_turns,
        session_max_new_tokens=session_max_new_tokens,
        include_family_results=include_family_results,
        include_primary_extended_scenarios=include_primary_extended_scenarios,
        cold_timeout_seconds=cold_timeout_seconds,
        warm_timeout_seconds=warm_timeout_seconds,
        scaling_timeout_seconds=scaling_timeout_seconds,
        session_timeout_seconds=session_timeout_seconds,
    )
    return RuntimeBenchmarkReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        host=build_host_summary(),
        benchmark_model_reference=benchmark_model_reference,
        device=device,
        specialization_planner_overhead={
            "specialization_enabled": planner_enabled.to_dict(),
            "specialization_disabled": planner_disabled.to_dict(),
            "mean_delta_ms": mean_delta_ms(planner_enabled, planner_disabled),
        },
        fallback_cost_when_no_specialization_applies=fallback_measurements,
        runtime_comparison=runtime_comparison,
    )


def measure_no_specialization_fallback_cost(
    *,
    device: str,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, object]:
    """Measure planner overhead when specialization does not apply."""

    with tempfile.TemporaryDirectory(
        prefix="ollm-runtime-benchmark-tiny-t5-"
    ) as temp_dir:
        temp_root = Path(temp_dir)
        model_dir = create_tiny_t5_fixture(temp_root)
        client = RuntimeClient()
        specialization_enabled_config = RuntimeConfig(
            model_reference=str(model_dir),
            models_dir=temp_root,
            device=device,
        )
        specialization_disabled_config = RuntimeConfig(
            model_reference=str(model_dir),
            models_dir=temp_root,
            device=device,
            backend="transformers-generic",
            use_specialization=False,
        )
        specialization_enabled = measure_callable(
            "fallback-when-no-specialization-applies-enabled",
            lambda: client.plan(specialization_enabled_config),
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            details={
                "model_reference": str(model_dir),
                "device": device,
                "use_specialization": True,
            },
        )
        specialization_disabled = measure_callable(
            "fallback-when-no-specialization-applies-disabled",
            lambda: client.plan(specialization_disabled_config),
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            details={
                "model_reference": str(model_dir),
                "device": device,
                "backend": "transformers-generic",
                "use_specialization": False,
            },
        )
    return {
        "specialization_enabled": specialization_enabled.to_dict(),
        "specialization_disabled": specialization_disabled.to_dict(),
        "mean_delta_ms": mean_delta_ms(specialization_enabled, specialization_disabled),
    }


def _measure_runtime_comparison(
    *,
    repo_root: Path,
    benchmark_model_reference: str,
    models_dir: Path,
    device: str,
    kv_cache_strategy: str | None,
    strategy_selector_profile: str,
    kv_cache_window_tokens: int | None,
    offload_cpu_layers: int,
    offload_cpu_policy: str,
    offload_gpu_layers: int,
    iterations: int,
    warmup_iterations: int,
    prompt_token_targets: tuple[int, ...],
    output_token_targets: tuple[int, ...],
    session_turns: int,
    session_max_new_tokens: int,
    include_family_results: bool,
    include_primary_extended_scenarios: bool,
    cold_timeout_seconds: float,
    warm_timeout_seconds: float,
    scaling_timeout_seconds: float,
    session_timeout_seconds: float,
) -> dict[str, object]:
    """Measure runtime comparisons for the requested target and supported families."""

    client = RuntimeClient()
    resolved_primary = client.resolve(benchmark_model_reference, models_dir)
    primary_target = benchmark_runtime_target(
        repo_root=repo_root,
        target=RuntimeComparisonTarget(
            family="requested",
            model_reference=benchmark_model_reference,
            is_materialized=(
                resolved_primary.model_path is not None
                and resolved_primary.model_path.exists()
            ),
            model_path=None
            if resolved_primary.model_path is None
            else str(resolved_primary.model_path),
        ),
        models_dir=models_dir,
        device=device,
        kv_cache_strategy=kv_cache_strategy,
        strategy_selector_profile=strategy_selector_profile,
        kv_cache_window_tokens=kv_cache_window_tokens,
        offload_cpu_layers=offload_cpu_layers,
        offload_cpu_policy=offload_cpu_policy,
        offload_gpu_layers=offload_gpu_layers,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        include_extended_scenarios=include_primary_extended_scenarios,
        prompt_token_targets=prompt_token_targets,
        output_token_targets=output_token_targets,
        session_turns=session_turns,
        session_max_new_tokens=session_max_new_tokens,
        cold_timeout_seconds=cold_timeout_seconds,
        warm_timeout_seconds=warm_timeout_seconds,
        scaling_timeout_seconds=scaling_timeout_seconds,
        session_timeout_seconds=session_timeout_seconds,
    )
    family_results = (
        [
            benchmark_runtime_target(
                repo_root=repo_root,
                target=target,
                models_dir=models_dir,
                device=device,
                kv_cache_strategy=kv_cache_strategy,
                strategy_selector_profile=strategy_selector_profile,
                kv_cache_window_tokens=kv_cache_window_tokens,
                offload_cpu_layers=offload_cpu_layers,
                offload_cpu_policy=offload_cpu_policy,
                offload_gpu_layers=offload_gpu_layers,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
                include_extended_scenarios=False,
                prompt_token_targets=prompt_token_targets,
                output_token_targets=output_token_targets,
                session_turns=session_turns,
                session_max_new_tokens=session_max_new_tokens,
                cold_timeout_seconds=cold_timeout_seconds,
                warm_timeout_seconds=warm_timeout_seconds,
                scaling_timeout_seconds=scaling_timeout_seconds,
                session_timeout_seconds=session_timeout_seconds,
            )
            for target in build_current_supported_family_targets(models_dir)
        ]
        if include_family_results
        else []
    )
    return {
        "primary_target": primary_target,
        "family_results": family_results,
        "all_family_cold_start_comparisons_available": all(
            bool(
                cast(dict[str, object], result["cold_start"]).get(
                    "comparison_available"
                )
            )
            for result in family_results
        ),
        "all_family_warm_runtime_comparisons_available": all(
            bool(
                cast(dict[str, object], result["warm_runtime"]).get(
                    "comparison_available"
                )
            )
            for result in family_results
        ),
    }

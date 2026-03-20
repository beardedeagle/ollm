import json
import sys
from pathlib import Path
from typing import cast

from ollm.async_io import subprocess_run_process
from ollm.runtime.benchmark_details import summarize_request_metrics
from ollm.runtime.benchmark_probes import (
    OutputScalingCase,
    OutputScalingProbeResult,
    PromptScalingCase,
    PromptScalingProbeResult,
    ReopenSessionGrowthProbeResult,
    ReopenSessionGrowthTurn,
    RuntimeProbeResult,
    SessionGrowthProbeResult,
    SessionGrowthTurn,
    WarmRuntimeProbeResult,
    parse_output_scaling_probe_result,
    parse_prompt_scaling_probe_result,
    parse_reopen_session_growth_probe_result,
    parse_session_growth_probe_result,
    parse_warm_runtime_probe_result,
    render_output_scaling_probe_json,
    render_prompt_scaling_probe_json,
    render_reopen_session_growth_probe_json,
    render_session_growth_probe_json,
    render_warm_runtime_probe_json,
)
from ollm.runtime.benchmarks import (
    BenchmarkMeasurement,
    BenchmarkStats,
    CommandBenchmarkSpec,
    RuntimeBenchmarkReport,
    RuntimeComparisonTarget,
    benchmark_runtime_target,
    build_current_supported_family_targets,
    build_host_summary,
    measure_runtime_probe,
    render_report_json,
    render_runtime_probe_json,
)
from tests.benchmark_support import (
    build_request_probe_metrics,
    build_stage_resources,
)


def test_render_report_json_round_trips() -> None:
    report = RuntimeBenchmarkReport(
        generated_at="2026-03-15T00:00:00+00:00",
        host=build_host_summary(),
        benchmark_model_reference="llama3-1B-chat",
        device="cpu",
        specialization_planner_overhead={
            "specialization_enabled": BenchmarkMeasurement(
                name="enabled",
                status="measured",
                stats=BenchmarkStats(
                    iterations=1,
                    warmup_iterations=0,
                    min_ms=1.0,
                    median_ms=1.0,
                    p95_ms=1.0,
                    max_ms=1.0,
                    mean_ms=1.0,
                ),
                details={},
            ).to_dict(),
            "specialization_disabled": BenchmarkMeasurement(
                name="disabled",
                status="measured",
                stats=BenchmarkStats(
                    iterations=1,
                    warmup_iterations=0,
                    min_ms=0.5,
                    median_ms=0.5,
                    p95_ms=0.5,
                    max_ms=0.5,
                    mean_ms=0.5,
                ),
                details={},
            ).to_dict(),
            "mean_delta_ms": 0.5,
        },
        fallback_cost_when_no_specialization_applies={"mean_delta_ms": 0.25},
        runtime_comparison={"comparison_available": False, "reason": "unsupported"},
    )

    payload = json.loads(render_report_json(report))

    assert payload["benchmark_model_reference"] == "llama3-1B-chat"
    assert payload["runtime_comparison"]["comparison_available"] is False


def test_render_runtime_probe_json_round_trips() -> None:
    probe = RuntimeProbeResult(
        load_ms=10.0,
        load_resources=build_stage_resources(),
        request=build_request_probe_metrics(),
    )

    payload = json.loads(render_runtime_probe_json(probe))

    assert payload["load_ms"] == 10.0
    request = cast(dict[str, object], payload["request"])
    resources = cast(dict[str, object], request["resources"])
    native_runtime_profile = cast(dict[str, object], request["native_runtime_profile"])
    cache_state = cast(dict[str, object], request["cache_state"])
    events = cast(dict[str, object], native_runtime_profile["events"])
    assert request["output_tokens"] == 4
    assert request["kv_cache_strategy"] == "chunked"
    adaptation = cast(dict[str, object], request["kv_cache_adaptation"])
    assert adaptation["adaptation_mode"] == "observe-only"
    assert adaptation["recommendation_available"] is True
    assert adaptation["recommended_strategy_id"] == "chunked"
    assert cache_state["strategy_id"] == "chunked"
    assert cache_state["persistence_format"] == "chunked-manifest"
    assert cache_state["residency_mode"] == "buffered-tail"
    assert cache_state["window_policy"] == "full-history"
    assert cache_state["cold_tier_encoding"] == "full-precision"
    assert cache_state["cold_tier_representation"] is None
    assert cache_state["persisted_artifact_count"] == 2
    assert cache_state["resident_layer_count"] == 2
    assert cache_state["resident_tokens"] == 128
    assert cache_state["resident_bytes"] == 4096
    assert cache_state["compaction_count"] == 0
    assert cache_state["cold_store_format"] is None
    assert resources["accelerator_kind"] == "cuda"
    assert "kvload" in events


def test_build_current_supported_family_targets_returns_unique_families(
    tmp_path: Path,
) -> None:
    targets = build_current_supported_family_targets(tmp_path)

    families = {target.family for target in targets}

    assert families == {"llama", "gemma3", "qwen3-next", "gpt-oss", "voxtral"}
    assert all(not target.is_materialized for target in targets)


def test_build_current_supported_family_targets_prefers_materialized_alias(
    tmp_path: Path,
) -> None:
    (tmp_path / "llama3-8B-chat").mkdir(parents=True)

    targets = build_current_supported_family_targets(tmp_path)

    llama_target = next(target for target in targets if target.family == "llama")
    assert llama_target.is_materialized is True
    assert llama_target.model_reference == "llama3-8B-chat"


def test_benchmark_runtime_target_reports_unmaterialized_family(tmp_path: Path) -> None:
    result = benchmark_runtime_target(
        repo_root=tmp_path,
        target=RuntimeComparisonTarget(
            family="gemma3",
            model_reference="gemma3-12B",
            is_materialized=False,
            model_path=str(tmp_path / "models" / "gemma3-12B"),
        ),
        models_dir=tmp_path,
        device="cpu",
        iterations=1,
        warmup_iterations=0,
    )

    cold_start = cast(dict[str, object], result["cold_start"])
    warm_runtime = cast(dict[str, object], result["warm_runtime"])
    assert cold_start["comparison_available"] is False
    assert warm_runtime["comparison_available"] is False
    generic_result = cast(dict[str, object], cold_start["generic"])
    optimized_result = cast(dict[str, object], cold_start["optimized_native"])
    assert generic_result["status"] == "unavailable"
    assert optimized_result["status"] == "unavailable"
    assert cold_start["reason"] == "model is not materialized locally"


def test_benchmark_runtime_target_reports_unmaterialized_requested_target(
    tmp_path: Path,
) -> None:
    missing_model = tmp_path / "missing-model"

    result = benchmark_runtime_target(
        repo_root=Path(__file__).resolve().parents[1],
        target=RuntimeComparisonTarget(
            family="requested",
            model_reference=str(missing_model),
            is_materialized=False,
            model_path=str(missing_model),
        ),
        models_dir=tmp_path,
        device="cpu",
        iterations=1,
        warmup_iterations=0,
        include_extended_scenarios=True,
    )

    assert result["materialized"] is False
    cold_start = cast(dict[str, object], result["cold_start"])
    assert cold_start["reason"] == "model is not materialized locally"
    prompt_length_scaling = cast(dict[str, object], result["prompt_length_scaling"])
    assert (
        cast(dict[str, object], prompt_length_scaling["generic"])["status"]
        == "unavailable"
    )


def test_measure_runtime_probe_failure_surfaces_reason(tmp_path: Path) -> None:
    spec = CommandBenchmarkSpec(
        name="runtime-probe-fail",
        command=(
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('missing model artifacts\\nextra detail\\n'); sys.exit(2)",
        ),
        timeout_seconds=10.0,
    )

    measurement = measure_runtime_probe(
        spec, iterations=1, warmup_iterations=0, cwd=tmp_path
    )

    assert measurement.status == "unavailable"
    assert measurement.details["reason"] == "missing model artifacts"


def test_summarize_request_metrics_includes_native_runtime_profile() -> None:
    summary = summarize_request_metrics([build_request_probe_metrics()])

    native_runtime_profile = cast(dict[str, object], summary["native_runtime_profile"])
    events = cast(dict[str, object], native_runtime_profile["events"])
    kvload = cast(dict[str, object], events["kvload"])

    assert native_runtime_profile["storage_paths"] == [
        "disk-kv-cache",
        "safetensor-io",
    ]
    assert cast(dict[str, object], summary["cache"])["kv_cache_strategy"] == "chunked"
    adaptation = cast(
        dict[str, object],
        cast(dict[str, object], summary["cache"])["kv_cache_adaptation"],
    )
    assert adaptation["adaptation_mode"] == "observe-only"
    assert adaptation["recommendation_available"] is True
    assert adaptation["recommended_strategy_id"] == "chunked"
    cache_state = cast(
        dict[str, object], cast(dict[str, object], summary["cache"])["cache_state"]
    )
    assert cache_state["policy_id"] == "test-policy"
    assert cache_state["persistence_format"] == "chunked-manifest"
    assert cache_state["residency_mode"] == "buffered-tail"
    assert cache_state["window_policy"] == "full-history"
    assert cache_state["cold_tier_encoding"] == "full-precision"
    assert cache_state["cold_tier_representation"] is None
    persisted_artifact_count = cast(
        dict[str, object], cache_state["persisted_artifact_count"]
    )
    assert persisted_artifact_count["mean"] == 2.0
    resident_layer_count = cast(dict[str, object], cache_state["resident_layer_count"])
    assert resident_layer_count["mean"] == 2.0
    resident_tokens = cast(dict[str, object], cache_state["resident_tokens"])
    assert resident_tokens["mean"] == 128.0
    resident_bytes = cast(dict[str, object], cache_state["resident_bytes"])
    assert resident_bytes["mean"] == 4096.0
    compaction_count = cast(dict[str, object], cache_state["compaction_count"])
    assert compaction_count["mean"] == 0.0
    assert cache_state["cold_store_format"] is None
    assert kvload["event_count"] == 2


def test_warm_probe_round_trips() -> None:
    payload = render_warm_runtime_probe_json(
        WarmRuntimeProbeResult(
            runtime_load_ms=10.0,
            runtime_load_resources=build_stage_resources(),
            warmup_iterations=1,
            measured_iterations=(build_request_probe_metrics(),),
        )
    )

    parsed = parse_warm_runtime_probe_result(payload)

    assert parsed.runtime_load_ms == 10.0
    assert parsed.measured_iterations[0].output_tokens == 4


def test_scaling_and_session_probes_round_trip() -> None:
    prompt_payload = render_prompt_scaling_probe_json(
        PromptScalingProbeResult(
            runtime_load_ms=10.0,
            runtime_load_resources=build_stage_resources(),
            cases=(
                PromptScalingCase(
                    requested_prompt_tokens=32,
                    request=build_request_probe_metrics(),
                ),
            ),
        )
    )
    output_payload = render_output_scaling_probe_json(
        OutputScalingProbeResult(
            runtime_load_ms=10.0,
            runtime_load_resources=build_stage_resources(),
            cases=(
                OutputScalingCase(
                    requested_max_new_tokens=8,
                    request=build_request_probe_metrics(),
                ),
            ),
        )
    )
    session_payload = render_session_growth_probe_json(
        SessionGrowthProbeResult(
            runtime_load_ms=10.0,
            runtime_load_resources=build_stage_resources(),
            turns=(
                SessionGrowthTurn(turn_index=1, request=build_request_probe_metrics()),
            ),
        )
    )

    parsed_prompt = parse_prompt_scaling_probe_result(prompt_payload)
    parsed_output = parse_output_scaling_probe_result(output_payload)
    parsed_session = parse_session_growth_probe_result(session_payload)

    assert parsed_prompt.cases[0].requested_prompt_tokens == 32
    assert parsed_output.cases[0].requested_max_new_tokens == 8
    assert parsed_session.turns[0].turn_index == 1


def test_reopen_session_growth_probe_round_trips() -> None:
    payload = render_reopen_session_growth_probe_json(
        ReopenSessionGrowthProbeResult(
            turns=(
                ReopenSessionGrowthTurn(
                    turn_index=1,
                    runtime_load_ms=12.0,
                    runtime_load_resources=build_stage_resources(),
                    request=build_request_probe_metrics(),
                ),
            ),
        )
    )

    parsed = parse_reopen_session_growth_probe_result(payload)

    assert len(parsed.turns) == 1
    assert parsed.turns[0].turn_index == 1
    assert parsed.turns[0].runtime_load_ms == 12.0
    assert parsed.turns[0].request.output_tokens == 4


def test_benchmark_runtime_cli_rejects_invalid_iterations() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess_run_process(
        (sys.executable, "scripts/benchmark_runtime.py", "--iterations", "0"),
        cwd=str(repo_root),
    )

    assert completed.returncode != 0
    assert "must be a positive integer" in completed.stderr


def test_benchmark_runtime_cli_rejects_invalid_prompt_scale_tokens() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess_run_process(
        (
            sys.executable,
            "scripts/benchmark_runtime.py",
            "--prompt-scale-tokens",
            "abc,128",
        ),
        cwd=str(repo_root),
    )

    assert completed.returncode != 0
    assert "comma-separated list of positive integers" in completed.stderr

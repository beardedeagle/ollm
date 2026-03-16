from pathlib import Path
import json
from typing import cast
import subprocess
import sys

from ollm.client import RuntimeClient
from ollm.runtime.benchmark_probes import (
    OutputScalingCase,
    OutputScalingProbeResult,
    PromptScalingCase,
    PromptScalingProbeResult,
    RequestProbeMetrics,
    RuntimeProbeResult,
    SessionGrowthProbeResult,
    SessionGrowthTurn,
    WarmRuntimeProbeResult,
    parse_output_scaling_probe_result,
    parse_prompt_scaling_probe_result,
    parse_session_growth_probe_result,
    parse_warm_runtime_probe_result,
    render_output_scaling_probe_json,
    render_prompt_scaling_probe_json,
    render_session_growth_probe_json,
    render_warm_runtime_probe_json,
)
from ollm.runtime.benchmark_resources import StageResourceSnapshot
from ollm.runtime.benchmarks import (
    BenchmarkMeasurement,
    BenchmarkStats,
    CommandBenchmarkSpec,
    RuntimeBenchmarkReport,
    benchmark_runtime_target,
    build_runtime_probe_command,
    build_current_supported_family_targets,
    build_host_summary,
    create_tiny_t5_fixture,
    measure_callable,
    measure_command,
    measure_runtime_probe,
    measure_no_specialization_fallback_cost,
    render_report_json,
    render_runtime_probe_json,
    RuntimeComparisonTarget,
)
from ollm.runtime.config import RuntimeConfig


def build_stage_resources() -> StageResourceSnapshot:
    return StageResourceSnapshot(
        current_rss_mb=128.0,
        peak_rss_mb=132.0,
        peak_rss_source="native",
        accelerator_kind="cuda",
        accelerator_current_mb=256.0,
        accelerator_peak_mb=300.0,
        accelerator_reserved_mb=280.0,
        accelerator_peak_reserved_mb=320.0,
        accelerator_peak_source="native",
        process_cpu_utilization_percent=88.0,
        accelerator_utilization=None,
    )


def build_request_probe_metrics() -> RequestProbeMetrics:
    return RequestProbeMetrics(
        total_ms=30.0,
        generation_ms=20.0,
        time_to_first_token_ms=5.0,
        inter_token_latencies_ms=(2.0, 2.5, 3.0),
        prompt_tokens=16,
        prompt_tokens_per_second=3200.0,
        output_tokens=4,
        output_tokens_per_second=200.0,
        cache_mode="disk-kv",
        cache_dir_size_mb=12.0,
        allocator_gap_mb=20.0,
        allocator_gap_ratio=0.066667,
        resources=build_stage_resources(),
        text_excerpt="Hello",
    )


def test_measure_callable_reports_stats() -> None:
    counter = {"value": 0}

    def operation() -> int:
        counter["value"] += 1
        return counter["value"]

    measurement = measure_callable(
        "increment", operation, iterations=3, warmup_iterations=1
    )

    assert measurement.status == "measured"
    assert measurement.stats is not None
    assert measurement.stats.iterations == 3
    assert counter["value"] == 4


def test_measure_command_reports_success(tmp_path: Path) -> None:
    spec = CommandBenchmarkSpec(
        name="python-ok",
        command=(sys.executable, "-c", "print('ok')"),
        timeout_seconds=10.0,
    )

    measurement = measure_command(spec, iterations=2, warmup_iterations=1, cwd=tmp_path)

    assert measurement.status == "measured"
    assert measurement.stats is not None
    stdout_excerpt = measurement.details["stdout_excerpt"]
    assert isinstance(stdout_excerpt, str)
    assert stdout_excerpt.strip() == "ok"


def test_measure_command_reports_unavailable_on_failure(tmp_path: Path) -> None:
    spec = CommandBenchmarkSpec(
        name="python-fail",
        command=(sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"),
        timeout_seconds=10.0,
    )

    measurement = measure_command(spec, iterations=2, warmup_iterations=1, cwd=tmp_path)

    assert measurement.status == "unavailable"
    assert measurement.stats is None
    assert measurement.details["returncode"] == 3
    stdout_excerpt = measurement.details["stdout_excerpt"]
    assert isinstance(stdout_excerpt, str)
    assert stdout_excerpt.strip() == "boom"


def test_measure_command_reports_unavailable_on_timeout(tmp_path: Path) -> None:
    spec = CommandBenchmarkSpec(
        name="python-timeout",
        command=(sys.executable, "-c", "import time; time.sleep(0.2)"),
        timeout_seconds=0.01,
    )

    measurement = measure_command(spec, iterations=1, warmup_iterations=0, cwd=tmp_path)

    assert measurement.status == "unavailable"
    assert measurement.stats is None
    assert measurement.details["timed_out"] is True
    stderr_excerpt = measurement.details["stderr_excerpt"]
    assert isinstance(stderr_excerpt, str)
    assert "timed out" in stderr_excerpt


def test_create_tiny_t5_fixture_supports_generic_plan(tmp_path: Path) -> None:
    model_dir = create_tiny_t5_fixture(tmp_path)
    client = RuntimeClient()
    config = RuntimeConfig(
        model_reference=str(model_dir), models_dir=tmp_path, device="cpu"
    )

    plan = client.plan(config)

    assert plan.backend_id == "transformers-generic"
    assert plan.specialization_state.value == "not-planned"


def test_measure_no_specialization_fallback_cost_returns_measurements(
    tmp_path: Path,
) -> None:
    report = measure_no_specialization_fallback_cost(
        device="cpu",
        iterations=2,
        warmup_iterations=1,
    )

    specialization_enabled = cast(dict[str, object], report["specialization_enabled"])
    specialization_disabled = cast(dict[str, object], report["specialization_disabled"])
    assert specialization_enabled["status"] == "measured"
    assert specialization_disabled["status"] == "measured"
    assert report["mean_delta_ms"] is not None


def test_build_runtime_probe_command_handles_specialization_flag(
    tmp_path: Path,
) -> None:
    command = build_runtime_probe_command(
        tmp_path,
        "llama3-1B-chat",
        models_dir=Path("models"),
        device="cpu",
        backend="transformers-generic",
        use_specialization=False,
    )

    assert "--probe-backend" in command
    assert "transformers-generic" in command
    assert "--probe-no-specialization" in command


def test_measure_runtime_probe_reports_success(tmp_path: Path) -> None:
    payload = render_runtime_probe_json(
        RuntimeProbeResult(
            load_ms=10.0,
            load_resources=build_stage_resources(),
            request=build_request_probe_metrics(),
        )
    )
    spec = CommandBenchmarkSpec(
        name="runtime-probe-ok",
        command=(sys.executable, "-c", f"print({payload!r})"),
        timeout_seconds=10.0,
    )

    measurement = measure_runtime_probe(
        spec, iterations=2, warmup_iterations=0, cwd=tmp_path
    )

    assert measurement.status == "measured"
    assert measurement.stats is not None
    load = cast(dict[str, object], measurement.details["load"])
    metrics = cast(dict[str, object], measurement.details["metrics"])
    throughput = cast(dict[str, object], metrics["throughput"])
    memory = cast(dict[str, object], metrics["memory"])
    assert cast(dict[str, object], load["latency_ms"])["mean"] == 10.0
    assert throughput["prompt_tokens_per_second"] is not None
    assert throughput["output_tokens_per_second"] is not None
    assert memory["peak_rss_mb"] is not None


def test_measure_runtime_probe_reports_unavailable_on_invalid_json(
    tmp_path: Path,
) -> None:
    spec = CommandBenchmarkSpec(
        name="runtime-probe-invalid",
        command=(sys.executable, "-c", "print('not-json')"),
        timeout_seconds=10.0,
    )

    measurement = measure_runtime_probe(
        spec, iterations=1, warmup_iterations=0, cwd=tmp_path
    )

    assert measurement.status == "unavailable"
    assert measurement.stats is None
    assert measurement.details["reason"] == "runtime probe did not emit valid JSON"


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
    assert request["output_tokens"] == 4
    assert resources["accelerator_kind"] == "cuda"


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


def test_benchmark_runtime_cli_rejects_invalid_iterations() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "scripts/benchmark_runtime.py", "--iterations", "0"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "must be a positive integer" in completed.stderr


def test_benchmark_runtime_cli_rejects_invalid_prompt_scale_tokens() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_runtime.py",
            "--prompt-scale-tokens",
            "abc,128",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "comma-separated list of positive integers" in completed.stderr

from pathlib import Path
import json
from typing import cast
import subprocess
import sys

from ollm.client import RuntimeClient
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
    RuntimeProbeResult,
)
from ollm.runtime.config import RuntimeConfig


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
    payload = {
        "load_ms": 10.0,
        "generation_ms": 20.0,
        "total_ms": 30.0,
        "output_tokens": 4,
        "output_tokens_per_second": 200.0,
        "rss_after_load_mb": 128.0,
        "rss_after_generate_mb": 132.0,
        "accelerator_kind": None,
        "accelerator_current_after_load_mb": None,
        "accelerator_current_after_generate_mb": None,
        "accelerator_peak_mb": None,
        "accelerator_reserved_after_load_mb": None,
        "accelerator_reserved_after_generate_mb": None,
        "accelerator_peak_reserved_mb": None,
        "text_excerpt": "Hello",
    }
    spec = CommandBenchmarkSpec(
        name="runtime-probe-ok",
        command=(sys.executable, "-c", f"import json; print(json.dumps({payload!r}))"),
        timeout_seconds=10.0,
    )

    measurement = measure_runtime_probe(
        spec, iterations=2, warmup_iterations=1, cwd=tmp_path
    )

    assert measurement.status == "measured"
    assert measurement.stats is not None
    metrics = cast(dict[str, object], measurement.details["metrics"])
    throughput = cast(dict[str, object], metrics["throughput"])
    memory = cast(dict[str, object], metrics["memory"])
    assert throughput["output_tokens_per_second"] is not None
    assert memory["rss_after_load_mb"] is not None


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
        generation_ms=20.0,
        total_ms=30.0,
        output_tokens=4,
        output_tokens_per_second=200.0,
        rss_after_load_mb=128.0,
        rss_after_generate_mb=132.0,
        accelerator_kind="cuda",
        accelerator_current_after_load_mb=256.0,
        accelerator_current_after_generate_mb=260.0,
        accelerator_peak_mb=300.0,
        accelerator_reserved_after_load_mb=280.0,
        accelerator_reserved_after_generate_mb=284.0,
        accelerator_peak_reserved_mb=320.0,
        text_excerpt="Hello",
    )

    payload = json.loads(render_runtime_probe_json(probe))

    assert payload["output_tokens"] == 4
    assert payload["accelerator_kind"] == "cuda"


def test_build_current_supported_family_targets_returns_unique_families(
    tmp_path: Path,
) -> None:
    targets = build_current_supported_family_targets(tmp_path)

    families = {target.family for target in targets}

    assert families == {"llama", "gemma3", "qwen3-next", "gpt-oss", "voxtral"}
    assert all(not target.is_materialized for target in targets)


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

    assert result["comparison_available"] is False
    generic_result = cast(dict[str, object], result["generic"])
    optimized_result = cast(dict[str, object], result["optimized_native"])
    assert generic_result["status"] == "unavailable"
    assert optimized_result["status"] == "unavailable"
    assert (
        result["reason"]
        == "Neither runtime benchmark completed successfully on this host"
    )


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

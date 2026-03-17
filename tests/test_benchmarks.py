import sys
from pathlib import Path
from typing import cast

from ollm.client import RuntimeClient
from ollm.runtime.benchmark_probes import RuntimeProbeResult
from ollm.runtime.benchmarks import (
    CommandBenchmarkSpec,
    build_runtime_probe_command,
    create_tiny_t5_fixture,
    measure_callable,
    measure_command,
    measure_no_specialization_fallback_cost,
    measure_runtime_probe,
    render_runtime_probe_json,
)
from ollm.runtime.config import RuntimeConfig
from tests.benchmark_support import (
    build_request_probe_metrics,
    build_stage_resources,
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


def test_measure_no_specialization_fallback_cost_returns_measurements() -> None:
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

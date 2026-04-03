import sys
from pathlib import Path
from typing import cast

import ollm.runtime.benchmark.targets as benchmark_targets_module
from ollm.client import RuntimeClient
from ollm.runtime.benchmark import (
    CommandBenchmarkSpec,
    build_runtime_probe_command,
    create_tiny_t5_fixture,
    measure_callable,
    measure_command,
    measure_no_specialization_fallback_cost,
    measure_runtime_probe,
    render_runtime_probe_json,
)
from ollm.runtime.benchmark.probe_registry import ProbeMode
from ollm.runtime.benchmark.probes import RuntimeProbeResult
from ollm.runtime.benchmark.types import (
    BenchmarkMeasurement,
    BenchmarkStats,
    RuntimeComparisonTarget,
    resolve_runtime_benchmark_profile,
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
        kv_cache_strategy="sliding-window-ring-buffer",
        kv_cache_window_tokens=64,
        probe_mode=ProbeMode.COLD,
    )

    assert "--probe-backend" in command
    assert "transformers-generic" in command
    assert "--probe-kv-cache-strategy" in command
    assert "sliding-window-ring-buffer" in command
    assert "--probe-kv-cache-window-tokens" in command
    assert "64" in command
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


def test_probe_execution_module_avoids_runtime_executor_private_helpers() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ollm"
        / "runtime"
        / "benchmark"
        / "probe_execution.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "RuntimeExecutor" not in source
    assert "._validate_request(" not in source
    assert "._build_inputs(" not in source
    assert "._build_generate_kwargs(" not in source
    assert "._prepare_generate_inputs(" not in source
    assert "._decode_response(" not in source


def test_benchmark_runtime_target_uses_session_specific_max_new_tokens(
    monkeypatch, tmp_path: Path
) -> None:
    session_commands: list[tuple[str, ...]] = []

    def measured(name: str) -> BenchmarkMeasurement:
        return BenchmarkMeasurement(
            name=name,
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
        )

    monkeypatch.setattr(
        benchmark_targets_module,
        "measure_runtime_probe",
        lambda spec, iterations, warmup_iterations, cwd: measured(spec.name),
    )
    monkeypatch.setattr(
        benchmark_targets_module,
        "measure_warm_runtime_probe",
        lambda spec, cwd: measured(spec.name),
    )
    monkeypatch.setattr(
        benchmark_targets_module,
        "measure_prompt_scaling_probe",
        lambda spec, cwd: measured(spec.name),
    )
    monkeypatch.setattr(
        benchmark_targets_module,
        "measure_output_scaling_probe",
        lambda spec, cwd: measured(spec.name),
    )

    def fake_measure_session_growth_probe(spec, *, cwd):
        del cwd
        session_commands.append(spec.command)
        return measured(spec.name)

    monkeypatch.setattr(
        benchmark_targets_module,
        "measure_session_growth_probe",
        fake_measure_session_growth_probe,
    )

    benchmark_targets_module.benchmark_runtime_target(
        repo_root=tmp_path,
        target=RuntimeComparisonTarget(
            family="gemma3",
            model_reference="gemma3-12B",
            is_materialized=True,
            model_path=str(tmp_path / "models" / "gemma3-12B"),
        ),
        models_dir=tmp_path / "models",
        device="cpu",
        iterations=1,
        warmup_iterations=0,
        include_extended_scenarios=True,
        output_token_targets=(16, 64, 128),
        session_turns=4,
        session_max_new_tokens=4,
    )

    assert len(session_commands) == 2
    for command in session_commands:
        assert command[command.index("--probe-max-new-tokens") + 1] == "4"


def test_resolve_runtime_benchmark_profile_quick_defaults() -> None:
    profile = resolve_runtime_benchmark_profile(profile="quick")

    assert profile.profile_id == "quick"
    assert profile.iterations == 1
    assert profile.warmup_iterations == 0
    assert profile.include_family_results is False
    assert profile.include_primary_extended_scenarios is False
    assert profile.cold_timeout_seconds == 90.0
    assert profile.session_max_new_tokens == 4


def test_resolve_runtime_benchmark_profile_full_defaults() -> None:
    profile = resolve_runtime_benchmark_profile(profile="full")

    assert profile.profile_id == "full"
    assert profile.iterations == 5
    assert profile.warmup_iterations == 1
    assert profile.include_family_results is True
    assert profile.include_primary_extended_scenarios is True
    assert profile.warm_timeout_seconds == 240.0
    assert profile.session_max_new_tokens == 4

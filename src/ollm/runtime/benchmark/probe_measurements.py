"""Probe-specific benchmark measurement helpers."""

from pathlib import Path

from ollm.runtime.benchmark.commands import (
    command_failure_measurement,
    probe_parse_failure_measurement,
    run_command,
    summarize_samples,
)
from ollm.runtime.benchmark.details import summarize_request_metrics
from ollm.runtime.benchmark.probes import (
    parse_output_scaling_probe_result,
    parse_prompt_scaling_probe_result,
    parse_session_growth_probe_result,
    parse_warm_runtime_probe_result,
)
from ollm.runtime.benchmark.types import BenchmarkMeasurement, CommandBenchmarkSpec


def measure_warm_runtime_probe(
    spec: CommandBenchmarkSpec,
    *,
    cwd: Path,
) -> BenchmarkMeasurement:
    result = run_command(spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds)
    if result.returncode != 0:
        return command_failure_measurement(spec, result, warmup_only=False)
    try:
        probe = parse_warm_runtime_probe_result(result.stdout)
    except ValueError:
        return probe_parse_failure_measurement(
            spec,
            result,
            warmup_only=False,
            reason="warm runtime probe did not emit valid JSON",
        )
    samples = list(probe.measured_iterations)
    return BenchmarkMeasurement(
        name=spec.name,
        status="measured",
        stats=summarize_samples(
            [sample.total_ms for sample in samples],
            len(samples),
            probe.warmup_iterations,
        ),
        details={
            "command": list(spec.command),
            "runtime_load_ms": probe.runtime_load_ms,
            "runtime_load_resources": probe.runtime_load_resources.to_dict(),
            "warmup_iterations": probe.warmup_iterations,
            "metrics": summarize_request_metrics(samples),
        },
    )


def measure_prompt_scaling_probe(
    spec: CommandBenchmarkSpec,
    *,
    cwd: Path,
) -> BenchmarkMeasurement:
    result = run_command(spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds)
    if result.returncode != 0:
        return command_failure_measurement(spec, result, warmup_only=False)
    try:
        probe = parse_prompt_scaling_probe_result(result.stdout)
    except ValueError:
        return probe_parse_failure_measurement(
            spec,
            result,
            warmup_only=False,
            reason="prompt scaling probe did not emit valid JSON",
        )
    return BenchmarkMeasurement(
        name=spec.name,
        status="measured",
        stats=None,
        details={
            "command": list(spec.command),
            "runtime_load_ms": probe.runtime_load_ms,
            "runtime_load_resources": probe.runtime_load_resources.to_dict(),
            "cases": [case.to_dict() for case in probe.cases],
        },
    )


def measure_output_scaling_probe(
    spec: CommandBenchmarkSpec,
    *,
    cwd: Path,
) -> BenchmarkMeasurement:
    result = run_command(spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds)
    if result.returncode != 0:
        return command_failure_measurement(spec, result, warmup_only=False)
    try:
        probe = parse_output_scaling_probe_result(result.stdout)
    except ValueError:
        return probe_parse_failure_measurement(
            spec,
            result,
            warmup_only=False,
            reason="output scaling probe did not emit valid JSON",
        )
    return BenchmarkMeasurement(
        name=spec.name,
        status="measured",
        stats=None,
        details={
            "command": list(spec.command),
            "runtime_load_ms": probe.runtime_load_ms,
            "runtime_load_resources": probe.runtime_load_resources.to_dict(),
            "cases": [case.to_dict() for case in probe.cases],
        },
    )


def measure_session_growth_probe(
    spec: CommandBenchmarkSpec,
    *,
    cwd: Path,
) -> BenchmarkMeasurement:
    result = run_command(spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds)
    if result.returncode != 0:
        return command_failure_measurement(spec, result, warmup_only=False)
    try:
        probe = parse_session_growth_probe_result(result.stdout)
    except ValueError:
        return probe_parse_failure_measurement(
            spec,
            result,
            warmup_only=False,
            reason="session growth probe did not emit valid JSON",
        )
    return BenchmarkMeasurement(
        name=spec.name,
        status="measured",
        stats=None,
        details={
            "command": list(spec.command),
            "runtime_load_ms": probe.runtime_load_ms,
            "runtime_load_resources": probe.runtime_load_resources.to_dict(),
            "turns": [turn.to_dict() for turn in probe.turns],
        },
    )

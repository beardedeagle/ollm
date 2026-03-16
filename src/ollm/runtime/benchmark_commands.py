"""Benchmark command execution and sample summarization helpers."""

import asyncio
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from ollm.runtime.benchmark_details import build_cold_probe_details, clip_text
from ollm.runtime.benchmark_probes import parse_runtime_probe_result
from ollm.runtime.benchmark_types import (
    BenchmarkMeasurement,
    BenchmarkStats,
    CommandBenchmarkSpec,
    CommandExecutionResult,
)

T = TypeVar("T")


def measure_callable(
    name: str,
    operation: Callable[[], T],
    *,
    iterations: int,
    warmup_iterations: int = 1,
    details: dict[str, object] | None = None,
) -> BenchmarkMeasurement:
    """Measure an in-process callable over repeated iterations."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    for _ in range(warmup_iterations):
        operation()
    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    return BenchmarkMeasurement(
        name=name,
        status="measured",
        stats=summarize_samples(samples_ms, iterations, warmup_iterations),
        details={} if details is None else details,
    )


def measure_command(
    spec: CommandBenchmarkSpec,
    *,
    iterations: int,
    warmup_iterations: int = 1,
    cwd: Path,
) -> BenchmarkMeasurement:
    """Measure a subprocess-backed benchmark command."""

    last_result: CommandExecutionResult | None = None
    for _ in range(warmup_iterations):
        last_result = run_command(
            spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds
        )
        if last_result.returncode != 0:
            return command_failure_measurement(spec, last_result, warmup_only=True)
    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        last_result = run_command(
            spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
        if last_result.returncode != 0:
            failure = command_failure_measurement(spec, last_result, warmup_only=False)
            failure_details = dict(failure.details)
            failure_details["failed_after_ms"] = round(duration_ms, 6)
            return BenchmarkMeasurement(
                name=failure.name,
                status=failure.status,
                stats=failure.stats,
                details=failure_details,
            )
        samples_ms.append(duration_ms)
    return BenchmarkMeasurement(
        name=spec.name,
        status="measured",
        stats=summarize_samples(samples_ms, iterations, warmup_iterations),
        details={
            "command": list(spec.command),
            "stdout_excerpt": clip_text(
                "" if last_result is None else last_result.stdout
            ),
            "stderr_excerpt": clip_text(
                "" if last_result is None else last_result.stderr
            ),
        },
    )


def measure_runtime_probe(
    spec: CommandBenchmarkSpec,
    *,
    iterations: int,
    warmup_iterations: int = 0,
    cwd: Path,
) -> BenchmarkMeasurement:
    """Measure a cold runtime probe command."""

    if warmup_iterations != 0:
        raise ValueError("cold runtime probes do not support warmup iterations")
    probe_samples = []
    for _ in range(iterations):
        result = run_command(
            spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds
        )
        if result.returncode != 0:
            return command_failure_measurement(spec, result, warmup_only=False)
        try:
            probe_samples.append(parse_runtime_probe_result(result.stdout))
        except ValueError:
            return probe_parse_failure_measurement(
                spec,
                result,
                warmup_only=False,
                reason="runtime probe did not emit valid JSON",
            )
    samples_ms = [sample.load_ms + sample.request.total_ms for sample in probe_samples]
    return BenchmarkMeasurement(
        name=spec.name,
        status="measured",
        stats=summarize_samples(samples_ms, iterations, warmup_iterations),
        details=build_cold_probe_details(spec, probe_samples),
    )


def run_command(
    command: tuple[str, ...], *, cwd: Path, timeout_seconds: float
) -> CommandExecutionResult:
    """Run a benchmark subprocess and capture its stdout and stderr."""

    return asyncio.run(
        _run_command_async(command=command, cwd=cwd, timeout_seconds=timeout_seconds)
    )


async def _run_command_async(
    *, command: tuple[str, ...], cwd: Path, timeout_seconds: float
) -> CommandExecutionResult:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return CommandExecutionResult(
            returncode=1,
            stdout="",
            stderr=str(exc),
            timed_out=False,
        )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        timeout_message = f"Command timed out after {timeout_seconds} seconds."
        stderr_text = coerce_subprocess_output(stderr)
        stderr_text = (
            timeout_message if not stderr_text else f"{stderr_text}\n{timeout_message}"
        )
        return CommandExecutionResult(
            returncode=124,
            stdout=coerce_subprocess_output(stdout),
            stderr=stderr_text,
            timed_out=True,
        )
    return CommandExecutionResult(
        returncode=process.returncode or 0,
        stdout=coerce_subprocess_output(stdout),
        stderr=coerce_subprocess_output(stderr),
        timed_out=False,
    )


def command_failure_measurement(
    spec: CommandBenchmarkSpec,
    result: CommandExecutionResult,
    *,
    warmup_only: bool,
) -> BenchmarkMeasurement:
    """Build an unavailable measurement for a failed command."""

    raw_stderr_reason = first_nonempty_line(result.stderr)
    raw_stdout_reason = first_nonempty_line(result.stdout)
    stderr_excerpt = clip_text(result.stderr)
    stdout_excerpt = clip_text(result.stdout)
    return BenchmarkMeasurement(
        name=spec.name,
        status="unavailable",
        stats=None,
        details={
            "command": list(spec.command),
            "returncode": result.returncode,
            "stdout_excerpt": stdout_excerpt,
            "stderr_excerpt": stderr_excerpt,
            "timed_out": result.timed_out,
            "warmup_only": warmup_only,
            "reason": command_failure_reason(
                result=result,
                stderr_reason=raw_stderr_reason,
                stdout_reason=raw_stdout_reason,
            ),
        },
    )


def probe_parse_failure_measurement(
    spec: CommandBenchmarkSpec,
    result: CommandExecutionResult,
    *,
    warmup_only: bool,
    reason: str,
) -> BenchmarkMeasurement:
    """Build an unavailable measurement for an invalid probe payload."""

    return BenchmarkMeasurement(
        name=spec.name,
        status="unavailable",
        stats=None,
        details={
            "command": list(spec.command),
            "stdout_excerpt": clip_text(result.stdout),
            "stderr_excerpt": clip_text(result.stderr),
            "timed_out": result.timed_out,
            "warmup_only": warmup_only,
            "reason": reason,
        },
    )


def summarize_samples(
    samples_ms: list[float], iterations: int, warmup_iterations: int
) -> BenchmarkStats:
    """Summarize repeated timing samples."""

    sorted_samples = sorted(samples_ms)
    p95_index = max(0, int(round((len(sorted_samples) - 1) * 0.95)))
    return BenchmarkStats(
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        min_ms=round(sorted_samples[0], 6),
        median_ms=round(statistics.median(sorted_samples), 6),
        p95_ms=round(sorted_samples[p95_index], 6),
        max_ms=round(sorted_samples[-1], 6),
        mean_ms=round(statistics.fmean(sorted_samples), 6),
    )


def coerce_subprocess_output(value: bytes | str) -> str:
    """Convert subprocess output to text."""

    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


def command_failure_reason(
    *,
    result: CommandExecutionResult,
    stderr_reason: str | None,
    stdout_reason: str | None,
) -> str:
    """Choose the best error reason for a failed benchmark command."""

    if result.timed_out:
        return f"Command timed out after benchmark timeout (returncode {result.returncode})"
    if stderr_reason is not None:
        return stderr_reason
    if stdout_reason is not None:
        return stdout_reason
    return f"Command failed with returncode {result.returncode}"


def first_nonempty_line(text: str) -> str | None:
    """Return the first non-empty line from a block of text."""

    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
from pathlib import Path
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Callable, TypeVar, cast

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.t5 import T5Config, T5ForConditionalGeneration

from ollm.app.types import ContentPart, Message, MessageRole, PromptRequest
from ollm.client import RuntimeClient
from ollm.runtime.catalog import list_model_catalog
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import DEFAULT_SYSTEM_PROMPT, GenerationConfig, RuntimeConfig
from ollm.runtime.output_control import suppress_module_prints

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class BenchmarkStats:
    iterations: int
    warmup_iterations: int
    min_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float
    mean_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    name: str
    status: str
    stats: BenchmarkStats | None
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "details": self.details,
        }
        payload["stats"] = None if self.stats is None else self.stats.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeBenchmarkReport:
    generated_at: str
    host: dict[str, object]
    benchmark_model_reference: str
    device: str
    specialization_planner_overhead: dict[str, object]
    fallback_cost_when_no_specialization_applies: dict[str, object]
    runtime_comparison: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "host": self.host,
            "benchmark_model_reference": self.benchmark_model_reference,
            "device": self.device,
            "specialization_planner_overhead": self.specialization_planner_overhead,
            "fallback_cost_when_no_specialization_applies": self.fallback_cost_when_no_specialization_applies,
            "runtime_comparison": self.runtime_comparison,
        }


@dataclass(frozen=True, slots=True)
class CommandBenchmarkSpec:
    name: str
    command: tuple[str, ...]
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class CommandExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeComparisonTarget:
    family: str
    model_reference: str
    is_materialized: bool
    model_path: str | None


@dataclass(frozen=True, slots=True)
class RuntimeProbeResult:
    load_ms: float
    generation_ms: float
    total_ms: float
    output_tokens: int
    output_tokens_per_second: float | None
    rss_after_load_mb: float | None
    rss_after_generate_mb: float | None
    accelerator_kind: str | None
    accelerator_current_after_load_mb: float | None
    accelerator_current_after_generate_mb: float | None
    accelerator_peak_mb: float | None
    accelerator_reserved_after_load_mb: float | None
    accelerator_reserved_after_generate_mb: float | None
    accelerator_peak_reserved_mb: float | None
    text_excerpt: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def measure_callable(
    name: str,
    operation: Callable[[], T],
    *,
    iterations: int,
    warmup_iterations: int = 1,
    details: dict[str, object] | None = None,
) -> BenchmarkMeasurement:
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
        stats=_summarize_samples(samples_ms, iterations, warmup_iterations),
        details={} if details is None else details,
    )


def measure_command(
    spec: CommandBenchmarkSpec,
    *,
    iterations: int,
    warmup_iterations: int = 1,
    cwd: Path,
) -> BenchmarkMeasurement:
    last_result: CommandExecutionResult | None = None
    for _ in range(warmup_iterations):
        last_result = run_command(
            spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds
        )
        if last_result.returncode != 0:
            return _command_failure_measurement(
                spec, last_result, warmup_iterations, warmup_only=True
            )

    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        last_result = run_command(
            spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
        if last_result.returncode != 0:
            failure = _command_failure_measurement(
                spec, last_result, warmup_iterations, warmup_only=False
            )
            failure_details = dict(failure.details)
            failure_details["failed_after_ms"] = round(duration_ms, 3)
            return BenchmarkMeasurement(
                name=failure.name,
                status=failure.status,
                stats=failure.stats,
                details=failure_details,
            )
        samples_ms.append(duration_ms)

    details: dict[str, object] = {
        "command": list(spec.command),
        "stdout_excerpt": _clip_text(
            last_result.stdout if last_result is not None else ""
        ),
        "stderr_excerpt": _clip_text(
            last_result.stderr if last_result is not None else ""
        ),
    }
    return BenchmarkMeasurement(
        name=spec.name,
        status="measured",
        stats=_summarize_samples(samples_ms, iterations, warmup_iterations),
        details=details,
    )


def run_command(
    command: tuple[str, ...], *, cwd: Path, timeout_seconds: float
) -> CommandExecutionResult:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = "" if exc.stderr is None else _coerce_subprocess_output(exc.stderr)
        timeout_message = f"Command timed out after {timeout_seconds} seconds."
        if stderr:
            stderr = f"{stderr}\n{timeout_message}"
        else:
            stderr = timeout_message
        stdout = "" if exc.stdout is None else _coerce_subprocess_output(exc.stdout)
        return CommandExecutionResult(
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )
    return CommandExecutionResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
    )


def measure_runtime_probe(
    spec: CommandBenchmarkSpec,
    *,
    iterations: int,
    warmup_iterations: int = 1,
    cwd: Path,
) -> BenchmarkMeasurement:
    last_result: CommandExecutionResult | None = None
    for _ in range(warmup_iterations):
        last_result = run_command(
            spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds
        )
        if last_result.returncode != 0:
            return _command_failure_measurement(
                spec, last_result, warmup_iterations, warmup_only=True
            )
        try:
            _parse_runtime_probe_result(last_result.stdout)
        except ValueError:
            return _probe_parse_failure_measurement(
                spec, last_result, warmup_iterations, warmup_only=True
            )

    probe_samples: list[RuntimeProbeResult] = []
    for _ in range(iterations):
        last_result = run_command(
            spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds
        )
        if last_result.returncode != 0:
            return _command_failure_measurement(
                spec, last_result, warmup_iterations, warmup_only=False
            )
        try:
            probe_samples.append(_parse_runtime_probe_result(last_result.stdout))
        except ValueError:
            return _probe_parse_failure_measurement(
                spec, last_result, warmup_iterations, warmup_only=False
            )

    return BenchmarkMeasurement(
        name=spec.name,
        status="measured",
        stats=_summarize_samples(
            [sample.total_ms for sample in probe_samples],
            iterations,
            warmup_iterations,
        ),
        details=_build_runtime_probe_details(spec, probe_samples),
    )


def build_runtime_benchmark_report(
    *,
    repo_root: Path,
    benchmark_model_reference: str,
    models_dir: Path,
    device: str,
    iterations: int = 5,
    warmup_iterations: int = 1,
) -> RuntimeBenchmarkReport:
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

    runtime_comparison = measure_runtime_comparison(
        repo_root=repo_root,
        benchmark_model_reference=benchmark_model_reference,
        models_dir=models_root,
        device=device,
        iterations=max(1, min(iterations, 3)),
        warmup_iterations=min(warmup_iterations, 1),
    )

    planner_summary: dict[str, object] = {
        "specialization_enabled": planner_enabled.to_dict(),
        "specialization_disabled": planner_disabled.to_dict(),
        "mean_delta_ms": _mean_delta_ms(planner_enabled, planner_disabled),
    }

    return RuntimeBenchmarkReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        host=build_host_summary(),
        benchmark_model_reference=benchmark_model_reference,
        device=device,
        specialization_planner_overhead=planner_summary,
        fallback_cost_when_no_specialization_applies=fallback_measurements,
        runtime_comparison=runtime_comparison,
    )


def build_host_summary() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def choose_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def measure_no_specialization_fallback_cost(
    *,
    device: str,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, object]:
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
            "mean_delta_ms": _mean_delta_ms(
                specialization_enabled, specialization_disabled
            ),
        }


def measure_runtime_comparison(
    *,
    repo_root: Path,
    benchmark_model_reference: str,
    models_dir: Path,
    device: str,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, object]:
    primary_target = benchmark_runtime_target(
        repo_root=repo_root,
        target=RuntimeComparisonTarget(
            family="requested",
            model_reference=benchmark_model_reference,
            is_materialized=True,
            model_path=None,
        ),
        models_dir=models_dir,
        device=device,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )
    family_results = [
        benchmark_runtime_target(
            repo_root=repo_root,
            target=target,
            models_dir=models_dir,
            device=device,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
        )
        for target in build_current_supported_family_targets(models_dir)
    ]
    comparison_available = primary_target["comparison_available"]
    reason = primary_target["reason"]
    speedup_ratio = primary_target["speedup_ratio"]
    return {
        "primary_target": primary_target,
        "family_results": family_results,
        "all_family_comparisons_available": all(
            result["comparison_available"] for result in family_results
        ),
        "comparison_available": comparison_available,
        "speedup_ratio": speedup_ratio,
        "reason": reason,
    }


def build_runtime_probe_command(
    repo_root: Path,
    model_reference: str,
    *,
    models_dir: Path,
    device: str,
    backend: str,
    use_specialization: bool,
    prompt: str = "Say hi.",
    max_new_tokens: int = 4,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str((repo_root / "scripts" / "benchmark_runtime.py").resolve()),
        "--probe-runtime",
        "--model",
        model_reference,
        "--models-dir",
        str(models_dir),
        "--device",
        device,
        "--probe-backend",
        backend,
        "--probe-prompt",
        prompt,
        "--probe-max-new-tokens",
        str(max_new_tokens),
    ]
    if not use_specialization:
        command.append("--probe-no-specialization")
    return tuple(command)


def build_current_supported_family_targets(
    models_dir: Path,
) -> tuple[RuntimeComparisonTarget, ...]:
    client = RuntimeClient()
    models_root = models_dir.expanduser().resolve()
    targets: list[RuntimeComparisonTarget] = []
    seen_families: set[str] = set()
    for entry in list_model_catalog():
        resolved_model = client.resolve(entry.model_id, models_root)
        if resolved_model.native_family is None:
            continue
        family_name = resolved_model.native_family.value
        if family_name in seen_families:
            continue
        seen_families.add(family_name)
        model_path = resolved_model.model_path
        targets.append(
            RuntimeComparisonTarget(
                family=family_name,
                model_reference=entry.model_id,
                is_materialized=model_path is not None and model_path.exists(),
                model_path=None if model_path is None else str(model_path),
            )
        )
    return tuple(targets)


def benchmark_runtime_target(
    *,
    repo_root: Path,
    target: RuntimeComparisonTarget,
    models_dir: Path,
    device: str,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, object]:
    if not target.is_materialized and target.family != "requested":
        generic = unavailable_measurement(
            name=f"{target.family}-generic-runtime-prompt",
            details={
                "family": target.family,
                "model_reference": target.model_reference,
                "model_path": target.model_path,
                "reason": "model is not materialized locally",
            },
        )
        optimized = unavailable_measurement(
            name=f"{target.family}-optimized-runtime-prompt",
            details={
                "family": target.family,
                "model_reference": target.model_reference,
                "model_path": target.model_path,
                "reason": "model is not materialized locally",
            },
        )
        return _runtime_target_payload(target, generic, optimized)

    generic = measure_runtime_probe(
        CommandBenchmarkSpec(
            name=f"{target.family}-generic-runtime-prompt",
            command=build_runtime_probe_command(
                repo_root,
                target.model_reference,
                models_dir=models_dir,
                device=device,
                backend="transformers-generic",
                use_specialization=False,
            ),
            timeout_seconds=180.0,
        ),
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        cwd=repo_root,
    )
    optimized = measure_runtime_probe(
        CommandBenchmarkSpec(
            name=f"{target.family}-optimized-runtime-prompt",
            command=build_runtime_probe_command(
                repo_root,
                target.model_reference,
                models_dir=models_dir,
                device=device,
                backend="optimized-native",
                use_specialization=True,
            ),
            timeout_seconds=180.0,
        ),
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        cwd=repo_root,
    )
    return _runtime_target_payload(target, generic, optimized)


def create_tiny_t5_fixture(root: Path) -> Path:
    model_dir = root / "tiny-t5"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "<pad>": 0,
        "</s>": 1,
        "<unk>": 2,
        "hello": 3,
        "world": 4,
        "benchmark": 5,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast_tokenizer.save_pretrained(model_dir)
    config = T5Config(
        vocab_size=len(vocab),
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        decoder_start_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        T5ForConditionalGeneration(config).save_pretrained(
            model_dir, safe_serialization=True
        )
    return model_dir


def run_runtime_probe(
    *,
    model_reference: str,
    models_dir: Path,
    device: str,
    backend: str,
    use_specialization: bool,
    prompt: str,
    max_new_tokens: int,
) -> RuntimeProbeResult:
    runtime_config = RuntimeConfig(
        model_reference=model_reference,
        models_dir=models_dir.expanduser().resolve(),
        device=device,
        backend=backend,
        use_specialization=use_specialization,
        use_cache=False,
    )
    generation_config = GenerationConfig(
        stream=False,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    client = RuntimeClient()
    executor = client.runtime_executor

    _reset_accelerator_metrics(device)
    load_started = time.perf_counter()
    runtime = client.load(runtime_config)
    _synchronize_runtime_device(runtime.config.device)
    load_ms = (time.perf_counter() - load_started) * 1000.0
    rss_after_load_mb = _current_rss_mb()
    accelerator_after_load = _capture_accelerator_metrics(runtime.config.device)

    request = PromptRequest(
        runtime_config=runtime_config,
        generation_config=generation_config,
        messages=[
            Message(
                role=MessageRole.SYSTEM,
                content=[ContentPart.text(DEFAULT_SYSTEM_PROMPT)],
            ),
            Message(role=MessageRole.USER, content=[ContentPart.text(prompt)]),
        ],
    )
    executor._validate_request(runtime, request)
    inputs = executor._build_inputs(runtime, request.messages)
    generate_kwargs = executor._build_generate_kwargs(runtime, request, None)

    generation_started = time.perf_counter()
    with torch.inference_mode():
        with suppress_module_prints(runtime.backend.print_suppression_modules):
            outputs = runtime.model.generate(**inputs, **generate_kwargs)
    _synchronize_runtime_device(runtime.config.device)
    generation_ms = (time.perf_counter() - generation_started) * 1000.0

    if hasattr(outputs, "detach"):
        outputs = outputs.detach()
    outputs = outputs.cpu()
    response_text = executor._decode_response(runtime, inputs, outputs)
    output_tokens = _count_output_tokens(runtime, inputs, outputs)
    tokens_per_second = None
    if generation_ms > 0:
        tokens_per_second = round(output_tokens / (generation_ms / 1000.0), 6)

    rss_after_generate_mb = _current_rss_mb()
    accelerator_after_generate = _capture_accelerator_metrics(runtime.config.device)
    return RuntimeProbeResult(
        load_ms=round(load_ms, 6),
        generation_ms=round(generation_ms, 6),
        total_ms=round(load_ms + generation_ms, 6),
        output_tokens=output_tokens,
        output_tokens_per_second=tokens_per_second,
        rss_after_load_mb=rss_after_load_mb,
        rss_after_generate_mb=rss_after_generate_mb,
        accelerator_kind=cast(
            str | None, accelerator_after_generate["accelerator_kind"]
        ),
        accelerator_current_after_load_mb=cast(
            float | None, accelerator_after_load["accelerator_current_mb"]
        ),
        accelerator_current_after_generate_mb=cast(
            float | None, accelerator_after_generate["accelerator_current_mb"]
        ),
        accelerator_peak_mb=cast(
            float | None, accelerator_after_generate["accelerator_peak_mb"]
        ),
        accelerator_reserved_after_load_mb=cast(
            float | None, accelerator_after_load["accelerator_reserved_mb"]
        ),
        accelerator_reserved_after_generate_mb=cast(
            float | None, accelerator_after_generate["accelerator_reserved_mb"]
        ),
        accelerator_peak_reserved_mb=cast(
            float | None, accelerator_after_generate["accelerator_peak_reserved_mb"]
        ),
        text_excerpt=_clip_text(response_text, max_chars=120),
    )


def render_runtime_probe_json(probe: RuntimeProbeResult) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def render_report_json(report: RuntimeBenchmarkReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def unavailable_measurement(
    name: str, *, details: dict[str, object]
) -> BenchmarkMeasurement:
    return BenchmarkMeasurement(
        name=name, status="unavailable", stats=None, details=details
    )


def _command_failure_measurement(
    spec: CommandBenchmarkSpec,
    result: CommandExecutionResult,
    warmup_iterations: int,
    *,
    warmup_only: bool,
) -> BenchmarkMeasurement:
    details: dict[str, object] = {
        "command": list(spec.command),
        "returncode": result.returncode,
        "stdout_excerpt": _clip_text(result.stdout),
        "stderr_excerpt": _clip_text(result.stderr),
        "timed_out": result.timed_out,
        "warmup_only": warmup_only,
    }
    return BenchmarkMeasurement(
        name=spec.name, status="unavailable", stats=None, details=details
    )


def _probe_parse_failure_measurement(
    spec: CommandBenchmarkSpec,
    result: CommandExecutionResult,
    warmup_iterations: int,
    *,
    warmup_only: bool,
) -> BenchmarkMeasurement:
    details: dict[str, object] = {
        "command": list(spec.command),
        "stdout_excerpt": _clip_text(result.stdout),
        "stderr_excerpt": _clip_text(result.stderr),
        "timed_out": result.timed_out,
        "warmup_only": warmup_only,
        "reason": "runtime probe did not emit valid JSON",
    }
    return BenchmarkMeasurement(
        name=spec.name, status="unavailable", stats=None, details=details
    )


def _summarize_samples(
    samples_ms: list[float], iterations: int, warmup_iterations: int
) -> BenchmarkStats:
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


def _summarize_numeric_values(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    p95_index = max(0, int(round((len(sorted_values) - 1) * 0.95)))
    return {
        "min": round(sorted_values[0], 6),
        "median": round(statistics.median(sorted_values), 6),
        "p95": round(sorted_values[p95_index], 6),
        "max": round(sorted_values[-1], 6),
        "mean": round(statistics.fmean(sorted_values), 6),
    }


def _summarize_optional_values(values: list[float | None]) -> dict[str, float] | None:
    filtered_values = [value for value in values if value is not None]
    if not filtered_values:
        return None
    return _summarize_numeric_values(filtered_values)


def _build_runtime_probe_details(
    spec: CommandBenchmarkSpec, samples: list[RuntimeProbeResult]
) -> dict[str, object]:
    accelerator_kinds = {
        sample.accelerator_kind for sample in samples if sample.accelerator_kind
    }
    accelerator_kind = None if len(accelerator_kinds) != 1 else accelerator_kinds.pop()
    return {
        "command": list(spec.command),
        "text_excerpt": samples[-1].text_excerpt,
        "metrics": {
            "latency_ms": {
                "load": _summarize_numeric_values(
                    [sample.load_ms for sample in samples]
                ),
                "generation": _summarize_numeric_values(
                    [sample.generation_ms for sample in samples]
                ),
            },
            "throughput": {
                "output_tokens": _summarize_numeric_values(
                    [float(sample.output_tokens) for sample in samples]
                ),
                "output_tokens_per_second": _summarize_optional_values(
                    [sample.output_tokens_per_second for sample in samples]
                ),
            },
            "memory": {
                "rss_after_load_mb": _summarize_optional_values(
                    [sample.rss_after_load_mb for sample in samples]
                ),
                "rss_after_generate_mb": _summarize_optional_values(
                    [sample.rss_after_generate_mb for sample in samples]
                ),
                "accelerator_kind": accelerator_kind,
                "accelerator_current_after_load_mb": _summarize_optional_values(
                    [sample.accelerator_current_after_load_mb for sample in samples]
                ),
                "accelerator_current_after_generate_mb": _summarize_optional_values(
                    [sample.accelerator_current_after_generate_mb for sample in samples]
                ),
                "accelerator_peak_mb": _summarize_optional_values(
                    [sample.accelerator_peak_mb for sample in samples]
                ),
                "accelerator_reserved_after_load_mb": _summarize_optional_values(
                    [sample.accelerator_reserved_after_load_mb for sample in samples]
                ),
                "accelerator_reserved_after_generate_mb": _summarize_optional_values(
                    [
                        sample.accelerator_reserved_after_generate_mb
                        for sample in samples
                    ]
                ),
                "accelerator_peak_reserved_mb": _summarize_optional_values(
                    [sample.accelerator_peak_reserved_mb for sample in samples]
                ),
            },
        },
    }


def _mean_delta_ms(
    left: BenchmarkMeasurement, right: BenchmarkMeasurement
) -> float | None:
    if left.stats is None or right.stats is None:
        return None
    return round(left.stats.mean_ms - right.stats.mean_ms, 6)


def _parse_runtime_probe_result(stdout: str) -> RuntimeProbeResult:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("runtime probe output was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("runtime probe output must be a JSON object")

    return RuntimeProbeResult(
        load_ms=_require_float(payload, "load_ms"),
        generation_ms=_require_float(payload, "generation_ms"),
        total_ms=_require_float(payload, "total_ms"),
        output_tokens=_require_int(payload, "output_tokens"),
        output_tokens_per_second=_optional_float(payload, "output_tokens_per_second"),
        rss_after_load_mb=_optional_float(payload, "rss_after_load_mb"),
        rss_after_generate_mb=_optional_float(payload, "rss_after_generate_mb"),
        accelerator_kind=_optional_string(payload, "accelerator_kind"),
        accelerator_current_after_load_mb=_optional_float(
            payload, "accelerator_current_after_load_mb"
        ),
        accelerator_current_after_generate_mb=_optional_float(
            payload, "accelerator_current_after_generate_mb"
        ),
        accelerator_peak_mb=_optional_float(payload, "accelerator_peak_mb"),
        accelerator_reserved_after_load_mb=_optional_float(
            payload, "accelerator_reserved_after_load_mb"
        ),
        accelerator_reserved_after_generate_mb=_optional_float(
            payload, "accelerator_reserved_after_generate_mb"
        ),
        accelerator_peak_reserved_mb=_optional_float(
            payload, "accelerator_peak_reserved_mb"
        ),
        text_excerpt=_require_string(payload, "text_excerpt"),
    )


def _require_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"runtime probe field '{key}' must be numeric")


def _optional_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"runtime probe field '{key}' must be numeric or null")


def _require_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int):
        return value
    raise ValueError(f"runtime probe field '{key}' must be an integer")


def _require_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    raise ValueError(f"runtime probe field '{key}' must be a string")


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"runtime probe field '{key}' must be a string or null")


def _current_rss_mb() -> float | None:
    if sys.platform == "win32":
        return None
    try:
        completed = subprocess.run(
            ("ps", "-o", "rss=", "-p", str(os.getpid())),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    if not value:
        return None
    return round(int(value) / 1024.0, 6)


def _reset_accelerator_metrics(device: str) -> None:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(resolved_device)


def _capture_accelerator_metrics(device: str) -> dict[str, object]:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved_device)
        return {
            "accelerator_kind": "cuda",
            "accelerator_current_mb": _bytes_to_mb(
                float(torch.cuda.memory_allocated(resolved_device))
            ),
            "accelerator_peak_mb": _bytes_to_mb(
                float(torch.cuda.max_memory_allocated(resolved_device))
            ),
            "accelerator_reserved_mb": _bytes_to_mb(
                float(torch.cuda.memory_reserved(resolved_device))
            ),
            "accelerator_peak_reserved_mb": _bytes_to_mb(
                float(torch.cuda.max_memory_reserved(resolved_device))
            ),
        }
    if (
        resolved_device.type == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        current_allocated = _mps_memory_stat("current_allocated_memory")
        driver_allocated = _mps_memory_stat("driver_allocated_memory")
        return {
            "accelerator_kind": "mps",
            "accelerator_current_mb": current_allocated,
            "accelerator_peak_mb": None,
            "accelerator_reserved_mb": driver_allocated,
            "accelerator_peak_reserved_mb": None,
        }
    return {
        "accelerator_kind": None,
        "accelerator_current_mb": None,
        "accelerator_peak_mb": None,
        "accelerator_reserved_mb": None,
        "accelerator_peak_reserved_mb": None,
    }


def _mps_memory_stat(name: str) -> float | None:
    if not hasattr(torch, "mps"):
        return None
    function = getattr(torch.mps, name, None)
    if function is None or not callable(function):
        return None
    try:
        value = function()
    except RuntimeError:
        return None
    if not isinstance(value, int | float):
        return None
    return _bytes_to_mb(float(value))


def _bytes_to_mb(value: float) -> float:
    return round(value / (1024.0 * 1024.0), 6)


def _synchronize_runtime_device(device: str) -> None:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved_device)
        return
    if (
        resolved_device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "synchronize")
    ):
        torch.mps.synchronize()


def _count_output_tokens(
    runtime, inputs: dict[str, object], outputs: torch.Tensor
) -> int:
    input_ids = cast(torch.Tensor, inputs["input_ids"])
    if runtime.plan.generic_model_kind is GenericModelKind.SEQ2SEQ_LM:
        return int(outputs.shape[-1])
    return max(0, int(outputs.shape[-1] - input_ids.shape[-1]))


def _runtime_comparison_unavailable_reason(
    generic: BenchmarkMeasurement,
    optimized: BenchmarkMeasurement,
) -> str:
    if generic.status != "measured" and optimized.status != "measured":
        return "Neither runtime benchmark completed successfully on this host"
    if generic.status != "measured":
        return "The generic benchmark did not complete successfully on this host"
    return "The optimized-native benchmark did not complete successfully on this host"


def _runtime_target_payload(
    target: RuntimeComparisonTarget,
    generic: BenchmarkMeasurement,
    optimized: BenchmarkMeasurement,
) -> dict[str, object]:
    comparison_available = (
        generic.status == "measured" and optimized.status == "measured"
    )
    speedup_ratio = None
    reason = None
    if comparison_available:
        assert generic.stats is not None
        assert optimized.stats is not None
        speedup_ratio = round(generic.stats.mean_ms / optimized.stats.mean_ms, 6)
    else:
        reason = _runtime_comparison_unavailable_reason(generic, optimized)
    return {
        "family": target.family,
        "model_reference": target.model_reference,
        "materialized": target.is_materialized,
        "model_path": target.model_path,
        "generic": generic.to_dict(),
        "optimized_native": optimized.to_dict(),
        "comparison_available": comparison_available,
        "speedup_ratio": speedup_ratio,
        "reason": reason,
    }


def _coerce_subprocess_output(output: str | bytes) -> str:
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _clip_text(text: str, *, max_chars: int = 400) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

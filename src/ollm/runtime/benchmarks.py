from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path
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

from ollm.client import RuntimeClient
from ollm.runtime.benchmark_probes import (
    RequestProbeMetrics,
    RuntimeProbeResult,
    parse_output_scaling_probe_result,
    parse_prompt_scaling_probe_result,
    parse_runtime_probe_result,
    parse_session_growth_probe_result,
    parse_warm_runtime_probe_result,
    render_output_scaling_probe_json,
    render_prompt_scaling_probe_json,
    render_runtime_probe_json,
    render_session_growth_probe_json,
    render_warm_runtime_probe_json,
    run_output_scaling_probe,
    run_prompt_scaling_probe,
    run_runtime_probe,
    run_session_growth_probe,
    run_warm_runtime_probe,
)
from ollm.runtime.benchmark_resources import (
    StageResourceSnapshot,
    summarize_optional_numeric_values,
)
from ollm.runtime.catalog import list_model_catalog
from ollm.runtime.config import RuntimeConfig

T = TypeVar("T")

DEFAULT_PROMPT_TOKEN_TARGETS = (32, 128, 512)
DEFAULT_OUTPUT_TOKEN_TARGETS = (16, 64, 128)
DEFAULT_SESSION_TURNS = 4

__all__ = [
    "BenchmarkMeasurement",
    "BenchmarkStats",
    "CommandBenchmarkSpec",
    "RuntimeBenchmarkReport",
    "RuntimeComparisonTarget",
    "benchmark_runtime_target",
    "build_current_supported_family_targets",
    "build_host_summary",
    "build_runtime_benchmark_report",
    "build_runtime_probe_command",
    "choose_default_device",
    "create_tiny_t5_fixture",
    "measure_callable",
    "measure_command",
    "measure_no_specialization_fallback_cost",
    "measure_runtime_probe",
    "render_output_scaling_probe_json",
    "render_prompt_scaling_probe_json",
    "render_report_json",
    "render_runtime_probe_json",
    "render_session_growth_probe_json",
    "render_warm_runtime_probe_json",
    "run_command",
    "run_output_scaling_probe",
    "run_prompt_scaling_probe",
    "run_runtime_probe",
    "run_session_growth_probe",
    "run_warm_runtime_probe",
    "unavailable_measurement",
]


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
            return _command_failure_measurement(spec, last_result, warmup_only=True)
    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        last_result = run_command(
            spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
        if last_result.returncode != 0:
            failure = _command_failure_measurement(spec, last_result, warmup_only=False)
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
        stats=_summarize_samples(samples_ms, iterations, warmup_iterations),
        details={
            "command": list(spec.command),
            "stdout_excerpt": _clip_text(
                "" if last_result is None else last_result.stdout
            ),
            "stderr_excerpt": _clip_text(
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
    if warmup_iterations != 0:
        raise ValueError("cold runtime probes do not support warmup iterations")
    probe_samples: list[RuntimeProbeResult] = []
    for _ in range(iterations):
        result = run_command(
            spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds
        )
        if result.returncode != 0:
            return _command_failure_measurement(spec, result, warmup_only=False)
        try:
            probe_samples.append(parse_runtime_probe_result(result.stdout))
        except ValueError:
            return _probe_parse_failure_measurement(
                spec,
                result,
                warmup_only=False,
                reason="runtime probe did not emit valid JSON",
            )
    samples_ms = [sample.load_ms + sample.request.total_ms for sample in probe_samples]
    return BenchmarkMeasurement(
        name=spec.name,
        status="measured",
        stats=_summarize_samples(samples_ms, iterations, warmup_iterations),
        details=_build_cold_probe_details(spec, probe_samples),
    )


def build_runtime_benchmark_report(
    *,
    repo_root: Path,
    benchmark_model_reference: str,
    models_dir: Path,
    device: str,
    iterations: int = 5,
    warmup_iterations: int = 1,
    prompt_token_targets: tuple[int, ...] = DEFAULT_PROMPT_TOKEN_TARGETS,
    output_token_targets: tuple[int, ...] = DEFAULT_OUTPUT_TOKEN_TARGETS,
    session_turns: int = DEFAULT_SESSION_TURNS,
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
        prompt_token_targets=prompt_token_targets,
        output_token_targets=output_token_targets,
        session_turns=session_turns,
    )
    return RuntimeBenchmarkReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        host=build_host_summary(),
        benchmark_model_reference=benchmark_model_reference,
        device=device,
        specialization_planner_overhead={
            "specialization_enabled": planner_enabled.to_dict(),
            "specialization_disabled": planner_disabled.to_dict(),
            "mean_delta_ms": _mean_delta_ms(planner_enabled, planner_disabled),
        },
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
    prompt_token_targets: tuple[int, ...],
    output_token_targets: tuple[int, ...],
    session_turns: int,
) -> dict[str, object]:
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
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        include_extended_scenarios=True,
        prompt_token_targets=prompt_token_targets,
        output_token_targets=output_token_targets,
        session_turns=session_turns,
    )
    family_results: list[dict[str, object]] = [
        benchmark_runtime_target(
            repo_root=repo_root,
            target=target,
            models_dir=models_dir,
            device=device,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            include_extended_scenarios=False,
            prompt_token_targets=prompt_token_targets,
            output_token_targets=output_token_targets,
            session_turns=session_turns,
        )
        for target in build_current_supported_family_targets(models_dir)
    ]
    return {
        "primary_target": primary_target,
        "family_results": family_results,
        "all_family_cold_start_comparisons_available": all(
            cast(dict[str, object], result["cold_start"])["comparison_available"]
            for result in family_results
        ),
        "all_family_warm_runtime_comparisons_available": all(
            cast(dict[str, object], result["warm_runtime"])["comparison_available"]
            for result in family_results
        ),
    }


def build_runtime_probe_command(
    repo_root: Path,
    model_reference: str,
    *,
    models_dir: Path,
    device: str,
    backend: str,
    use_specialization: bool,
    probe_mode: str = "cold",
    prompt: str = "Say hi.",
    max_new_tokens: int = 4,
    iterations: int = 1,
    warmup_iterations: int = 0,
    prompt_token_targets: tuple[int, ...] = DEFAULT_PROMPT_TOKEN_TARGETS,
    output_token_targets: tuple[int, ...] = DEFAULT_OUTPUT_TOKEN_TARGETS,
    session_turns: int = DEFAULT_SESSION_TURNS,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str((repo_root / "scripts" / "benchmark_runtime.py").resolve()),
        "--probe-runtime",
        "--probe-mode",
        probe_mode,
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
        "--probe-iterations",
        str(iterations),
        "--probe-warmup-iterations",
        str(warmup_iterations),
        "--probe-prompt-token-targets",
        ",".join(str(value) for value in prompt_token_targets),
        "--probe-output-token-targets",
        ",".join(str(value) for value in output_token_targets),
        "--probe-session-turns",
        str(session_turns),
    ]
    if not use_specialization:
        command.append("--probe-no-specialization")
    return tuple(command)


def build_current_supported_family_targets(
    models_dir: Path,
) -> tuple[RuntimeComparisonTarget, ...]:
    client = RuntimeClient()
    models_root = models_dir.expanduser().resolve()
    targets_by_family: dict[str, RuntimeComparisonTarget] = {}
    for entry in list_model_catalog():
        resolved_model = client.resolve(entry.model_id, models_root)
        if resolved_model.native_family is None:
            continue
        family_name = resolved_model.native_family.value
        model_path = resolved_model.model_path
        candidate = RuntimeComparisonTarget(
            family=family_name,
            model_reference=entry.model_id,
            is_materialized=model_path is not None and model_path.exists(),
            model_path=None if model_path is None else str(model_path),
        )
        existing = targets_by_family.get(family_name)
        if existing is None or (
            candidate.is_materialized and not existing.is_materialized
        ):
            targets_by_family[family_name] = candidate
    return tuple(
        targets_by_family[family_name] for family_name in sorted(targets_by_family)
    )


def benchmark_runtime_target(
    *,
    repo_root: Path,
    target: RuntimeComparisonTarget,
    models_dir: Path,
    device: str,
    iterations: int,
    warmup_iterations: int,
    include_extended_scenarios: bool = False,
    prompt_token_targets: tuple[int, ...] = DEFAULT_PROMPT_TOKEN_TARGETS,
    output_token_targets: tuple[int, ...] = DEFAULT_OUTPUT_TOKEN_TARGETS,
    session_turns: int = DEFAULT_SESSION_TURNS,
) -> dict[str, object]:
    client = RuntimeClient()
    if not target.is_materialized:
        unavailable = unavailable_measurement(
            name=f"{target.family}-runtime",
            details={
                "family": target.family,
                "model_reference": target.model_reference,
                "model_path": target.model_path,
                "reason": "model is not materialized locally",
            },
        )
        payload: dict[str, object] = {
            "family": target.family,
            "model_reference": target.model_reference,
            "materialized": target.is_materialized,
            "model_path": target.model_path,
            "cold_start": _backend_pair_payload(unavailable, unavailable),
            "warm_runtime": _backend_pair_payload(unavailable, unavailable),
        }
        if include_extended_scenarios:
            payload["prompt_length_scaling"] = {
                "generic": unavailable.to_dict(),
                "optimized_native": unavailable.to_dict(),
            }
            payload["output_length_scaling"] = {
                "generic": unavailable.to_dict(),
                "optimized_native": unavailable.to_dict(),
            }
            payload["session_growth"] = {
                "generic": unavailable.to_dict(),
                "optimized_native": unavailable.to_dict(),
            }
        return payload
    cold_generic = measure_runtime_probe(
        CommandBenchmarkSpec(
            name=f"{target.family}-generic-cold-start",
            command=build_runtime_probe_command(
                repo_root,
                target.model_reference,
                models_dir=models_dir,
                device=device,
                backend="transformers-generic",
                use_specialization=False,
                probe_mode="cold",
                iterations=max(1, iterations),
            ),
            timeout_seconds=240.0,
        ),
        iterations=max(1, iterations),
        warmup_iterations=0,
        cwd=repo_root,
    )
    cold_optimized = measure_runtime_probe(
        CommandBenchmarkSpec(
            name=f"{target.family}-optimized-cold-start",
            command=build_runtime_probe_command(
                repo_root,
                target.model_reference,
                models_dir=models_dir,
                device=device,
                backend="optimized-native",
                use_specialization=True,
                probe_mode="cold",
                iterations=max(1, iterations),
            ),
            timeout_seconds=240.0,
        ),
        iterations=max(1, iterations),
        warmup_iterations=0,
        cwd=repo_root,
    )
    warm_generic = _measure_warm_runtime_probe(
        CommandBenchmarkSpec(
            name=f"{target.family}-generic-warm-runtime",
            command=build_runtime_probe_command(
                repo_root,
                target.model_reference,
                models_dir=models_dir,
                device=device,
                backend="transformers-generic",
                use_specialization=False,
                probe_mode="warm",
                iterations=max(1, iterations),
                warmup_iterations=warmup_iterations,
            ),
            timeout_seconds=240.0,
        ),
        cwd=repo_root,
    )
    warm_optimized = _measure_warm_runtime_probe(
        CommandBenchmarkSpec(
            name=f"{target.family}-optimized-warm-runtime",
            command=build_runtime_probe_command(
                repo_root,
                target.model_reference,
                models_dir=models_dir,
                device=device,
                backend="optimized-native",
                use_specialization=True,
                probe_mode="warm",
                iterations=max(1, iterations),
                warmup_iterations=warmup_iterations,
            ),
            timeout_seconds=240.0,
        ),
        cwd=repo_root,
    )
    refreshed_target = _refresh_runtime_target(
        client=client,
        target=target,
        models_dir=models_dir,
    )
    payload: dict[str, object] = {
        "family": target.family,
        "model_reference": target.model_reference,
        "materialized": refreshed_target.is_materialized,
        "model_path": refreshed_target.model_path,
        "cold_start": _backend_pair_payload(cold_generic, cold_optimized),
        "warm_runtime": _backend_pair_payload(warm_generic, warm_optimized),
    }
    if include_extended_scenarios:
        payload["prompt_length_scaling"] = {
            "generic": _measure_prompt_scaling_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-generic-prompt-scaling",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="transformers-generic",
                        use_specialization=False,
                        probe_mode="prompt-scaling",
                        max_new_tokens=min(output_token_targets),
                        prompt_token_targets=prompt_token_targets,
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
            "optimized_native": _measure_prompt_scaling_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-optimized-prompt-scaling",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="optimized-native",
                        use_specialization=True,
                        probe_mode="prompt-scaling",
                        max_new_tokens=min(output_token_targets),
                        prompt_token_targets=prompt_token_targets,
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
        }
        payload["output_length_scaling"] = {
            "generic": _measure_output_scaling_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-generic-output-scaling",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="transformers-generic",
                        use_specialization=False,
                        probe_mode="output-scaling",
                        prompt="Explain KV cache in one sentence.",
                        output_token_targets=output_token_targets,
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
            "optimized_native": _measure_output_scaling_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-optimized-output-scaling",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="optimized-native",
                        use_specialization=True,
                        probe_mode="output-scaling",
                        prompt="Explain KV cache in one sentence.",
                        output_token_targets=output_token_targets,
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
        }
        payload["session_growth"] = {
            "generic": _measure_session_growth_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-generic-session-growth",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="transformers-generic",
                        use_specialization=False,
                        probe_mode="session-growth",
                        session_turns=session_turns,
                        max_new_tokens=min(output_token_targets),
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
            "optimized_native": _measure_session_growth_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-optimized-session-growth",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="optimized-native",
                        use_specialization=True,
                        probe_mode="session-growth",
                        session_turns=session_turns,
                        max_new_tokens=min(output_token_targets),
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
        }
    return payload


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


def render_report_json(report: RuntimeBenchmarkReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def unavailable_measurement(
    name: str, *, details: dict[str, object]
) -> BenchmarkMeasurement:
    return BenchmarkMeasurement(
        name=name, status="unavailable", stats=None, details=details
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
        stderr = timeout_message if not stderr else f"{stderr}\n{timeout_message}"
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


def _measure_warm_runtime_probe(
    spec: CommandBenchmarkSpec,
    *,
    cwd: Path,
) -> BenchmarkMeasurement:
    result = run_command(spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds)
    if result.returncode != 0:
        return _command_failure_measurement(spec, result, warmup_only=False)
    try:
        probe = parse_warm_runtime_probe_result(result.stdout)
    except ValueError:
        return _probe_parse_failure_measurement(
            spec,
            result,
            warmup_only=False,
            reason="warm runtime probe did not emit valid JSON",
        )
    samples = list(probe.measured_iterations)
    stats = _summarize_samples(
        [sample.total_ms for sample in samples],
        len(samples),
        probe.warmup_iterations,
    )
    return BenchmarkMeasurement(
        name=spec.name,
        status="measured",
        stats=stats,
        details={
            "command": list(spec.command),
            "runtime_load_ms": probe.runtime_load_ms,
            "runtime_load_resources": probe.runtime_load_resources.to_dict(),
            "warmup_iterations": probe.warmup_iterations,
            "metrics": _summarize_request_metrics(samples),
        },
    )


def _measure_prompt_scaling_probe(
    spec: CommandBenchmarkSpec,
    *,
    cwd: Path,
) -> BenchmarkMeasurement:
    result = run_command(spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds)
    if result.returncode != 0:
        return _command_failure_measurement(spec, result, warmup_only=False)
    try:
        probe = parse_prompt_scaling_probe_result(result.stdout)
    except ValueError:
        return _probe_parse_failure_measurement(
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


def _measure_output_scaling_probe(
    spec: CommandBenchmarkSpec,
    *,
    cwd: Path,
) -> BenchmarkMeasurement:
    result = run_command(spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds)
    if result.returncode != 0:
        return _command_failure_measurement(spec, result, warmup_only=False)
    try:
        probe = parse_output_scaling_probe_result(result.stdout)
    except ValueError:
        return _probe_parse_failure_measurement(
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


def _measure_session_growth_probe(
    spec: CommandBenchmarkSpec,
    *,
    cwd: Path,
) -> BenchmarkMeasurement:
    result = run_command(spec.command, cwd=cwd, timeout_seconds=spec.timeout_seconds)
    if result.returncode != 0:
        return _command_failure_measurement(spec, result, warmup_only=False)
    try:
        probe = parse_session_growth_probe_result(result.stdout)
    except ValueError:
        return _probe_parse_failure_measurement(
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


def _build_cold_probe_details(
    spec: CommandBenchmarkSpec,
    samples: list[RuntimeProbeResult],
) -> dict[str, object]:
    return {
        "command": list(spec.command),
        "load": {
            "latency_ms": _summarize_numeric_values(
                [sample.load_ms for sample in samples]
            ),
            "resources": _summarize_stage_resources(
                [sample.load_resources for sample in samples]
            ),
        },
        "metrics": _summarize_request_metrics([sample.request for sample in samples]),
        "text_excerpt": samples[-1].request.text_excerpt,
    }


def _summarize_request_metrics(
    samples: list[RequestProbeMetrics],
) -> dict[str, object]:
    return {
        "latency_ms": {
            "total": _summarize_numeric_values([sample.total_ms for sample in samples]),
            "generation": _summarize_numeric_values(
                [sample.generation_ms for sample in samples]
            ),
            "time_to_first_token": _optional_summary_dict(
                [
                    sample.time_to_first_token_ms
                    for sample in samples
                    if sample.time_to_first_token_ms is not None
                ]
            ),
            "inter_token_latency": _optional_summary_dict(
                [
                    latency
                    for sample in samples
                    for latency in sample.inter_token_latencies_ms
                ]
            ),
        },
        "throughput": {
            "prompt_tokens": _summarize_numeric_values(
                [float(sample.prompt_tokens) for sample in samples]
            ),
            "prompt_tokens_per_second": _optional_summary_dict(
                [
                    sample.prompt_tokens_per_second
                    for sample in samples
                    if sample.prompt_tokens_per_second is not None
                ]
            ),
            "output_tokens": _summarize_numeric_values(
                [float(sample.output_tokens) for sample in samples]
            ),
            "output_tokens_per_second": _optional_summary_dict(
                [
                    sample.output_tokens_per_second
                    for sample in samples
                    if sample.output_tokens_per_second is not None
                ]
            ),
        },
        "memory": _summarize_stage_resources([sample.resources for sample in samples]),
        "cache": {
            "cache_mode": _single_optional_string(
                [sample.cache_mode for sample in samples]
            ),
            "cache_dir_size_mb": _optional_summary_dict(
                [
                    sample.cache_dir_size_mb
                    for sample in samples
                    if sample.cache_dir_size_mb is not None
                ]
            ),
        },
        "allocator": {
            "allocator_gap_mb": _optional_summary_dict(
                [
                    sample.allocator_gap_mb
                    for sample in samples
                    if sample.allocator_gap_mb is not None
                ]
            ),
            "allocator_gap_ratio": _optional_summary_dict(
                [
                    sample.allocator_gap_ratio
                    for sample in samples
                    if sample.allocator_gap_ratio is not None
                ]
            ),
        },
    }


def _summarize_stage_resources(
    snapshots: list[StageResourceSnapshot],
) -> dict[str, object]:
    accelerator_utilizations = [
        snapshot.accelerator_utilization
        for snapshot in snapshots
        if snapshot.accelerator_utilization is not None
    ]
    gpu_utilization_means = [
        utilization.gpu_utilization_percent.mean
        for utilization in accelerator_utilizations
        if utilization.gpu_utilization_percent is not None
    ]
    memory_utilization_means = [
        utilization.memory_utilization_percent.mean
        for utilization in accelerator_utilizations
        if utilization.memory_utilization_percent is not None
    ]
    return {
        "current_rss_mb": _optional_summary_dict(
            [
                snapshot.current_rss_mb
                for snapshot in snapshots
                if snapshot.current_rss_mb is not None
            ]
        ),
        "peak_rss_mb": _optional_summary_dict(
            [
                snapshot.peak_rss_mb
                for snapshot in snapshots
                if snapshot.peak_rss_mb is not None
            ]
        ),
        "peak_rss_source": _single_optional_string(
            [
                snapshot.peak_rss_source
                for snapshot in snapshots
                if snapshot.peak_rss_source
            ]
        ),
        "accelerator_kind": _single_optional_string(
            [
                snapshot.accelerator_kind
                for snapshot in snapshots
                if snapshot.accelerator_kind
            ]
        ),
        "accelerator_current_mb": _optional_summary_dict(
            [
                snapshot.accelerator_current_mb
                for snapshot in snapshots
                if snapshot.accelerator_current_mb is not None
            ]
        ),
        "accelerator_peak_mb": _optional_summary_dict(
            [
                snapshot.accelerator_peak_mb
                for snapshot in snapshots
                if snapshot.accelerator_peak_mb is not None
            ]
        ),
        "accelerator_reserved_mb": _optional_summary_dict(
            [
                snapshot.accelerator_reserved_mb
                for snapshot in snapshots
                if snapshot.accelerator_reserved_mb is not None
            ]
        ),
        "accelerator_peak_reserved_mb": _optional_summary_dict(
            [
                snapshot.accelerator_peak_reserved_mb
                for snapshot in snapshots
                if snapshot.accelerator_peak_reserved_mb is not None
            ]
        ),
        "accelerator_peak_source": _single_optional_string(
            [
                snapshot.accelerator_peak_source
                for snapshot in snapshots
                if snapshot.accelerator_peak_source
            ]
        ),
        "process_cpu_utilization_percent": _optional_summary_dict(
            [
                snapshot.process_cpu_utilization_percent
                for snapshot in snapshots
                if snapshot.process_cpu_utilization_percent is not None
            ]
        ),
        "accelerator_utilization_percent": _optional_summary_dict(
            gpu_utilization_means
        ),
        "accelerator_memory_utilization_percent": _optional_summary_dict(
            memory_utilization_means
        ),
    }


def _backend_pair_payload(
    generic: BenchmarkMeasurement, optimized: BenchmarkMeasurement
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
        reason = _measurement_reason(generic)
        if reason is None:
            reason = _measurement_reason(optimized)
        if reason is None:
            reason = _runtime_comparison_unavailable_reason(generic, optimized)
    return {
        "generic": generic.to_dict(),
        "optimized_native": optimized.to_dict(),
        "comparison_available": comparison_available,
        "speedup_ratio": speedup_ratio,
        "reason": reason,
    }


def _command_failure_measurement(
    spec: CommandBenchmarkSpec,
    result: CommandExecutionResult,
    *,
    warmup_only: bool,
) -> BenchmarkMeasurement:
    raw_stderr_reason = _first_nonempty_line(result.stderr)
    raw_stdout_reason = _first_nonempty_line(result.stdout)
    stderr_excerpt = _clip_text(result.stderr)
    stdout_excerpt = _clip_text(result.stdout)
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
            "reason": _command_failure_reason(
                result=result,
                stderr_reason=raw_stderr_reason,
                stdout_reason=raw_stdout_reason,
            ),
        },
    )


def _probe_parse_failure_measurement(
    spec: CommandBenchmarkSpec,
    result: CommandExecutionResult,
    *,
    warmup_only: bool,
    reason: str,
) -> BenchmarkMeasurement:
    return BenchmarkMeasurement(
        name=spec.name,
        status="unavailable",
        stats=None,
        details={
            "command": list(spec.command),
            "stdout_excerpt": _clip_text(result.stdout),
            "stderr_excerpt": _clip_text(result.stderr),
            "timed_out": result.timed_out,
            "warmup_only": warmup_only,
            "reason": reason,
        },
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


def _single_optional_string(values: list[str]) -> str | None:
    if not values:
        return None
    if len(set(values)) != 1:
        return None
    return values[0]


def _optional_summary_dict(values: list[float]) -> dict[str, float] | None:
    summary = summarize_optional_numeric_values(values)
    if summary is None:
        return None
    return summary.to_dict()


def _mean_delta_ms(
    left: BenchmarkMeasurement, right: BenchmarkMeasurement
) -> float | None:
    if left.stats is None or right.stats is None:
        return None
    return round(left.stats.mean_ms - right.stats.mean_ms, 6)


def _coerce_subprocess_output(value: bytes | str) -> str:
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


def _clip_text(text: str, max_chars: int = 400) -> str:
    normalized_text = " ".join(text.split())
    if len(normalized_text) <= max_chars:
        return normalized_text
    return normalized_text[: max_chars - 3] + "..."


def _runtime_comparison_unavailable_reason(
    generic: BenchmarkMeasurement,
    optimized: BenchmarkMeasurement,
) -> str:
    if generic.status != "measured" and optimized.status != "measured":
        return "Neither runtime benchmark completed successfully on this host"
    if generic.status != "measured":
        return (
            "The generic runtime benchmark did not complete successfully on this host"
        )
    return "The optimized-native runtime benchmark did not complete successfully on this host"


def _measurement_reason(measurement: BenchmarkMeasurement) -> str | None:
    reason = measurement.details.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    return None


def _command_failure_reason(
    *,
    result: CommandExecutionResult,
    stderr_reason: str | None,
    stdout_reason: str | None,
) -> str:
    if result.timed_out:
        return f"Command timed out after benchmark timeout (returncode {result.returncode})"
    if stderr_reason is not None:
        return stderr_reason
    if stdout_reason is not None:
        return stdout_reason
    return f"Command failed with returncode {result.returncode}"


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _refresh_runtime_target(
    *,
    client: RuntimeClient,
    target: RuntimeComparisonTarget,
    models_dir: Path,
) -> RuntimeComparisonTarget:
    resolved_model = client.resolve(target.model_reference, models_dir)
    model_path = resolved_model.model_path
    return RuntimeComparisonTarget(
        family=target.family,
        model_reference=target.model_reference,
        is_materialized=model_path is not None and model_path.exists(),
        model_path=None if model_path is None else str(model_path),
    )

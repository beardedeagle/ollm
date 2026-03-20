import sys
from pathlib import Path
from typing import cast

import torch

import ollm.runtime.benchmark_targets as benchmark_targets_module
from ollm.app.types import ContentPart, Message, MessageRole
from ollm.client import RuntimeClient
from ollm.runtime.backends.base import BackendRuntime
from ollm.runtime.benchmark_probe_execution import (
    build_prompt_request,
    execute_request_probe,
)
from ollm.runtime.benchmark_probes import RuntimeProbeResult
from ollm.runtime.benchmark_types import (
    BenchmarkMeasurement,
    BenchmarkStats,
    RuntimeComparisonTarget,
    resolve_runtime_benchmark_profile,
)
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
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.catalog import ModelModality
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.loader import LoadedRuntime
from ollm.runtime.plan import RuntimePlan, SpecializationState
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel
from tests.benchmark_support import (
    build_request_probe_metrics,
    build_stage_resources,
)


class BenchmarkProcessorInputs(dict):
    def __init__(self):
        super().__init__(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "token_type_ids": torch.tensor([[0, 0, 0]]),
            }
        )

    def to(self, device, dtype=None):
        del device, dtype
        return self


class BenchmarkProcessor:
    def __init__(self):
        self.inputs = BenchmarkProcessorInputs()

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt,
        tokenize,
        return_dict,
        return_tensors,
    ):
        del messages, add_generation_prompt, tokenize, return_dict, return_tensors
        return self.inputs

    def batch_decode(self, outputs, skip_special_tokens=False):
        del outputs, skip_special_tokens
        return ["decoded-benchmark"]


class BenchmarkModel:
    def __init__(self):
        self.generate_kwargs: dict[str, object] = {}

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4]])


class BenchmarkTokenizer:
    def decode(self, tensor, skip_special_tokens=False):
        del tensor, skip_special_tokens
        return "decoded-benchmark"


def _build_processor_runtime() -> LoadedRuntime:
    config = RuntimeConfig(
        model_reference="gemma3-12B",
        models_dir=Path("models"),
        device="cpu",
        backend="optimized-native",
        stats=False,
    )
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("gemma3-12B"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="gemma3-12B",
        model_path=config.resolved_models_dir() / "gemma3-12B",
        repo_id="google/gemma-3-12b-it",
        revision=None,
        catalog_entry=None,
        capabilities=CapabilityProfile(
            support_level=SupportLevel.OPTIMIZED,
            modalities=(ModelModality.TEXT, ModelModality.IMAGE),
            requires_processor=True,
            supports_disk_cache=True,
        ),
        native_family=None,
        resolution_message="gemma test",
        architecture="Gemma3ForConditionalGeneration",
        model_type="gemma3",
        generic_model_kind=None,
    )
    plan = RuntimePlan(
        resolved_model=resolved_model,
        backend_id="optimized-native",
        model_path=resolved_model.model_path,
        support_level=SupportLevel.OPTIMIZED,
        generic_model_kind=None,
        supports_disk_cache=True,
        supports_cpu_offload=True,
        supports_gpu_offload=False,
        specialization_enabled=True,
        specialization_applied=False,
        specialization_provider_id="gemma3-native",
        specialization_state=SpecializationState.PLANNED,
        reason="benchmark test",
    )
    return LoadedRuntime(
        resolved_model=resolved_model,
        config=config,
        plan=plan,
        backend=BackendRuntime(
            backend_id="optimized-native",
            model=BenchmarkModel(),
            tokenizer=BenchmarkTokenizer(),
            processor=BenchmarkProcessor(),
            device=torch.device("cpu"),
            stats=None,
            print_suppression_modules=(),
            create_cache=lambda cache_dir, cache_strategy=None, cache_lifecycle=None: (
                None
            ),
            apply_offload=lambda runtime_config: None,
        ),
        model_path=resolved_model.model_path,
    )


def test_execute_request_probe_strips_processor_token_type_ids() -> None:
    runtime = _build_processor_runtime()
    request = build_prompt_request(
        runtime_config=runtime.config,
        generation_config=GenerationConfig(stream=False, max_new_tokens=1),
        messages=[
            Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
        ],
    )

    execution = execute_request_probe(runtime=runtime, request=request)

    assert execution.response_text == "decoded-benchmark"
    assert "token_type_ids" not in runtime.model.generate_kwargs


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
        kv_cache_strategy="streamed-segmented",
    )

    assert "--probe-backend" in command
    assert "transformers-generic" in command
    assert "--probe-kv-cache-strategy" in command
    assert "streamed-segmented" in command
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

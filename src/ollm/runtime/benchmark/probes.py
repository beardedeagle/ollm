"""Runtime benchmark probe execution entrypoints."""

from pathlib import Path
from typing import cast

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.client import RuntimeClient
from ollm.runtime.benchmark.probe_execution import (
    _clear_backend_stats,
    build_prompt_request,
    build_scaling_prompt,
    execute_request_probe,
)
from ollm.runtime.benchmark.probe_reopen import run_reopen_session_growth_probe
from ollm.runtime.benchmark.probe_serialization import (
    parse_output_scaling_probe_result,
    parse_prompt_scaling_probe_result,
    parse_reopen_session_growth_probe_result,
    parse_runtime_probe_result,
    parse_session_growth_probe_result,
    parse_warm_runtime_probe_result,
    render_output_scaling_probe_json,
    render_prompt_scaling_probe_json,
    render_reopen_session_growth_probe_json,
    render_runtime_probe_json,
    render_session_growth_probe_json,
    render_warm_runtime_probe_json,
)
from ollm.runtime.benchmark.probe_types import (
    EventTimingSummary,
    NativeRuntimeProfile,
    OutputScalingCase,
    OutputScalingProbeResult,
    PromptScalingCase,
    PromptScalingProbeResult,
    ReopenSessionGrowthProbeResult,
    ReopenSessionGrowthTurn,
    RequestProbeExecution,
    RequestProbeMetrics,
    RuntimeProbeResult,
    SessionGrowthProbeResult,
    SessionGrowthTurn,
    WarmRuntimeProbeResult,
)
from ollm.runtime.benchmark.resources import measure_stage
from ollm.runtime.config import DEFAULT_SYSTEM_PROMPT, GenerationConfig, RuntimeConfig
from ollm.runtime.loader import LoadedRuntime
from ollm.runtime.strategy_selector import DEFAULT_STRATEGY_SELECTOR_PROFILE

__all__ = [
    "EventTimingSummary",
    "NativeRuntimeProfile",
    "OutputScalingCase",
    "OutputScalingProbeResult",
    "PromptScalingCase",
    "PromptScalingProbeResult",
    "ReopenSessionGrowthProbeResult",
    "ReopenSessionGrowthTurn",
    "RequestProbeExecution",
    "RequestProbeMetrics",
    "RuntimeProbeResult",
    "SessionGrowthProbeResult",
    "SessionGrowthTurn",
    "WarmRuntimeProbeResult",
    "parse_output_scaling_probe_result",
    "parse_prompt_scaling_probe_result",
    "parse_reopen_session_growth_probe_result",
    "parse_runtime_probe_result",
    "parse_session_growth_probe_result",
    "parse_warm_runtime_probe_result",
    "render_output_scaling_probe_json",
    "render_prompt_scaling_probe_json",
    "render_reopen_session_growth_probe_json",
    "render_runtime_probe_json",
    "render_session_growth_probe_json",
    "render_warm_runtime_probe_json",
    "run_output_scaling_probe",
    "run_prompt_scaling_probe",
    "run_runtime_probe",
    "run_session_growth_probe",
    "run_warm_runtime_probe",
    "run_reopen_session_growth_probe",
]


def _build_probe_runtime_config(
    *,
    model_reference: str,
    models_dir: Path,
    device: str,
    backend: str,
    use_specialization: bool,
    kv_cache_strategy: str | None,
    strategy_selector_profile: str,
    kv_cache_window_tokens: int | None,
    offload_cpu_layers: int,
    offload_cpu_policy: str,
    offload_gpu_layers: int,
) -> RuntimeConfig:
    return RuntimeConfig(
        model_reference=model_reference,
        models_dir=models_dir.expanduser().resolve(),
        device=device,
        backend=backend,
        use_specialization=use_specialization,
        use_cache=True,
        kv_cache_strategy=kv_cache_strategy,
        strategy_selector_profile=strategy_selector_profile,
        kv_cache_window_tokens=kv_cache_window_tokens,
        offload_cpu_layers=offload_cpu_layers,
        offload_cpu_policy=offload_cpu_policy,
        offload_gpu_layers=offload_gpu_layers,
        stats=True,
    )


def run_runtime_probe(
    *,
    model_reference: str,
    models_dir: Path,
    device: str,
    backend: str,
    use_specialization: bool,
    prompt: str,
    max_new_tokens: int,
    kv_cache_strategy: str | None = None,
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE,
    kv_cache_window_tokens: int | None = None,
    offload_cpu_layers: int = 0,
    offload_cpu_policy: str = "auto",
    offload_gpu_layers: int = 0,
) -> RuntimeProbeResult:
    runtime_config = _build_probe_runtime_config(
        model_reference=model_reference,
        models_dir=models_dir,
        device=device,
        backend=backend,
        use_specialization=use_specialization,
        kv_cache_strategy=kv_cache_strategy,
        strategy_selector_profile=strategy_selector_profile,
        kv_cache_window_tokens=kv_cache_window_tokens,
        offload_cpu_layers=offload_cpu_layers,
        offload_cpu_policy=offload_cpu_policy,
        offload_gpu_layers=offload_gpu_layers,
    )
    generation_config = GenerationConfig(
        stream=True,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    client = RuntimeClient()
    runtime_result = measure_stage(
        runtime_config.device,
        lambda: client.load(runtime_config),
    )
    runtime = cast(LoadedRuntime, runtime_result[0])
    _clear_backend_stats(runtime)
    execution = execute_request_probe(
        runtime=runtime,
        request=build_prompt_request(
            runtime_config=runtime.config,
            generation_config=generation_config,
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=[ContentPart.text(DEFAULT_SYSTEM_PROMPT)],
                ),
                Message(role=MessageRole.USER, content=[ContentPart.text(prompt)]),
            ],
        ),
    )
    return RuntimeProbeResult(
        load_ms=runtime_result[1],
        load_resources=runtime_result[2],
        request=execution.metrics,
    )


def run_warm_runtime_probe(
    *,
    model_reference: str,
    models_dir: Path,
    device: str,
    backend: str,
    use_specialization: bool,
    prompt: str,
    max_new_tokens: int,
    iterations: int,
    warmup_iterations: int,
    kv_cache_strategy: str | None = None,
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE,
    kv_cache_window_tokens: int | None = None,
    offload_cpu_layers: int = 0,
    offload_cpu_policy: str = "auto",
    offload_gpu_layers: int = 0,
) -> WarmRuntimeProbeResult:
    runtime_config = _build_probe_runtime_config(
        model_reference=model_reference,
        models_dir=models_dir,
        device=device,
        backend=backend,
        use_specialization=use_specialization,
        kv_cache_strategy=kv_cache_strategy,
        strategy_selector_profile=strategy_selector_profile,
        kv_cache_window_tokens=kv_cache_window_tokens,
        offload_cpu_layers=offload_cpu_layers,
        offload_cpu_policy=offload_cpu_policy,
        offload_gpu_layers=offload_gpu_layers,
    )
    generation_config = GenerationConfig(
        stream=True,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    client = RuntimeClient()
    runtime_result = measure_stage(
        runtime_config.device,
        lambda: client.load(runtime_config),
    )
    runtime = cast(LoadedRuntime, runtime_result[0])
    _clear_backend_stats(runtime)
    request = build_prompt_request(
        runtime_config=runtime.config,
        generation_config=generation_config,
        messages=[
            Message(
                role=MessageRole.SYSTEM,
                content=[ContentPart.text(DEFAULT_SYSTEM_PROMPT)],
            ),
            Message(role=MessageRole.USER, content=[ContentPart.text(prompt)]),
        ],
    )
    for _ in range(warmup_iterations):
        execute_request_probe(runtime=runtime, request=request)
    return WarmRuntimeProbeResult(
        runtime_load_ms=runtime_result[1],
        runtime_load_resources=runtime_result[2],
        warmup_iterations=warmup_iterations,
        measured_iterations=tuple(
            execute_request_probe(runtime=runtime, request=request).metrics
            for _ in range(iterations)
        ),
    )


def run_prompt_scaling_probe(
    *,
    model_reference: str,
    models_dir: Path,
    device: str,
    backend: str,
    use_specialization: bool,
    prompt_token_targets: tuple[int, ...],
    max_new_tokens: int,
    kv_cache_strategy: str | None = None,
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE,
    kv_cache_window_tokens: int | None = None,
    offload_cpu_layers: int = 0,
    offload_cpu_policy: str = "auto",
    offload_gpu_layers: int = 0,
) -> PromptScalingProbeResult:
    runtime_config = _build_probe_runtime_config(
        model_reference=model_reference,
        models_dir=models_dir,
        device=device,
        backend=backend,
        use_specialization=use_specialization,
        kv_cache_strategy=kv_cache_strategy,
        strategy_selector_profile=strategy_selector_profile,
        kv_cache_window_tokens=kv_cache_window_tokens,
        offload_cpu_layers=offload_cpu_layers,
        offload_cpu_policy=offload_cpu_policy,
        offload_gpu_layers=offload_gpu_layers,
    )
    generation_config = GenerationConfig(
        stream=True,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    client = RuntimeClient()
    runtime_result = measure_stage(
        runtime_config.device,
        lambda: client.load(runtime_config),
    )
    runtime = cast(LoadedRuntime, runtime_result[0])
    _clear_backend_stats(runtime)
    return PromptScalingProbeResult(
        runtime_load_ms=runtime_result[1],
        runtime_load_resources=runtime_result[2],
        cases=tuple(
            PromptScalingCase(
                requested_prompt_tokens=target,
                request=execute_request_probe(
                    runtime=runtime,
                    request=build_prompt_request(
                        runtime_config=runtime.config,
                        generation_config=generation_config,
                        messages=[
                            Message(
                                role=MessageRole.SYSTEM,
                                content=[ContentPart.text(DEFAULT_SYSTEM_PROMPT)],
                            ),
                            Message(
                                role=MessageRole.USER,
                                content=[
                                    ContentPart.text(
                                        build_scaling_prompt(
                                            target_prompt_tokens=target
                                        )
                                    )
                                ],
                            ),
                        ],
                    ),
                ).metrics,
            )
            for target in prompt_token_targets
        ),
    )


def run_output_scaling_probe(
    *,
    model_reference: str,
    models_dir: Path,
    device: str,
    backend: str,
    use_specialization: bool,
    prompt: str,
    output_token_targets: tuple[int, ...],
    kv_cache_strategy: str | None = None,
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE,
    kv_cache_window_tokens: int | None = None,
    offload_cpu_layers: int = 0,
    offload_cpu_policy: str = "auto",
    offload_gpu_layers: int = 0,
) -> OutputScalingProbeResult:
    runtime_config = _build_probe_runtime_config(
        model_reference=model_reference,
        models_dir=models_dir,
        device=device,
        backend=backend,
        use_specialization=use_specialization,
        kv_cache_strategy=kv_cache_strategy,
        strategy_selector_profile=strategy_selector_profile,
        kv_cache_window_tokens=kv_cache_window_tokens,
        offload_cpu_layers=offload_cpu_layers,
        offload_cpu_policy=offload_cpu_policy,
        offload_gpu_layers=offload_gpu_layers,
    )
    client = RuntimeClient()
    runtime_result = measure_stage(
        runtime_config.device,
        lambda: client.load(runtime_config),
    )
    runtime = cast(LoadedRuntime, runtime_result[0])
    _clear_backend_stats(runtime)
    return OutputScalingProbeResult(
        runtime_load_ms=runtime_result[1],
        runtime_load_resources=runtime_result[2],
        cases=tuple(
            OutputScalingCase(
                requested_max_new_tokens=target,
                request=execute_request_probe(
                    runtime=runtime,
                    request=build_prompt_request(
                        runtime_config=runtime.config,
                        generation_config=GenerationConfig(
                            stream=True,
                            max_new_tokens=target,
                            temperature=0.0,
                        ),
                        messages=[
                            Message(
                                role=MessageRole.SYSTEM,
                                content=[ContentPart.text(DEFAULT_SYSTEM_PROMPT)],
                            ),
                            Message(
                                role=MessageRole.USER,
                                content=[ContentPart.text(prompt)],
                            ),
                        ],
                    ),
                ).metrics,
            )
            for target in output_token_targets
        ),
    )


def run_session_growth_probe(
    *,
    model_reference: str,
    models_dir: Path,
    device: str,
    backend: str,
    use_specialization: bool,
    session_turns: int,
    max_new_tokens: int,
    kv_cache_strategy: str | None = None,
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE,
    kv_cache_window_tokens: int | None = None,
    offload_cpu_layers: int = 0,
    offload_cpu_policy: str = "auto",
    offload_gpu_layers: int = 0,
) -> SessionGrowthProbeResult:
    runtime_config = _build_probe_runtime_config(
        model_reference=model_reference,
        models_dir=models_dir,
        device=device,
        backend=backend,
        use_specialization=use_specialization,
        kv_cache_strategy=kv_cache_strategy,
        strategy_selector_profile=strategy_selector_profile,
        kv_cache_window_tokens=kv_cache_window_tokens,
        offload_cpu_layers=offload_cpu_layers,
        offload_cpu_policy=offload_cpu_policy,
        offload_gpu_layers=offload_gpu_layers,
    )
    generation_config = GenerationConfig(
        stream=True,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    client = RuntimeClient()
    runtime_result = measure_stage(
        runtime_config.device,
        lambda: client.load(runtime_config),
    )
    runtime = cast(LoadedRuntime, runtime_result[0])
    _clear_backend_stats(runtime)
    history: list[Message] = []
    turns: list[SessionGrowthTurn] = []
    for turn_index in range(1, session_turns + 1):
        user_message = Message(
            role=MessageRole.USER,
            content=[
                ContentPart.text(
                    f"Turn {turn_index}: summarize the benchmark status in one sentence."
                )
            ],
        )
        execution = execute_request_probe(
            runtime=runtime,
            request=build_prompt_request(
                runtime_config=runtime.config,
                generation_config=generation_config,
                messages=[
                    Message(
                        role=MessageRole.SYSTEM,
                        content=[ContentPart.text(DEFAULT_SYSTEM_PROMPT)],
                    ),
                    *history,
                    user_message,
                ],
            ),
        )
        history.append(user_message)
        history.append(Message.assistant_text(execution.response_text))
        turns.append(
            SessionGrowthTurn(turn_index=turn_index, request=execution.metrics)
        )
    return SessionGrowthProbeResult(
        runtime_load_ms=runtime_result[1],
        runtime_load_resources=runtime_result[2],
        turns=tuple(turns),
    )

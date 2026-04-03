"""Typed registry for benchmark probe modes and their contracts."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast

from ollm.runtime.benchmark.probe_serialization import (
    render_output_scaling_probe_json,
    render_prompt_scaling_probe_json,
    render_reopen_session_growth_probe_json,
    render_runtime_probe_json,
    render_session_growth_probe_json,
    render_warm_runtime_probe_json,
)
from ollm.runtime.benchmark.probe_types import (
    OutputScalingProbeResult,
    PromptScalingProbeResult,
    ReopenSessionGrowthProbeResult,
    RuntimeProbeResult,
    SessionGrowthProbeResult,
    WarmRuntimeProbeResult,
)
from ollm.runtime.benchmark.probes import (
    run_output_scaling_probe,
    run_prompt_scaling_probe,
    run_reopen_session_growth_probe,
    run_runtime_probe,
    run_session_growth_probe,
    run_warm_runtime_probe,
)
from ollm.runtime.benchmark.types import (
    DEFAULT_OUTPUT_TOKEN_TARGETS,
    DEFAULT_PROMPT_TOKEN_TARGETS,
    DEFAULT_SESSION_TURNS,
)
from ollm.runtime.strategy_selector import DEFAULT_STRATEGY_SELECTOR_PROFILE


class ProbeMode(str, Enum):
    COLD = "cold"
    WARM = "warm"
    PROMPT_SCALING = "prompt-scaling"
    OUTPUT_SCALING = "output-scaling"
    SESSION_GROWTH = "session-growth"
    REOPEN_SESSION_GROWTH = "reopen-session-growth"


@dataclass(frozen=True, slots=True)
class ProbeInvocation:
    model_reference: str
    models_dir: Path
    device: str
    backend: str
    use_specialization: bool
    prompt: str = "Say hi."
    max_new_tokens: int = 4
    kv_cache_strategy: str | None = None
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE
    kv_cache_window_tokens: int | None = None
    offload_cpu_layers: int = 0
    offload_cpu_policy: str = "auto"
    offload_gpu_layers: int = 0
    iterations: int = 1
    warmup_iterations: int = 0
    prompt_token_targets: tuple[int, ...] = DEFAULT_PROMPT_TOKEN_TARGETS
    output_token_targets: tuple[int, ...] = DEFAULT_OUTPUT_TOKEN_TARGETS
    session_turns: int = DEFAULT_SESSION_TURNS


@dataclass(frozen=True, slots=True)
class ProbeDefinition:
    mode: ProbeMode
    run_kind: str
    runner: Callable[[ProbeInvocation], object]
    renderer: Callable[[object], str]
    history_request_extractor: Callable[[Mapping[str, object]], Mapping[str, object] | None]


def probe_mode_choices() -> tuple[str, ...]:
    return tuple(mode.value for mode in ProbeMode)


def get_probe_definition(mode: ProbeMode) -> ProbeDefinition:
    return _PROBE_DEFINITIONS[mode]


def _run_cold_probe(invocation: ProbeInvocation) -> object:
    return run_runtime_probe(
        model_reference=invocation.model_reference,
        models_dir=invocation.models_dir,
        device=invocation.device,
        backend=invocation.backend,
        use_specialization=invocation.use_specialization,
        kv_cache_strategy=invocation.kv_cache_strategy,
        strategy_selector_profile=invocation.strategy_selector_profile,
        kv_cache_window_tokens=invocation.kv_cache_window_tokens,
        offload_cpu_layers=invocation.offload_cpu_layers,
        offload_cpu_policy=invocation.offload_cpu_policy,
        offload_gpu_layers=invocation.offload_gpu_layers,
        prompt=invocation.prompt,
        max_new_tokens=invocation.max_new_tokens,
    )


def _run_warm_probe(invocation: ProbeInvocation) -> object:
    return run_warm_runtime_probe(
        model_reference=invocation.model_reference,
        models_dir=invocation.models_dir,
        device=invocation.device,
        backend=invocation.backend,
        use_specialization=invocation.use_specialization,
        kv_cache_strategy=invocation.kv_cache_strategy,
        strategy_selector_profile=invocation.strategy_selector_profile,
        kv_cache_window_tokens=invocation.kv_cache_window_tokens,
        offload_cpu_layers=invocation.offload_cpu_layers,
        offload_cpu_policy=invocation.offload_cpu_policy,
        offload_gpu_layers=invocation.offload_gpu_layers,
        prompt=invocation.prompt,
        max_new_tokens=invocation.max_new_tokens,
        iterations=invocation.iterations,
        warmup_iterations=invocation.warmup_iterations,
    )


def _run_prompt_scaling_probe(invocation: ProbeInvocation) -> object:
    return run_prompt_scaling_probe(
        model_reference=invocation.model_reference,
        models_dir=invocation.models_dir,
        device=invocation.device,
        backend=invocation.backend,
        use_specialization=invocation.use_specialization,
        kv_cache_strategy=invocation.kv_cache_strategy,
        strategy_selector_profile=invocation.strategy_selector_profile,
        kv_cache_window_tokens=invocation.kv_cache_window_tokens,
        offload_cpu_layers=invocation.offload_cpu_layers,
        offload_cpu_policy=invocation.offload_cpu_policy,
        offload_gpu_layers=invocation.offload_gpu_layers,
        prompt_token_targets=invocation.prompt_token_targets,
        max_new_tokens=invocation.max_new_tokens,
    )


def _run_output_scaling_probe(invocation: ProbeInvocation) -> object:
    return run_output_scaling_probe(
        model_reference=invocation.model_reference,
        models_dir=invocation.models_dir,
        device=invocation.device,
        backend=invocation.backend,
        use_specialization=invocation.use_specialization,
        kv_cache_strategy=invocation.kv_cache_strategy,
        strategy_selector_profile=invocation.strategy_selector_profile,
        kv_cache_window_tokens=invocation.kv_cache_window_tokens,
        offload_cpu_layers=invocation.offload_cpu_layers,
        offload_cpu_policy=invocation.offload_cpu_policy,
        offload_gpu_layers=invocation.offload_gpu_layers,
        prompt=invocation.prompt,
        output_token_targets=invocation.output_token_targets,
    )


def _run_session_growth_probe(invocation: ProbeInvocation) -> object:
    return run_session_growth_probe(
        model_reference=invocation.model_reference,
        models_dir=invocation.models_dir,
        device=invocation.device,
        backend=invocation.backend,
        use_specialization=invocation.use_specialization,
        kv_cache_strategy=invocation.kv_cache_strategy,
        strategy_selector_profile=invocation.strategy_selector_profile,
        kv_cache_window_tokens=invocation.kv_cache_window_tokens,
        offload_cpu_layers=invocation.offload_cpu_layers,
        offload_cpu_policy=invocation.offload_cpu_policy,
        offload_gpu_layers=invocation.offload_gpu_layers,
        session_turns=invocation.session_turns,
        max_new_tokens=invocation.max_new_tokens,
    )


def _run_reopen_session_growth_probe(invocation: ProbeInvocation) -> object:
    return run_reopen_session_growth_probe(
        model_reference=invocation.model_reference,
        models_dir=invocation.models_dir,
        device=invocation.device,
        backend=invocation.backend,
        use_specialization=invocation.use_specialization,
        kv_cache_strategy=invocation.kv_cache_strategy,
        strategy_selector_profile=invocation.strategy_selector_profile,
        kv_cache_window_tokens=invocation.kv_cache_window_tokens,
        offload_cpu_layers=invocation.offload_cpu_layers,
        offload_cpu_policy=invocation.offload_cpu_policy,
        offload_gpu_layers=invocation.offload_gpu_layers,
        session_turns=invocation.session_turns,
        max_new_tokens=invocation.max_new_tokens,
    )


def _render_runtime_probe(value: object) -> str:
    return render_runtime_probe_json(cast(RuntimeProbeResult, value))


def _render_warm_probe(value: object) -> str:
    return render_warm_runtime_probe_json(cast(WarmRuntimeProbeResult, value))


def _render_prompt_scaling_probe(value: object) -> str:
    return render_prompt_scaling_probe_json(cast(PromptScalingProbeResult, value))


def _render_output_scaling_probe(value: object) -> str:
    return render_output_scaling_probe_json(cast(OutputScalingProbeResult, value))


def _render_session_growth_probe(value: object) -> str:
    return render_session_growth_probe_json(cast(SessionGrowthProbeResult, value))


def _render_reopen_session_growth_probe(value: object) -> str:
    return render_reopen_session_growth_probe_json(
        cast(ReopenSessionGrowthProbeResult, value)
    )


def _request_from_cold_probe(payload: Mapping[str, object]) -> Mapping[str, object] | None:
    return _optional_mapping(payload.get("request"))


def _request_from_warm_probe(payload: Mapping[str, object]) -> Mapping[str, object] | None:
    measured = payload.get("measured_iterations")
    if not isinstance(measured, list) or not measured:
        return None
    return _optional_mapping(measured[0])


def _request_from_scaling_probe(
    payload: Mapping[str, object],
) -> Mapping[str, object] | None:
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        return None
    first_case = _optional_mapping(cases[0])
    if first_case is None:
        return None
    return _optional_mapping(first_case.get("request"))


def _request_from_session_probe(
    payload: Mapping[str, object],
) -> Mapping[str, object] | None:
    turns = payload.get("turns")
    if not isinstance(turns, list) or not turns:
        return None
    first_turn = _optional_mapping(turns[0])
    if first_turn is None:
        return None
    return _optional_mapping(first_turn.get("request"))


def _optional_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, object], value)


_PROBE_DEFINITIONS = {
    ProbeMode.COLD: ProbeDefinition(
        mode=ProbeMode.COLD,
        run_kind="probe-cold",
        runner=_run_cold_probe,
        renderer=_render_runtime_probe,
        history_request_extractor=_request_from_cold_probe,
    ),
    ProbeMode.WARM: ProbeDefinition(
        mode=ProbeMode.WARM,
        run_kind="probe-warm",
        runner=_run_warm_probe,
        renderer=_render_warm_probe,
        history_request_extractor=_request_from_warm_probe,
    ),
    ProbeMode.PROMPT_SCALING: ProbeDefinition(
        mode=ProbeMode.PROMPT_SCALING,
        run_kind="probe-prompt-scaling",
        runner=_run_prompt_scaling_probe,
        renderer=_render_prompt_scaling_probe,
        history_request_extractor=_request_from_scaling_probe,
    ),
    ProbeMode.OUTPUT_SCALING: ProbeDefinition(
        mode=ProbeMode.OUTPUT_SCALING,
        run_kind="probe-output-scaling",
        runner=_run_output_scaling_probe,
        renderer=_render_output_scaling_probe,
        history_request_extractor=_request_from_scaling_probe,
    ),
    ProbeMode.SESSION_GROWTH: ProbeDefinition(
        mode=ProbeMode.SESSION_GROWTH,
        run_kind="probe-session-growth",
        runner=_run_session_growth_probe,
        renderer=_render_session_growth_probe,
        history_request_extractor=_request_from_session_probe,
    ),
    ProbeMode.REOPEN_SESSION_GROWTH: ProbeDefinition(
        mode=ProbeMode.REOPEN_SESSION_GROWTH,
        run_kind="probe-reopen-session-growth",
        runner=_run_reopen_session_growth_probe,
        renderer=_render_reopen_session_growth_probe,
        history_request_extractor=_request_from_session_probe,
    ),
}

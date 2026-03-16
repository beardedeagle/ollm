from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time
from typing import cast

import torch

from ollm.app.types import ContentPart, Message, MessageRole, PromptRequest
from ollm.client import RuntimeClient
from ollm.runtime.benchmark_resources import (
    AcceleratorUtilizationSnapshot,
    NumericSummary,
    StageResourceSnapshot,
    cache_dir_size_mb,
    measure_stage,
)
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import DEFAULT_SYSTEM_PROMPT, GenerationConfig, RuntimeConfig
from ollm.runtime.errors import PromptExecutionError
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import LoadedRuntime
from ollm.runtime.output_control import suppress_module_prints
from ollm.runtime.streaming import BufferedTextStreamer


@dataclass(frozen=True, slots=True)
class RequestProbeMetrics:
    total_ms: float
    generation_ms: float
    time_to_first_token_ms: float | None
    inter_token_latencies_ms: tuple[float, ...]
    prompt_tokens: int
    prompt_tokens_per_second: float | None
    output_tokens: int
    output_tokens_per_second: float | None
    cache_mode: str
    cache_dir_size_mb: float | None
    allocator_gap_mb: float | None
    allocator_gap_ratio: float | None
    resources: StageResourceSnapshot
    text_excerpt: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["resources"] = self.resources.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeProbeResult:
    load_ms: float
    load_resources: StageResourceSnapshot
    request: RequestProbeMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "load_ms": self.load_ms,
            "load_resources": self.load_resources.to_dict(),
            "request": self.request.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WarmRuntimeProbeResult:
    runtime_load_ms: float
    runtime_load_resources: StageResourceSnapshot
    warmup_iterations: int
    measured_iterations: tuple[RequestProbeMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_load_ms": self.runtime_load_ms,
            "runtime_load_resources": self.runtime_load_resources.to_dict(),
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": [
                metrics.to_dict() for metrics in self.measured_iterations
            ],
        }


@dataclass(frozen=True, slots=True)
class PromptScalingCase:
    requested_prompt_tokens: int
    request: RequestProbeMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_prompt_tokens": self.requested_prompt_tokens,
            "request": self.request.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PromptScalingProbeResult:
    runtime_load_ms: float
    runtime_load_resources: StageResourceSnapshot
    cases: tuple[PromptScalingCase, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_load_ms": self.runtime_load_ms,
            "runtime_load_resources": self.runtime_load_resources.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class OutputScalingCase:
    requested_max_new_tokens: int
    request: RequestProbeMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_max_new_tokens": self.requested_max_new_tokens,
            "request": self.request.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class OutputScalingProbeResult:
    runtime_load_ms: float
    runtime_load_resources: StageResourceSnapshot
    cases: tuple[OutputScalingCase, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_load_ms": self.runtime_load_ms,
            "runtime_load_resources": self.runtime_load_resources.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class SessionGrowthTurn:
    turn_index: int
    request: RequestProbeMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "request": self.request.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SessionGrowthProbeResult:
    runtime_load_ms: float
    runtime_load_resources: StageResourceSnapshot
    turns: tuple[SessionGrowthTurn, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_load_ms": self.runtime_load_ms,
            "runtime_load_resources": self.runtime_load_resources.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
        }


@dataclass(frozen=True, slots=True)
class _RequestProbeExecution:
    metrics: RequestProbeMetrics
    response_text: str


class TimedBufferedTextStreamer(BufferedTextStreamer):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer,
            sink=_NullStreamSink(),
            skip_prompt=True,
            skip_special_tokens=False,
        )
        self._token_timestamps: list[float] = []

    @property
    def token_timestamps(self) -> tuple[float, ...]:
        return tuple(self._token_timestamps)

    def put(self, value) -> None:
        tensor_value = value
        if hasattr(tensor_value, "shape"):
            if len(tensor_value.shape) > 1 and tensor_value.shape[0] > 1:
                raise ValueError("TimedBufferedTextStreamer only supports batch size 1")
            if len(tensor_value.shape) > 1:
                tensor_value = tensor_value[0]
        skip_prompt_tokens = bool(self.skip_prompt and self.next_tokens_are_prompt)
        if not skip_prompt_tokens:
            if isinstance(tensor_value, torch.Tensor):
                token_count = tensor_value.numel()
            else:
                token_count = len(list(tensor_value))
            now = time.perf_counter()
            self._token_timestamps.extend(now for _ in range(token_count))
        super().put(value)


class _NullStreamSink:
    def on_status(self, message: str) -> None:
        del message

    def on_text(self, text: str) -> None:
        del text

    def on_complete(self, text: str) -> None:
        del text


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
        use_cache=True,
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
    load_ms = runtime_result[1]
    load_resources = runtime_result[2]
    execution = _execute_request_probe(
        runtime=runtime,
        request=_build_prompt_request(
            runtime_config=runtime_config,
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
        load_ms=load_ms,
        load_resources=load_resources,
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
) -> WarmRuntimeProbeResult:
    runtime_config = RuntimeConfig(
        model_reference=model_reference,
        models_dir=models_dir.expanduser().resolve(),
        device=device,
        backend=backend,
        use_specialization=use_specialization,
        use_cache=True,
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
    load_ms = runtime_result[1]
    load_resources = runtime_result[2]
    request = _build_prompt_request(
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
    for _ in range(warmup_iterations):
        _execute_request_probe(runtime=runtime, request=request)
    measured_iterations = tuple(
        _execute_request_probe(runtime=runtime, request=request).metrics
        for _ in range(iterations)
    )
    return WarmRuntimeProbeResult(
        runtime_load_ms=load_ms,
        runtime_load_resources=load_resources,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
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
) -> PromptScalingProbeResult:
    runtime_config = RuntimeConfig(
        model_reference=model_reference,
        models_dir=models_dir.expanduser().resolve(),
        device=device,
        backend=backend,
        use_specialization=use_specialization,
        use_cache=True,
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
    load_ms = runtime_result[1]
    load_resources = runtime_result[2]
    cases = tuple(
        PromptScalingCase(
            requested_prompt_tokens=target,
            request=_execute_request_probe(
                runtime=runtime,
                request=_build_prompt_request(
                    runtime_config=runtime_config,
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
                                    _build_scaling_prompt(target_prompt_tokens=target)
                                )
                            ],
                        ),
                    ],
                ),
            ).metrics,
        )
        for target in prompt_token_targets
    )
    return PromptScalingProbeResult(
        runtime_load_ms=load_ms,
        runtime_load_resources=load_resources,
        cases=cases,
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
) -> OutputScalingProbeResult:
    runtime_config = RuntimeConfig(
        model_reference=model_reference,
        models_dir=models_dir.expanduser().resolve(),
        device=device,
        backend=backend,
        use_specialization=use_specialization,
        use_cache=True,
    )
    client = RuntimeClient()
    runtime_result = measure_stage(
        runtime_config.device,
        lambda: client.load(runtime_config),
    )
    runtime = cast(LoadedRuntime, runtime_result[0])
    load_ms = runtime_result[1]
    load_resources = runtime_result[2]
    cases = tuple(
        OutputScalingCase(
            requested_max_new_tokens=target,
            request=_execute_request_probe(
                runtime=runtime,
                request=_build_prompt_request(
                    runtime_config=runtime_config,
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
    )
    return OutputScalingProbeResult(
        runtime_load_ms=load_ms,
        runtime_load_resources=load_resources,
        cases=cases,
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
) -> SessionGrowthProbeResult:
    runtime_config = RuntimeConfig(
        model_reference=model_reference,
        models_dir=models_dir.expanduser().resolve(),
        device=device,
        backend=backend,
        use_specialization=use_specialization,
        use_cache=True,
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
    load_ms = runtime_result[1]
    load_resources = runtime_result[2]
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
        request_messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=[ContentPart.text(DEFAULT_SYSTEM_PROMPT)],
            ),
            *history,
            user_message,
        ]
        execution = _execute_request_probe(
            runtime=runtime,
            request=_build_prompt_request(
                runtime_config=runtime_config,
                generation_config=generation_config,
                messages=request_messages,
            ),
        )
        history.append(user_message)
        history.append(Message.assistant_text(execution.response_text))
        turns.append(
            SessionGrowthTurn(turn_index=turn_index, request=execution.metrics)
        )
    return SessionGrowthProbeResult(
        runtime_load_ms=load_ms,
        runtime_load_resources=load_resources,
        turns=tuple(turns),
    )


def render_runtime_probe_json(probe: RuntimeProbeResult) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def render_warm_runtime_probe_json(probe: WarmRuntimeProbeResult) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def render_prompt_scaling_probe_json(probe: PromptScalingProbeResult) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def render_output_scaling_probe_json(probe: OutputScalingProbeResult) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def render_session_growth_probe_json(probe: SessionGrowthProbeResult) -> str:
    return json.dumps(probe.to_dict(), indent=2, sort_keys=True)


def parse_runtime_probe_result(stdout: str) -> RuntimeProbeResult:
    payload = _load_probe_payload(stdout)
    load_resources = _parse_stage_resources(_require_mapping(payload, "load_resources"))
    request = _parse_request_probe_metrics(_require_mapping(payload, "request"))
    return RuntimeProbeResult(
        load_ms=_require_float(payload, "load_ms"),
        load_resources=load_resources,
        request=request,
    )


def parse_warm_runtime_probe_result(stdout: str) -> WarmRuntimeProbeResult:
    payload = _load_probe_payload(stdout)
    request_items = _require_sequence(payload, "measured_iterations")
    return WarmRuntimeProbeResult(
        runtime_load_ms=_require_float(payload, "runtime_load_ms"),
        runtime_load_resources=_parse_stage_resources(
            _require_mapping(payload, "runtime_load_resources")
        ),
        warmup_iterations=_require_int(payload, "warmup_iterations"),
        measured_iterations=tuple(
            _parse_request_probe_metrics(
                _require_object_mapping(item, f"measured_iterations[{index}]")
            )
            for index, item in enumerate(request_items)
        ),
    )


def parse_prompt_scaling_probe_result(stdout: str) -> PromptScalingProbeResult:
    payload = _load_probe_payload(stdout)
    case_items = _require_sequence(payload, "cases")
    return PromptScalingProbeResult(
        runtime_load_ms=_require_float(payload, "runtime_load_ms"),
        runtime_load_resources=_parse_stage_resources(
            _require_mapping(payload, "runtime_load_resources")
        ),
        cases=tuple(
            PromptScalingCase(
                requested_prompt_tokens=_require_int(
                    case_payload, "requested_prompt_tokens"
                ),
                request=_parse_request_probe_metrics(
                    _require_mapping(case_payload, "request")
                ),
            )
            for case_payload in (
                _require_object_mapping(item, f"cases[{index}]")
                for index, item in enumerate(case_items)
            )
        ),
    )


def parse_output_scaling_probe_result(stdout: str) -> OutputScalingProbeResult:
    payload = _load_probe_payload(stdout)
    case_items = _require_sequence(payload, "cases")
    return OutputScalingProbeResult(
        runtime_load_ms=_require_float(payload, "runtime_load_ms"),
        runtime_load_resources=_parse_stage_resources(
            _require_mapping(payload, "runtime_load_resources")
        ),
        cases=tuple(
            OutputScalingCase(
                requested_max_new_tokens=_require_int(
                    case_payload, "requested_max_new_tokens"
                ),
                request=_parse_request_probe_metrics(
                    _require_mapping(case_payload, "request")
                ),
            )
            for case_payload in (
                _require_object_mapping(item, f"cases[{index}]")
                for index, item in enumerate(case_items)
            )
        ),
    )


def parse_session_growth_probe_result(stdout: str) -> SessionGrowthProbeResult:
    payload = _load_probe_payload(stdout)
    turn_items = _require_sequence(payload, "turns")
    return SessionGrowthProbeResult(
        runtime_load_ms=_require_float(payload, "runtime_load_ms"),
        runtime_load_resources=_parse_stage_resources(
            _require_mapping(payload, "runtime_load_resources")
        ),
        turns=tuple(
            SessionGrowthTurn(
                turn_index=_require_int(turn_payload, "turn_index"),
                request=_parse_request_probe_metrics(
                    _require_mapping(turn_payload, "request")
                ),
            )
            for turn_payload in (
                _require_object_mapping(item, f"turns[{index}]")
                for index, item in enumerate(turn_items)
            )
        ),
    )


def _execute_request_probe(
    *,
    runtime: LoadedRuntime,
    request: PromptRequest,
) -> _RequestProbeExecution:
    executor = RuntimeExecutor()
    executor._validate_request(runtime, request)
    inputs = executor._build_inputs(runtime, request.messages)
    prompt_tokens = _count_prompt_tokens(inputs)
    streamer = TimedBufferedTextStreamer(runtime.tokenizer)
    generate_kwargs = executor._build_generate_kwargs(runtime, request, streamer)
    cache_mode = _cache_mode(runtime, request)
    generation_result, generation_ms, generation_resources = measure_stage(
        runtime.config.device,
        lambda: _generate_outputs(runtime, inputs, generate_kwargs),
        sample_accelerator_utilization=True,
    )
    outputs = cast(tuple[object, float], generation_result)[0]
    generation_started = cast(tuple[object, float], generation_result)[1]
    output_tensor = cast(torch.Tensor, outputs)
    if hasattr(output_tensor, "detach"):
        output_tensor = output_tensor.detach()
    cpu_outputs = output_tensor.cpu()
    response_text = executor._decode_response(runtime, inputs, cpu_outputs)
    output_tokens = _count_output_tokens(runtime, inputs, cpu_outputs)
    total_ms = generation_ms
    token_timestamps = streamer.token_timestamps
    time_to_first_token_ms = None
    if token_timestamps:
        time_to_first_token_ms = round(
            (token_timestamps[0] - generation_started) * 1000.0, 6
        )
    inter_token_latencies_ms = _inter_token_latencies(token_timestamps)
    prompt_tokens_per_second = None
    if time_to_first_token_ms is not None and time_to_first_token_ms > 0:
        prompt_tokens_per_second = round(
            prompt_tokens / (time_to_first_token_ms / 1000.0),
            6,
        )
    output_tokens_per_second = None
    if generation_ms > 0:
        output_tokens_per_second = round(
            output_tokens / (generation_ms / 1000.0),
            6,
        )
    cache_dir_size = None
    if cache_mode == "disk-kv":
        cache_dir_size = cache_dir_size_mb(
            request.runtime_config.resolved_cache_dir() / "kv_cache"
        )
    allocator_gap_mb = None
    allocator_gap_ratio = None
    if (
        generation_resources.accelerator_peak_reserved_mb is not None
        and generation_resources.accelerator_peak_mb is not None
    ):
        allocator_gap_mb = round(
            generation_resources.accelerator_peak_reserved_mb
            - generation_resources.accelerator_peak_mb,
            6,
        )
        if generation_resources.accelerator_peak_reserved_mb > 0:
            allocator_gap_ratio = round(
                allocator_gap_mb / generation_resources.accelerator_peak_reserved_mb,
                6,
            )
    metrics = RequestProbeMetrics(
        total_ms=round(total_ms, 6),
        generation_ms=round(generation_ms, 6),
        time_to_first_token_ms=time_to_first_token_ms,
        inter_token_latencies_ms=inter_token_latencies_ms,
        prompt_tokens=prompt_tokens,
        prompt_tokens_per_second=prompt_tokens_per_second,
        output_tokens=output_tokens,
        output_tokens_per_second=output_tokens_per_second,
        cache_mode=cache_mode,
        cache_dir_size_mb=cache_dir_size,
        allocator_gap_mb=allocator_gap_mb,
        allocator_gap_ratio=allocator_gap_ratio,
        resources=generation_resources,
        text_excerpt=_clip_text(response_text, max_chars=120),
    )
    return _RequestProbeExecution(metrics=metrics, response_text=response_text)


def _generate_outputs(
    runtime: LoadedRuntime,
    inputs: dict[str, object],
    generate_kwargs: dict[str, object],
):
    generation_started = time.perf_counter()
    try:
        with torch.inference_mode():
            with suppress_module_prints(runtime.backend.print_suppression_modules):
                return runtime.model.generate(
                    **inputs, **generate_kwargs
                ), generation_started
    except TypeError as exc:
        if "streamer" not in str(exc):
            raise
        retry_kwargs = dict(generate_kwargs)
        retry_kwargs.pop("streamer", None)
        with torch.inference_mode():
            with suppress_module_prints(runtime.backend.print_suppression_modules):
                return runtime.model.generate(
                    **inputs, **retry_kwargs
                ), generation_started


def _build_prompt_request(
    *,
    runtime_config: RuntimeConfig,
    generation_config: GenerationConfig,
    messages: list[Message],
) -> PromptRequest:
    return PromptRequest(
        runtime_config=runtime_config,
        generation_config=generation_config,
        messages=messages,
    )


def _count_prompt_tokens(inputs: dict[str, object]) -> int:
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise PromptExecutionError("Benchmark probe expected tensor-backed input_ids")
    if input_ids.ndim == 1:
        return int(input_ids.shape[0])
    return int(input_ids.shape[-1])


def _count_output_tokens(
    runtime: LoadedRuntime, inputs: dict[str, object], outputs: torch.Tensor
) -> int:
    input_ids = cast(torch.Tensor, inputs["input_ids"])
    if runtime.plan.generic_model_kind is GenericModelKind.SEQ2SEQ_LM:
        return int(outputs.shape[-1])
    return max(0, int(outputs.shape[-1] - input_ids.shape[-1]))


def _cache_mode(runtime: LoadedRuntime, request: PromptRequest) -> str:
    if not request.runtime_config.use_cache:
        return "none"
    if runtime.plan.supports_disk_cache:
        return "disk-kv"
    return "transformers-dynamic"


def _inter_token_latencies(
    token_timestamps: tuple[float, ...],
) -> tuple[float, ...]:
    if len(token_timestamps) < 2:
        return ()
    latencies = [
        round((right - left) * 1000.0, 6)
        for left, right in zip(token_timestamps, token_timestamps[1:])
    ]
    return tuple(latencies)


def _build_scaling_prompt(*, target_prompt_tokens: int) -> str:
    repeated_word_count = max(1, target_prompt_tokens)
    repeated_words = " ".join("benchmark" for _ in range(repeated_word_count))
    return (
        "Benchmark scaling probe input. "
        "Repeat and summarize this synthetic workload: "
        f"{repeated_words}"
    )


def _clip_text(text: str, *, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _load_probe_payload(stdout: str) -> Mapping[str, object]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("runtime probe output was not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("runtime probe output must be a JSON object")
    return cast(Mapping[str, object], payload)


def _parse_request_probe_metrics(payload: Mapping[str, object]) -> RequestProbeMetrics:
    inter_token_values = _require_sequence(payload, "inter_token_latencies_ms")
    return RequestProbeMetrics(
        total_ms=_require_float(payload, "total_ms"),
        generation_ms=_require_float(payload, "generation_ms"),
        time_to_first_token_ms=_optional_float(payload, "time_to_first_token_ms"),
        inter_token_latencies_ms=tuple(
            _require_numeric_value(value, "inter_token_latencies_ms[]")
            for value in inter_token_values
        ),
        prompt_tokens=_require_int(payload, "prompt_tokens"),
        prompt_tokens_per_second=_optional_float(payload, "prompt_tokens_per_second"),
        output_tokens=_require_int(payload, "output_tokens"),
        output_tokens_per_second=_optional_float(payload, "output_tokens_per_second"),
        cache_mode=_require_string(payload, "cache_mode"),
        cache_dir_size_mb=_optional_float(payload, "cache_dir_size_mb"),
        allocator_gap_mb=_optional_float(payload, "allocator_gap_mb"),
        allocator_gap_ratio=_optional_float(payload, "allocator_gap_ratio"),
        resources=_parse_stage_resources(_require_mapping(payload, "resources")),
        text_excerpt=_require_string(payload, "text_excerpt"),
    )


def _parse_stage_resources(payload: Mapping[str, object]) -> StageResourceSnapshot:
    utilization_payload = payload.get("accelerator_utilization")
    accelerator_utilization = None
    if isinstance(utilization_payload, Mapping):
        utilization_mapping = cast(Mapping[str, object], utilization_payload)
        gpu_utilization_payload = utilization_mapping.get("gpu_utilization_percent")
        memory_utilization_payload = utilization_mapping.get(
            "memory_utilization_percent"
        )
        accelerator_utilization = AcceleratorUtilizationSnapshot(
            gpu_utilization_percent=_parse_optional_summary(gpu_utilization_payload),
            memory_utilization_percent=_parse_optional_summary(
                memory_utilization_payload
            ),
        )
    return StageResourceSnapshot(
        current_rss_mb=_optional_float(payload, "current_rss_mb"),
        peak_rss_mb=_optional_float(payload, "peak_rss_mb"),
        peak_rss_source=_optional_string(payload, "peak_rss_source"),
        accelerator_kind=_optional_string(payload, "accelerator_kind"),
        accelerator_current_mb=_optional_float(payload, "accelerator_current_mb"),
        accelerator_peak_mb=_optional_float(payload, "accelerator_peak_mb"),
        accelerator_reserved_mb=_optional_float(payload, "accelerator_reserved_mb"),
        accelerator_peak_reserved_mb=_optional_float(
            payload, "accelerator_peak_reserved_mb"
        ),
        accelerator_peak_source=_optional_string(payload, "accelerator_peak_source"),
        process_cpu_utilization_percent=_optional_float(
            payload, "process_cpu_utilization_percent"
        ),
        accelerator_utilization=accelerator_utilization,
    )


def _parse_optional_summary(value: object) -> NumericSummary | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("numeric summary must be an object or null")
    payload = cast(Mapping[str, object], value)
    return NumericSummary(
        min=_require_float(payload, "min"),
        median=_require_float(payload, "median"),
        p95=_require_float(payload, "p95"),
        max=_require_float(payload, "max"),
        mean=_require_float(payload, "mean"),
    )


def _require_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"probe field '{key}' must be an object")
    return cast(Mapping[str, object], value)


def _require_object_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"probe field '{field_name}' must be an object")
    return cast(Mapping[str, object], value)


def _require_sequence(payload: Mapping[str, object], key: str) -> tuple[object, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"probe field '{key}' must be a list")
    return tuple(value)


def _require_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"probe field '{key}' must be numeric")


def _optional_float(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"probe field '{key}' must be numeric or null")


def _require_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int):
        return value
    raise ValueError(f"probe field '{key}' must be an integer")


def _require_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    raise ValueError(f"probe field '{key}' must be a string")


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"probe field '{key}' must be a string or null")


def _require_numeric_value(value: object, key: str) -> float:
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"probe field '{key}' must contain only numeric values")

"""Benchmark probe result types."""

from dataclasses import asdict, dataclass

from ollm.runtime.benchmark_resources import StageResourceSnapshot


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
class RequestProbeExecution:
    metrics: RequestProbeMetrics
    response_text: str

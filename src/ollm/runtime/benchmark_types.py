"""Shared benchmark dataclasses, constants, and serialization helpers."""

import json
from dataclasses import asdict, dataclass

DEFAULT_PROMPT_TOKEN_TARGETS = (32, 128, 512)
DEFAULT_OUTPUT_TOKEN_TARGETS = (16, 64, 128)
DEFAULT_SESSION_TURNS = 4


@dataclass(frozen=True, slots=True)
class BenchmarkStats:
    """Summarize timing samples for one benchmark measurement."""

    iterations: int
    warmup_iterations: int
    min_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float
    mean_ms: float

    def to_dict(self) -> dict[str, float | int]:
        """Return the benchmark stats as a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    """Represent a single benchmark result and its attached details."""

    name: str
    status: str
    stats: BenchmarkStats | None
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return the benchmark measurement as a JSON-serializable dictionary."""

        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "details": self.details,
        }
        payload["stats"] = None if self.stats is None else self.stats.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeBenchmarkReport:
    """Represent the full runtime benchmark report payload."""

    generated_at: str
    host: dict[str, object]
    benchmark_model_reference: str
    device: str
    specialization_planner_overhead: dict[str, object]
    fallback_cost_when_no_specialization_applies: dict[str, object]
    runtime_comparison: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return the report as a JSON-serializable dictionary."""

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
    """Describe a subprocess-backed benchmark command."""

    name: str
    command: tuple[str, ...]
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class CommandExecutionResult:
    """Capture a completed benchmark command execution."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeComparisonTarget:
    """Describe a benchmarkable runtime comparison target."""

    family: str
    model_reference: str
    is_materialized: bool
    model_path: str | None


def render_report_json(report: RuntimeBenchmarkReport) -> str:
    """Render a runtime benchmark report as stable JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def unavailable_measurement(
    name: str, *, details: dict[str, object]
) -> BenchmarkMeasurement:
    """Return an unavailable benchmark measurement payload."""

    return BenchmarkMeasurement(
        name=name,
        status="unavailable",
        stats=None,
        details=details,
    )

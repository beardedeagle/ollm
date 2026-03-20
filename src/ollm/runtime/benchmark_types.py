"""Shared benchmark dataclasses, constants, and serialization helpers."""

import json
from dataclasses import asdict, dataclass

DEFAULT_RUNTIME_BENCHMARK_PROFILE = "quick"
KNOWN_RUNTIME_BENCHMARK_PROFILES = ("quick", "full")
DEFAULT_PROMPT_TOKEN_TARGETS = (32, 128, 512)
DEFAULT_OUTPUT_TOKEN_TARGETS = (16, 64, 128)
DEFAULT_SESSION_TURNS = 4
DEFAULT_SESSION_MAX_NEW_TOKENS = 4
QUICK_PROMPT_TOKEN_TARGETS = (32,)
QUICK_OUTPUT_TOKEN_TARGETS = (16,)
QUICK_SESSION_TURNS = 1
QUICK_SESSION_MAX_NEW_TOKENS = 4


@dataclass(frozen=True, slots=True)
class RuntimeBenchmarkProfile:
    """Describe the scope and budgets for one benchmark profile."""

    profile_id: str
    iterations: int
    warmup_iterations: int
    prompt_token_targets: tuple[int, ...]
    output_token_targets: tuple[int, ...]
    session_turns: int
    session_max_new_tokens: int
    include_family_results: bool
    include_primary_extended_scenarios: bool
    cold_timeout_seconds: float
    warm_timeout_seconds: float
    scaling_timeout_seconds: float
    session_timeout_seconds: float


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


def normalize_runtime_benchmark_profile(profile: str | None) -> str:
    """Validate and normalize a benchmark profile identifier."""

    normalized = (
        DEFAULT_RUNTIME_BENCHMARK_PROFILE
        if profile is None
        else profile.strip().lower()
    )
    if normalized not in KNOWN_RUNTIME_BENCHMARK_PROFILES:
        allowed = ", ".join(KNOWN_RUNTIME_BENCHMARK_PROFILES)
        raise ValueError(f"--profile must be one of: {allowed}")
    return normalized


def resolve_runtime_benchmark_profile(
    *,
    profile: str | None,
    iterations: int | None = None,
    warmup_iterations: int | None = None,
    prompt_token_targets: tuple[int, ...] | None = None,
    output_token_targets: tuple[int, ...] | None = None,
    session_turns: int | None = None,
    session_max_new_tokens: int | None = None,
) -> RuntimeBenchmarkProfile:
    """Resolve a benchmark profile and apply any explicit CLI overrides."""

    normalized_profile = normalize_runtime_benchmark_profile(profile)
    if normalized_profile == "quick":
        return RuntimeBenchmarkProfile(
            profile_id=normalized_profile,
            iterations=1 if iterations is None else iterations,
            warmup_iterations=0 if warmup_iterations is None else warmup_iterations,
            prompt_token_targets=(
                QUICK_PROMPT_TOKEN_TARGETS
                if prompt_token_targets is None
                else prompt_token_targets
            ),
            output_token_targets=(
                QUICK_OUTPUT_TOKEN_TARGETS
                if output_token_targets is None
                else output_token_targets
            ),
            session_turns=QUICK_SESSION_TURNS
            if session_turns is None
            else session_turns,
            session_max_new_tokens=(
                QUICK_SESSION_MAX_NEW_TOKENS
                if session_max_new_tokens is None
                else session_max_new_tokens
            ),
            include_family_results=False,
            include_primary_extended_scenarios=False,
            cold_timeout_seconds=90.0,
            warm_timeout_seconds=150.0,
            scaling_timeout_seconds=180.0,
            session_timeout_seconds=180.0,
        )
    return RuntimeBenchmarkProfile(
        profile_id=normalized_profile,
        iterations=5 if iterations is None else iterations,
        warmup_iterations=1 if warmup_iterations is None else warmup_iterations,
        prompt_token_targets=(
            DEFAULT_PROMPT_TOKEN_TARGETS
            if prompt_token_targets is None
            else prompt_token_targets
        ),
        output_token_targets=(
            DEFAULT_OUTPUT_TOKEN_TARGETS
            if output_token_targets is None
            else output_token_targets
        ),
        session_turns=DEFAULT_SESSION_TURNS if session_turns is None else session_turns,
        session_max_new_tokens=(
            DEFAULT_SESSION_MAX_NEW_TOKENS
            if session_max_new_tokens is None
            else session_max_new_tokens
        ),
        include_family_results=True,
        include_primary_extended_scenarios=True,
        cold_timeout_seconds=240.0,
        warm_timeout_seconds=240.0,
        scaling_timeout_seconds=300.0,
        session_timeout_seconds=300.0,
    )


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

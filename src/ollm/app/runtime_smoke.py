"""Reusable runtime smoke validation over the shared application service."""

import json
from dataclasses import asdict, dataclass

from ollm.app.service import ApplicationService
from ollm.app.types import ContentPart, PromptResponse
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.inspection import PlanJsonPayload

DEFAULT_RUNTIME_SMOKE_SYSTEM_PROMPT = (
    "You are a runtime smoke validator. When the user asks for the validation "
    "token, reply with PASS and nothing else."
)
DEFAULT_RUNTIME_SMOKE_PROMPT = "Return the validation token PASS."
DEFAULT_RUNTIME_SMOKE_CHAT_TURNS: tuple[str, ...] = (
    "Return PASS and nothing else.",
    "Return PASS again and nothing else.",
)
DEFAULT_RUNTIME_SMOKE_EXPECTATIONS: tuple[str, ...] = ("PASS",)


@dataclass(frozen=True, slots=True)
class RuntimeSmokeTurnResult:
    """Validation result for a single prompt or chat turn."""

    prompt_text: str
    response_text: str
    matched_expectations: tuple[str, ...]
    missing_expectations: tuple[str, ...]
    ok: bool
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class RuntimeSmokeChatResult:
    """Validation result for a stateful chat session."""

    system_prompt: str
    turns: tuple[RuntimeSmokeTurnResult, ...]
    ok: bool


@dataclass(frozen=True, slots=True)
class RuntimeSmokeGenerationPayload:
    """Serialized generation settings used by the smoke run."""

    max_new_tokens: int
    temperature: float
    top_p: float | None
    top_k: int | None
    seed: int | None
    stream: bool


@dataclass(frozen=True, slots=True)
class RuntimeSmokeReport:
    """Structured runtime smoke report."""

    ok: bool
    expected_contains: tuple[str, ...]
    generation: RuntimeSmokeGenerationPayload
    plan: PlanJsonPayload
    prompt: RuntimeSmokeTurnResult
    chat: RuntimeSmokeChatResult


def normalize_runtime_smoke_expectations(
    expectations: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate and normalize expected response substrings."""

    normalized: list[str] = []
    for expectation in expectations:
        stripped_expectation = expectation.strip()
        if not stripped_expectation:
            raise ValueError("runtime smoke expectations cannot be empty")
        normalized.append(stripped_expectation)
    if not normalized:
        raise ValueError("runtime smoke requires at least one expectation")
    return tuple(normalized)


def generation_payload(
    generation_config: GenerationConfig,
) -> RuntimeSmokeGenerationPayload:
    """Serialize generation settings used for a smoke run."""

    return RuntimeSmokeGenerationPayload(
        max_new_tokens=generation_config.max_new_tokens,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        top_k=generation_config.top_k,
        seed=generation_config.seed,
        stream=generation_config.stream,
    )


def run_runtime_smoke(
    *,
    service: ApplicationService,
    runtime_config: RuntimeConfig,
    generation_config: GenerationConfig,
    prompt_text: str = DEFAULT_RUNTIME_SMOKE_PROMPT,
    chat_turns: tuple[str, ...] = DEFAULT_RUNTIME_SMOKE_CHAT_TURNS,
    system_prompt: str = DEFAULT_RUNTIME_SMOKE_SYSTEM_PROMPT,
    expected_contains: tuple[str, ...] = DEFAULT_RUNTIME_SMOKE_EXPECTATIONS,
) -> RuntimeSmokeReport:
    """Run one-shot and chat-session smoke validation against a real runtime."""

    normalized_expectations = normalize_runtime_smoke_expectations(expected_contains)
    if not prompt_text.strip():
        raise ValueError("runtime smoke prompt_text cannot be empty")
    if not system_prompt.strip():
        raise ValueError("runtime smoke system_prompt cannot be empty")
    if not chat_turns:
        raise ValueError("runtime smoke requires at least one chat turn")
    for turn_text in chat_turns:
        if not turn_text.strip():
            raise ValueError("runtime smoke chat turns cannot be empty")

    runtime_config.validate()
    generation_config.validate()

    plan = service.describe_plan(runtime_config)
    prompt_result = _run_prompt_validation(
        service=service,
        runtime_config=runtime_config,
        generation_config=generation_config,
        system_prompt=system_prompt,
        prompt_text=prompt_text,
        expected_contains=normalized_expectations,
    )
    chat_result = _run_chat_validation(
        service=service,
        runtime_config=runtime_config,
        generation_config=generation_config,
        system_prompt=system_prompt,
        chat_turns=chat_turns,
        expected_contains=normalized_expectations,
    )
    return RuntimeSmokeReport(
        ok=prompt_result.ok and chat_result.ok,
        expected_contains=normalized_expectations,
        generation=generation_payload(generation_config),
        plan=plan,
        prompt=prompt_result,
        chat=chat_result,
    )


def runtime_smoke_report_payload(report: RuntimeSmokeReport) -> dict[str, object]:
    """Return a JSON-serializable runtime smoke payload."""

    return asdict(report)


def render_runtime_smoke_report_json(report: RuntimeSmokeReport) -> str:
    """Render a runtime smoke report as formatted JSON."""

    return json.dumps(runtime_smoke_report_payload(report), indent=2)


def _run_prompt_validation(
    *,
    service: ApplicationService,
    runtime_config: RuntimeConfig,
    generation_config: GenerationConfig,
    system_prompt: str,
    prompt_text: str,
    expected_contains: tuple[str, ...],
) -> RuntimeSmokeTurnResult:
    response = service.prompt_parts(
        [ContentPart.text(prompt_text)],
        runtime_config=runtime_config,
        generation_config=generation_config,
        system_prompt=system_prompt,
    )
    return _evaluate_response(
        prompt_text=prompt_text,
        response=response,
        expected_contains=expected_contains,
    )


def _run_chat_validation(
    *,
    service: ApplicationService,
    runtime_config: RuntimeConfig,
    generation_config: GenerationConfig,
    system_prompt: str,
    chat_turns: tuple[str, ...],
    expected_contains: tuple[str, ...],
) -> RuntimeSmokeChatResult:
    session = service.create_session(
        runtime_config=runtime_config,
        generation_config=generation_config,
        session_name="runtime-smoke",
        system_prompt=system_prompt,
    )
    turn_results: list[RuntimeSmokeTurnResult] = []
    for prompt_text in chat_turns:
        response = session.prompt_text(prompt_text)
        turn_results.append(
            _evaluate_response(
                prompt_text=prompt_text,
                response=response,
                expected_contains=expected_contains,
            )
        )
    turns = tuple(turn_results)
    return RuntimeSmokeChatResult(
        system_prompt=system_prompt,
        turns=turns,
        ok=all(turn.ok for turn in turns),
    )


def _evaluate_response(
    *,
    prompt_text: str,
    response: PromptResponse,
    expected_contains: tuple[str, ...],
) -> RuntimeSmokeTurnResult:
    normalized_response = _normalize_text(response.text)
    matched_expectations: list[str] = []
    missing_expectations: list[str] = []
    for expectation in expected_contains:
        normalized_expectation = _normalize_text(expectation)
        if normalized_expectation in normalized_response:
            matched_expectations.append(expectation)
        else:
            missing_expectations.append(expectation)
    return RuntimeSmokeTurnResult(
        prompt_text=prompt_text,
        response_text=response.text,
        matched_expectations=tuple(matched_expectations),
        missing_expectations=tuple(missing_expectations),
        ok=bool(response.text.strip()) and not missing_expectations,
        metadata=dict(response.metadata),
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.upper().split())

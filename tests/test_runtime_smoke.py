import sys
from pathlib import Path
from typing import cast

from ollm.app.doctor import DoctorService
from ollm.app.runtime_smoke import (
    render_runtime_smoke_report_json,
    run_runtime_smoke,
)
from ollm.app.service import ApplicationService
from ollm.client import RuntimeClient
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader
from tests.fakes import FakeDoctorService, FakeRuntimeExecutor, FakeRuntimeLoader

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "runtime_smoke.py"


def build_service() -> ApplicationService:
    loader = FakeRuntimeLoader()
    executor = FakeRuntimeExecutor()
    return ApplicationService(
        runtime_client=RuntimeClient(
            runtime_loader=cast(RuntimeLoader, loader),
            runtime_executor=cast(RuntimeExecutor, executor),
        ),
        doctor_service=cast(DoctorService, FakeDoctorService()),
    )


def test_runtime_smoke_passes_when_expected_substring_is_present(
    tmp_path: Path,
) -> None:
    report = run_runtime_smoke(
        service=build_service(),
        runtime_config=RuntimeConfig(
            model_reference="llama3-1B-chat",
            models_dir=tmp_path / "models",
        ),
        generation_config=GenerationConfig(stream=False, max_new_tokens=8),
        prompt_text="validation token one",
        chat_turns=("validation token two", "validation token three"),
        system_prompt="Return the prompt back.",
        expected_contains=("validation token",),
    )

    assert report.ok is True
    assert report.prompt.ok is True
    assert all(turn.ok for turn in report.chat.turns)
    assert report.plan["runtime_plan"]["backend_id"] == "optimized-native"


def test_runtime_smoke_fails_when_expected_substring_is_missing(
    tmp_path: Path,
) -> None:
    report = run_runtime_smoke(
        service=build_service(),
        runtime_config=RuntimeConfig(
            model_reference="llama3-1B-chat",
            models_dir=tmp_path / "models",
        ),
        generation_config=GenerationConfig(stream=False, max_new_tokens=8),
        prompt_text="echoable prompt",
        chat_turns=("chat prompt one", "chat prompt two"),
        system_prompt="Echo the prompt text.",
        expected_contains=("PASS",),
    )

    assert report.ok is False
    assert report.prompt.missing_expectations == ("PASS",)
    assert report.chat.turns[0].missing_expectations == ("PASS",)


def test_runtime_smoke_report_json_includes_plan_and_chat_results(
    tmp_path: Path,
) -> None:
    report = run_runtime_smoke(
        service=build_service(),
        runtime_config=RuntimeConfig(
            model_reference="llama3-1B-chat",
            models_dir=tmp_path / "models",
        ),
        generation_config=GenerationConfig(stream=False, max_new_tokens=8),
        prompt_text="validation token one",
        chat_turns=("validation token two",),
        system_prompt="Return the prompt back.",
        expected_contains=("validation token",),
    )

    rendered = render_runtime_smoke_report_json(report)

    assert '"expected_contains"' in rendered
    assert '"runtime_plan"' in rendered
    assert '"chat"' in rendered
    assert '"generation"' in rendered


def test_runtime_smoke_script_help_renders() -> None:
    from ollm.async_io import subprocess_run_process

    completed = subprocess_run_process((sys.executable, str(SCRIPT_PATH), "--help"))

    assert completed.returncode == 0
    assert "runtime smoke validation" in completed.stdout.lower()

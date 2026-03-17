from pathlib import Path
from typing import cast

from ollm.app.doctor import DoctorService
from ollm.app.service import ApplicationService
from ollm.app.types import ContentPart
from ollm.client import RuntimeClient
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader
from tests.fakes import FakeDoctorService, FakeRuntimeExecutor, FakeRuntimeLoader


def build_service() -> tuple[
    ApplicationService,
    FakeRuntimeLoader,
    FakeRuntimeExecutor,
    FakeDoctorService,
]:
    loader = FakeRuntimeLoader()
    executor = FakeRuntimeExecutor()
    doctor = FakeDoctorService()
    service = ApplicationService(
        runtime_client=RuntimeClient(
            runtime_loader=cast(RuntimeLoader, loader),
            runtime_executor=cast(RuntimeExecutor, executor),
        ),
        doctor_service=cast(DoctorService, doctor),
    )
    return service, loader, executor, doctor


def test_application_service_describe_plan_uses_shared_runtime_client(
    tmp_path: Path,
) -> None:
    service, loader, _, _ = build_service()
    runtime_config = RuntimeConfig(
        model_reference="llama3-1B-chat",
        models_dir=tmp_path / "models",
        backend="transformers-generic",
        use_specialization=False,
    )

    payload = service.describe_plan(runtime_config)

    assert loader.plan_calls[-1].backend == "transformers-generic"
    assert payload["runtime_plan"]["backend_id"] == "transformers-generic"


def test_application_service_prompt_parts_executes_through_shared_runtime_stack(
    tmp_path: Path,
) -> None:
    service, loader, executor, _ = build_service()

    response = service.prompt_parts(
        [ContentPart.text("hello service")],
        runtime_config=RuntimeConfig(
            model_reference="llama3-1B-chat",
            models_dir=tmp_path / "models",
        ),
        generation_config=GenerationConfig(stream=False),
    )

    assert response.text == "echo:hello service"
    assert loader.load_calls == ["llama3-1B-chat"]
    assert executor.prompts == ["hello service"]


def test_application_service_create_session_uses_shared_runtime_stack(
    tmp_path: Path,
) -> None:
    service, loader, executor, _ = build_service()
    session = service.create_session(
        runtime_config=RuntimeConfig(
            model_reference="llama3-1B-chat",
            models_dir=tmp_path / "models",
        ),
        generation_config=GenerationConfig(stream=False),
        session_name="default",
        system_prompt="You are concise.",
    )

    response = session.prompt_text("List planets")

    assert response.text == "echo:List planets"
    assert loader.load_calls == ["llama3-1B-chat"]
    assert executor.prompts == ["List planets"]


def test_application_service_doctor_uses_shared_doctor_service(tmp_path: Path) -> None:
    service, _, _, _ = build_service()

    report = service.run_doctor(
        runtime_config=RuntimeConfig(
            model_reference="llama3-1B-chat",
            models_dir=tmp_path / "models",
        )
    )

    assert report.ok() is True
    assert report.checks[0].name == "doctor:fake"


def test_application_service_download_uses_shared_runtime_loader(
    tmp_path: Path,
) -> None:
    service, loader, _, _ = build_service()
    models_dir = tmp_path / "models"

    model_path = service.download_model(
        "llama3-1B-chat",
        models_dir,
        force_download=True,
    )

    assert model_path == models_dir / "llama3-1B-chat"
    assert loader.download_calls == [("llama3-1B-chat", models_dir, True)]

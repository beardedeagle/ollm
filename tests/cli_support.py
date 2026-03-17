import re
from typing import cast

from typer.testing import CliRunner

from ollm.app.doctor import DoctorService
from ollm.app.service import ApplicationService
from ollm.cli.main import create_app
from ollm.cli.services import CommandServices
from ollm.client import RuntimeClient
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader
from tests.fakes import FakeDoctorService, FakeRuntimeExecutor, FakeRuntimeLoader


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so assertions work regardless of terminal."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def build_test_app():
    loader = FakeRuntimeLoader()
    executor = FakeRuntimeExecutor()
    client = RuntimeClient(
        runtime_loader=cast(RuntimeLoader, loader),
        runtime_executor=cast(RuntimeExecutor, executor),
    )
    services = CommandServices(
        application_service=ApplicationService(
            runtime_client=client,
            doctor_service=cast(DoctorService, FakeDoctorService()),
        )
    )
    return CliRunner(), loader, create_app(services)


def build_real_app():
    runtime_loader = RuntimeLoader()
    client = RuntimeClient(
        runtime_loader=runtime_loader,
        runtime_executor=RuntimeExecutor(),
    )
    services = CommandServices(
        application_service=ApplicationService(
            runtime_client=client,
            doctor_service=DoctorService(runtime_loader=runtime_loader),
        )
    )
    return CliRunner(), create_app(services)

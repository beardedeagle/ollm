from dataclasses import dataclass

from ollm.app.doctor import DoctorService
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader


@dataclass(frozen=True, slots=True)
class CommandServices:
    runtime_loader: RuntimeLoader
    runtime_executor: RuntimeExecutor
    doctor_service: DoctorService


def build_default_services() -> CommandServices:
    return CommandServices(
        runtime_loader=RuntimeLoader(),
        runtime_executor=RuntimeExecutor(),
        doctor_service=DoctorService(),
    )


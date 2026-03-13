from dataclasses import dataclass

from ollm.app.doctor import DoctorService
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader
from ollm.runtime.resolver import ModelResolver


@dataclass(frozen=True, slots=True)
class CommandServices:
    runtime_loader: RuntimeLoader
    runtime_executor: RuntimeExecutor
    doctor_service: DoctorService



def build_default_services() -> CommandServices:
    resolver = ModelResolver()
    return CommandServices(
        runtime_loader=RuntimeLoader(resolver=resolver),
        runtime_executor=RuntimeExecutor(),
        doctor_service=DoctorService(resolver=resolver),
    )

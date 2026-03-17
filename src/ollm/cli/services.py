from dataclasses import dataclass

from ollm.app.service import ApplicationService
from ollm.client import RuntimeClient
from ollm.runtime.loader import RuntimeLoader
from ollm.runtime.resolver import ModelResolver


@dataclass(frozen=True, slots=True)
class CommandServices:
    application_service: ApplicationService


def build_default_services() -> CommandServices:
    resolver = ModelResolver()
    runtime_client = RuntimeClient(runtime_loader=RuntimeLoader(resolver=resolver))
    return CommandServices(
        application_service=ApplicationService(runtime_client=runtime_client),
    )

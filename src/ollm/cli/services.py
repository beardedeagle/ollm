from dataclasses import dataclass

from ollm.app.service import ApplicationService, build_default_application_service


@dataclass(frozen=True, slots=True)
class CommandServices:
    application_service: ApplicationService


def build_default_services() -> CommandServices:
    return CommandServices(
        application_service=build_default_application_service(),
    )

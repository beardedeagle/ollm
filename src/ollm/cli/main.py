import typer

from ollm.cli.chat import register_chat_surfaces
from ollm.cli.doctor import register_doctor_command
from ollm.cli.models import register_models_command
from ollm.cli.prompt import register_prompt_command
from ollm.cli.server import register_server_command
from ollm.cli.services import CommandServices, build_default_services


def create_app(services: CommandServices | None = None) -> typer.Typer:
    resolved_services = services or build_default_services()
    app = typer.Typer(
        add_completion=False,
        help="oLLM terminal interface.",
        invoke_without_command=True,
        no_args_is_help=False,
    )

    register_chat_surfaces(app, resolved_services)
    register_prompt_command(app, resolved_services)
    register_doctor_command(app, resolved_services)
    register_models_command(app, resolved_services)
    register_server_command(app, resolved_services)
    return app


app = create_app()


def main() -> None:
    app()

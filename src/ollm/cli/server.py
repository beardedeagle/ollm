import os

import typer

from ollm.cli.services import CommandServices
from ollm.runtime.settings import (
    ServerSettingsOverrides,
    load_app_settings,
    resolve_server_settings,
)
from ollm.server.runtime import ServerDependenciesError, serve_application


def register_server_command(app: typer.Typer, services: CommandServices) -> None:
    @app.command("serve")
    def serve_command(
        host: str | None = typer.Option(
            None, "--host", help="Server bind host. Defaults to 127.0.0.1."
        ),
        port: int | None = typer.Option(
            None, "--port", min=1, max=65535, help="Server bind port."
        ),
        reload: bool | None = typer.Option(
            None,
            "--reload/--no-reload",
            help="Enable uvicorn reload mode for the local-only server.",
        ),
        log_level: str | None = typer.Option(
            None, "--log-level", help="Server log level."
        ),
        response_store_backend: str | None = typer.Option(
            None,
            "--response-store-backend",
            help="Responses API storage backend: none, memory, or plugin.",
        ),
        response_store_factory: str | None = typer.Option(
            None,
            "--response-store-factory",
            help="Plugin factory import path for the responses store.",
        ),
    ) -> None:
        settings = load_app_settings()
        server_settings = resolve_server_settings(
            settings.server,
            ServerSettingsOverrides(
                host=host,
                port=port,
                reload=reload,
                log_level=log_level,
                response_store_backend=response_store_backend,
                response_store_factory=response_store_factory,
            ),
        )
        if server_settings.reload and response_store_backend is not None:
            os.environ["OLLM_SERVER__RESPONSE_STORE_BACKEND"] = response_store_backend
        if server_settings.reload and response_store_factory is not None:
            os.environ["OLLM_SERVER__RESPONSE_STORE_FACTORY"] = response_store_factory
        try:
            serve_application(server_settings, services.application_service)
        except ServerDependenciesError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)

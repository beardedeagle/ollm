"""Optional local-only server scaffold and lifecycle boundary for oLLM."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol, cast

from ollm.app.service import ApplicationService, build_default_application_service
from ollm.runtime.settings import ServerSettings

SERVER_EXTRA_INSTALL_HINT = (
    "Install server support with `uv sync --extra server` or "
    '`pip install --no-build-isolation -e ".[server]"`.'
)


class ServerDependenciesError(RuntimeError):
    """Raised when optional server transport dependencies are unavailable."""


class FastAPIApplication(Protocol):
    """Minimal FastAPI application protocol used by the server scaffold."""

    state: object


class FastAPIFactory(Protocol):
    """Protocol for the imported FastAPI application constructor."""

    def __call__(
        self,
        *,
        title: str,
        description: str,
        version: str,
    ) -> FastAPIApplication: ...


class FastAPIModule(Protocol):
    """Protocol for the imported FastAPI module."""

    FastAPI: FastAPIFactory


class UvicornServer(Protocol):
    """Protocol for the imported uvicorn server instance."""

    def run(self) -> None: ...


class UvicornConfigFactory(Protocol):
    """Protocol for the imported uvicorn config constructor."""

    def __call__(
        self,
        app: object,
        *,
        host: str,
        port: int,
        reload: bool,
        log_level: str,
        factory: bool = False,
    ) -> object: ...


class UvicornServerFactory(Protocol):
    """Protocol for the imported uvicorn server constructor."""

    def __call__(self, config: object) -> UvicornServer: ...


class UvicornModule(Protocol):
    """Protocol for the imported uvicorn module."""

    Config: UvicornConfigFactory
    Server: UvicornServerFactory


def _package_version() -> str:
    try:
        return version("ollm")
    except PackageNotFoundError:
        return "0.0.0"


def _load_fastapi_module() -> FastAPIModule:
    try:
        return cast(FastAPIModule, import_module("fastapi"))
    except ModuleNotFoundError as exc:
        raise ServerDependenciesError(SERVER_EXTRA_INSTALL_HINT) from exc


def _load_uvicorn_module() -> UvicornModule:
    try:
        return cast(UvicornModule, import_module("uvicorn"))
    except ModuleNotFoundError as exc:
        raise ServerDependenciesError(SERVER_EXTRA_INSTALL_HINT) from exc


def create_server_app(
    application_service: ApplicationService | None = None,
) -> FastAPIApplication:
    """Create the local-only server scaffold application."""
    fastapi = _load_fastapi_module()
    app = fastapi.FastAPI(
        title="oLLM",
        description="Local-only oLLM server scaffold",
        version=_package_version(),
    )
    resolved_application_service = (
        build_default_application_service()
        if application_service is None
        else application_service
    )
    setattr(app.state, "application_service", resolved_application_service)
    setattr(app.state, "server_mode", "scaffold")
    return app


def serve_application(
    server_settings: ServerSettings,
    application_service: ApplicationService | None = None,
) -> None:
    """Run the local-only server scaffold through uvicorn."""
    uvicorn = _load_uvicorn_module()
    if server_settings.reload:
        config = uvicorn.Config(
            "ollm.server.runtime:create_server_app",
            host=server_settings.host,
            port=server_settings.port,
            reload=True,
            log_level=server_settings.log_level,
            factory=True,
        )
    else:
        config = uvicorn.Config(
            create_server_app(application_service),
            host=server_settings.host,
            port=server_settings.port,
            reload=False,
            log_level=server_settings.log_level,
        )
    uvicorn.Server(config).run()

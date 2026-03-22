"""Optional local-only REST server and lifecycle boundary for oLLM."""

from collections.abc import Callable
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol, cast

from ollm.app.service import ApplicationService, build_default_application_service
from ollm.runtime.settings import ServerSettings
from ollm.server.dependencies import (
    SERVER_EXTRA_INSTALL_HINT,
    ServerDependenciesError,
)
from ollm.server.routes import HTTPExceptionFactory, register_rest_routes
from ollm.server.session_store import ServerSessionStore

LOCAL_SERVER_MODE = "local-only"
OPENAPI_SCHEMA_PATH = "/openapi.json"
OPENAPI_DOCS_PATH = "/docs"
OPENAPI_REDOC_PATH = "/redoc"
SERVER_DESCRIPTION = (
    "Local-only oLLM REST API with native runtime controls and an "
    "OpenAI-compatible chat surface."
)


class FastAPIApplication(Protocol):
    """Minimal FastAPI application protocol used by the local server."""

    state: object
    routes: object
    openapi_url: str | None
    docs_url: str | None
    redoc_url: str | None

    def openapi(self) -> object: ...

    def get(
        self,
        path: str,
        *,
        response_model: type[object],
        summary: str,
        tags: list[str],
    ) -> Callable[[Callable[..., object]], Callable[..., object]]: ...

    def post(
        self,
        path: str,
        *,
        response_model: type[object],
        summary: str,
        tags: list[str],
    ) -> Callable[[Callable[..., object]], Callable[..., object]]: ...


class FastAPIFactory(Protocol):
    """Protocol for the imported FastAPI application constructor."""

    def __call__(
        self,
        *,
        title: str,
        description: str,
        version: str,
        openapi_url: str | None = OPENAPI_SCHEMA_PATH,
        docs_url: str | None = OPENAPI_DOCS_PATH,
        redoc_url: str | None = OPENAPI_REDOC_PATH,
    ) -> FastAPIApplication: ...


class FastAPIModule(Protocol):
    """Protocol for the imported FastAPI module."""

    FastAPI: FastAPIFactory
    HTTPException: object


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
    """Create the local-only server application."""
    fastapi = _load_fastapi_module()
    app = fastapi.FastAPI(
        title="oLLM",
        description=SERVER_DESCRIPTION,
        version=_package_version(),
        openapi_url=OPENAPI_SCHEMA_PATH,
        docs_url=OPENAPI_DOCS_PATH,
        redoc_url=OPENAPI_REDOC_PATH,
    )
    resolved_application_service = (
        build_default_application_service()
        if application_service is None
        else application_service
    )
    setattr(app.state, "application_service", resolved_application_service)
    setattr(app.state, "server_mode", LOCAL_SERVER_MODE)
    setattr(app.state, "session_store", ServerSessionStore())
    register_rest_routes(
        app,
        cast(HTTPExceptionFactory, fastapi.HTTPException),
    )
    return app


def serve_application(
    server_settings: ServerSettings,
    application_service: ApplicationService | None = None,
) -> None:
    """Run the local-only server through uvicorn."""
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

from types import SimpleNamespace
from typing import cast

from ollm.app.doctor import DoctorService
from ollm.app.service import ApplicationService
from ollm.client import RuntimeClient
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader
from ollm.runtime.settings import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    ServerSettings,
)
from ollm.server.runtime import (
    SERVER_EXTRA_INSTALL_HINT,
    ServerDependenciesError,
    create_server_app,
    serve_application,
)
from tests.fakes import FakeDoctorService, FakeRuntimeExecutor, FakeRuntimeLoader


class FakeFastAPIApp:
    def __init__(self) -> None:
        self.state = SimpleNamespace()


class FakeFastAPIModule:
    def __init__(self) -> None:
        self.apps: list[tuple[str, str, str, FakeFastAPIApp]] = []

    def FastAPI(
        self,
        *,
        title: str,
        description: str,
        version: str,
    ) -> FakeFastAPIApp:
        app = FakeFastAPIApp()
        self.apps.append((title, description, version, app))
        return app


class FakeUvicornServer:
    def __init__(self, runs: list[object], config: object) -> None:
        self._runs = runs
        self._config = config

    def run(self) -> None:
        self._runs.append(self._config)


class FakeUvicornModule:
    def __init__(self) -> None:
        self.configs: list[dict[str, object]] = []
        self.runs: list[object] = []

    def Config(
        self,
        app: object,
        *,
        host: str,
        port: int,
        reload: bool,
        log_level: str,
        factory: bool = False,
    ) -> object:
        config = {
            "app": app,
            "host": host,
            "port": port,
            "reload": reload,
            "log_level": log_level,
            "factory": factory,
        }
        self.configs.append(config)
        return config

    def Server(self, config: object) -> FakeUvicornServer:
        return FakeUvicornServer(self.runs, config)


def build_application_service() -> ApplicationService:
    loader = FakeRuntimeLoader()
    executor = FakeRuntimeExecutor()
    doctor = FakeDoctorService()
    return ApplicationService(
        runtime_client=RuntimeClient(
            runtime_loader=cast(RuntimeLoader, loader),
            runtime_executor=cast(RuntimeExecutor, executor),
        ),
        doctor_service=cast(DoctorService, doctor),
    )


def test_create_server_app_attaches_application_service_to_app_state(
    monkeypatch,
) -> None:
    fastapi_module = FakeFastAPIModule()
    application_service = build_application_service()
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )

    app = create_server_app(application_service)

    assert getattr(app.state, "application_service") is application_service
    assert getattr(app.state, "server_mode") == "scaffold"
    assert fastapi_module.apps[0][0] == "oLLM"


def test_create_server_app_reports_missing_server_dependencies_cleanly(
    monkeypatch,
) -> None:
    def missing_fastapi():
        raise ServerDependenciesError(SERVER_EXTRA_INSTALL_HINT)

    monkeypatch.setattr("ollm.server.runtime._load_fastapi_module", missing_fastapi)

    try:
        create_server_app()
    except ServerDependenciesError as exc:
        assert SERVER_EXTRA_INSTALL_HINT in str(exc)
    else:
        raise AssertionError("create_server_app should report missing server deps")


def test_serve_application_builds_uvicorn_server_with_local_only_defaults(
    monkeypatch,
) -> None:
    fastapi_module = FakeFastAPIModule()
    uvicorn_module = FakeUvicornModule()
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )
    monkeypatch.setattr(
        "ollm.server.runtime._load_uvicorn_module",
        lambda: uvicorn_module,
    )

    serve_application(ServerSettings())

    assert uvicorn_module.configs[0]["host"] == DEFAULT_SERVER_HOST
    assert uvicorn_module.configs[0]["port"] == DEFAULT_SERVER_PORT
    assert uvicorn_module.configs[0]["reload"] is False
    assert uvicorn_module.configs[0]["factory"] is False
    assert isinstance(uvicorn_module.configs[0]["app"], FakeFastAPIApp)
    assert len(uvicorn_module.runs) == 1


def test_serve_application_uses_factory_import_string_when_reload_enabled(
    monkeypatch,
) -> None:
    uvicorn_module = FakeUvicornModule()
    monkeypatch.setattr(
        "ollm.server.runtime._load_uvicorn_module",
        lambda: uvicorn_module,
    )

    serve_application(
        ServerSettings(host="127.0.0.1", port=9001, reload=True, log_level="debug")
    )

    assert uvicorn_module.configs[0]["app"] == "ollm.server.runtime:create_server_app"
    assert uvicorn_module.configs[0]["reload"] is True
    assert uvicorn_module.configs[0]["factory"] is True
    assert uvicorn_module.configs[0]["log_level"] == "debug"


def test_serve_application_uses_injected_application_service_without_reload(
    monkeypatch,
) -> None:
    fastapi_module = FakeFastAPIModule()
    uvicorn_module = FakeUvicornModule()
    application_service = build_application_service()
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )
    monkeypatch.setattr(
        "ollm.server.runtime._load_uvicorn_module",
        lambda: uvicorn_module,
    )

    serve_application(ServerSettings(), application_service)

    app = cast(FakeFastAPIApp, uvicorn_module.configs[0]["app"])
    assert getattr(app.state, "application_service") is application_service

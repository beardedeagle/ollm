from collections.abc import Callable
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
from ollm.server.models import (
    HealthResponseModel,
    ModelInfoResponseModel,
    ModelsListResponseModel,
    PlanRequestModel,
    PlanResponseModel,
    PromptRequestModel,
    PromptResponseModel,
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
        self.routes: dict[tuple[str, str], Callable[..., object]] = {}

    def get(
        self,
        path: str,
        *,
        response_model: type[object],
        summary: str,
        tags: list[str],
    ):
        del response_model, summary, tags
        return self._register("GET", path)

    def post(
        self,
        path: str,
        *,
        response_model: type[object],
        summary: str,
        tags: list[str],
    ):
        del response_model, summary, tags
        return self._register("POST", path)

    def _register(self, method: str, path: str):
        def decorator(handler: Callable[..., object]) -> Callable[..., object]:
            self.routes[(method, path)] = handler
            return handler

        return decorator


class FakeHTTPException(Exception):
    def __init__(self, *, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeFastAPIModule:
    def __init__(self) -> None:
        self.apps: list[tuple[str, str, str, FakeFastAPIApp]] = []
        self.HTTPException = FakeHTTPException

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

    app = cast(FakeFastAPIApp, create_server_app(application_service))

    assert getattr(app.state, "application_service") is application_service
    assert getattr(app.state, "server_mode") == "scaffold"
    assert fastapi_module.apps[0][0] == "oLLM"
    assert ("GET", "/v1/health") in app.routes
    assert ("GET", "/v1/models") in app.routes
    assert ("GET", "/v1/models/{model_reference:path}") in app.routes
    assert ("POST", "/v1/plan") in app.routes
    assert ("POST", "/v1/prompt") in app.routes


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


def test_health_route_returns_scaffold_metadata(monkeypatch) -> None:
    fastapi_module = FakeFastAPIModule()
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )

    app = cast(FakeFastAPIApp, create_server_app(build_application_service()))
    health = cast(
        Callable[[], HealthResponseModel],
        app.routes[("GET", "/v1/health")],
    )
    payload = health()

    assert payload.ok is True
    assert payload.service == "ollm"
    assert payload.server_mode == "scaffold"


def test_plan_route_uses_application_service_for_runtime_planning(
    monkeypatch,
) -> None:
    fastapi_module = FakeFastAPIModule()
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )

    app = cast(FakeFastAPIApp, create_server_app(build_application_service()))
    plan_handler = cast(
        Callable[[PlanRequestModel], PlanResponseModel],
        app.routes[("POST", "/v1/plan")],
    )
    payload = plan_handler(
        PlanRequestModel.model_validate(
            {
                "runtime": {
                    "model_reference": "llama3-1B-chat",
                    "backend": "transformers-generic",
                }
            }
        )
    )

    assert payload.runtime_config.model_reference == "llama3-1B-chat"
    assert payload.runtime_plan.backend_id == "transformers-generic"


def test_prompt_route_executes_through_application_service(monkeypatch) -> None:
    fastapi_module = FakeFastAPIModule()
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )

    app = cast(FakeFastAPIApp, create_server_app(build_application_service()))
    prompt_handler = cast(
        Callable[[PromptRequestModel], PromptResponseModel],
        app.routes[("POST", "/v1/prompt")],
    )
    payload = prompt_handler(
        PromptRequestModel.model_validate(
            {"prompt": "hello server", "runtime": {"model_reference": "llama3-1B-chat"}}
        )
    )

    assert payload.text == "echo:hello server"
    assert payload.metadata == {}


def test_model_info_route_uses_application_service(monkeypatch) -> None:
    fastapi_module = FakeFastAPIModule()
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )

    app = cast(FakeFastAPIApp, create_server_app(build_application_service()))
    model_info = cast(
        Callable[[str], ModelInfoResponseModel],
        app.routes[("GET", "/v1/models/{model_reference:path}")],
    )
    payload = model_info("llama3-1B-chat")

    assert payload.model_reference == "llama3-1B-chat"
    assert payload.runtime_plan.backend_id == "optimized-native"


def test_models_list_route_returns_entries(monkeypatch) -> None:
    fastapi_module = FakeFastAPIModule()
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )

    app = cast(FakeFastAPIApp, create_server_app(build_application_service()))
    models_list = cast(
        Callable[[], ModelsListResponseModel],
        app.routes[("GET", "/v1/models")],
    )
    payload = models_list()

    assert payload.models


def test_plan_route_reports_bad_request_for_invalid_backend(monkeypatch) -> None:
    fastapi_module = FakeFastAPIModule()
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )

    app = cast(FakeFastAPIApp, create_server_app(build_application_service()))
    plan_handler = cast(
        Callable[[PlanRequestModel], PlanResponseModel],
        app.routes[("POST", "/v1/plan")],
    )

    try:
        plan_handler(
            PlanRequestModel.model_validate({"runtime": {"backend": "bad-backend"}})
        )
    except FakeHTTPException as exc:
        assert exc.status_code == 400
        assert "--backend must be one of" in exc.detail
    else:
        raise AssertionError("plan route should return a 400-style HTTP exception")

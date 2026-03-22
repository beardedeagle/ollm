from typing import cast

from ollm.runtime.settings import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    ServerSettings,
)
from ollm.server.runtime import (
    LOCAL_SERVER_MODE,
    OPENAPI_DOCS_PATH,
    OPENAPI_REDOC_PATH,
    OPENAPI_SCHEMA_PATH,
    SERVER_DESCRIPTION,
    SERVER_EXTRA_INSTALL_HINT,
    ServerDependenciesError,
    create_server_app,
    serve_application,
)
from tests.server_support import (
    FakeFastAPIApp,
    FakeFastAPIModule,
    FakeUvicornModule,
    build_application_service,
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
    assert getattr(app.state, "server_mode") == LOCAL_SERVER_MODE
    assert getattr(app.state, "session_store") is not None
    assert fastapi_module.apps[0][0] == "oLLM"
    assert fastapi_module.apps[0][1] == SERVER_DESCRIPTION
    assert app.openapi_url == OPENAPI_SCHEMA_PATH
    assert app.docs_url == OPENAPI_DOCS_PATH
    assert app.redoc_url == OPENAPI_REDOC_PATH
    assert ("GET", "/v1/health") in app.routes
    assert ("GET", "/v1/models") in app.routes
    assert ("GET", "/v1/ollm/models") in app.routes
    assert ("POST", "/v1/chat/completions") in app.routes
    assert ("POST", "/v1/prompt/stream") in app.routes
    assert ("POST", "/v1/sessions") in app.routes


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

import pytest
from typing import cast
from ollm.server.runtime import (
    LOCAL_SERVER_MODE,
    OPENAPI_DOCS_PATH,
    OPENAPI_REDOC_PATH,
    OPENAPI_SCHEMA_PATH,
    SERVER_DESCRIPTION,
    create_server_app,
)
from tests.server_support import build_application_service

pytest.importorskip("fastapi")


def test_server_app_publishes_openapi_and_docs_routes() -> None:
    app = create_server_app(build_application_service())
    assert getattr(app.state, "server_mode") == LOCAL_SERVER_MODE
    assert app.openapi_url == OPENAPI_SCHEMA_PATH
    assert app.docs_url == OPENAPI_DOCS_PATH
    assert app.redoc_url == OPENAPI_REDOC_PATH

    route_paths = {
        getattr(route, "path", "") for route in cast(list[object], app.routes)
    }
    assert OPENAPI_SCHEMA_PATH in route_paths
    assert OPENAPI_DOCS_PATH in route_paths
    assert OPENAPI_REDOC_PATH in route_paths

    schema = cast(dict[str, object], app.openapi())
    info = cast(dict[str, object], schema["info"])
    paths = cast(dict[str, object], schema["paths"])
    assert info["title"] == "oLLM"
    assert info["description"] == SERVER_DESCRIPTION
    for path in (
        "/v1/health",
        "/v1/models",
        "/v1/models/{model_reference}",
        "/v1/plan",
        "/v1/prompt",
        "/v1/prompt/stream",
        "/v1/sessions",
        "/v1/sessions/{session_id}",
        "/v1/sessions/{session_id}/prompt",
        "/v1/sessions/{session_id}/prompt/stream",
    ):
        assert path in paths

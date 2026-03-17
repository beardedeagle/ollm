from collections.abc import Callable
from typing import cast

from ollm.server.models import (
    HealthResponseModel,
    ModelInfoResponseModel,
    ModelsListResponseModel,
    PlanRequestModel,
    PlanResponseModel,
    PromptRequestModel,
    PromptResponseModel,
    SessionCreateRequestModel,
    SessionPromptRequestModel,
    SessionResponseModel,
)
from ollm.server.runtime import LOCAL_SERVER_MODE, create_server_app
from tests.server_support import (
    FakeFastAPIApp,
    FakeFastAPIModule,
    FakeHTTPException,
    build_application_service,
)


def test_health_route_returns_local_only_metadata(monkeypatch) -> None:
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
    assert payload.server_mode == LOCAL_SERVER_MODE


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


def test_prompt_stream_route_uses_sse_builder(monkeypatch) -> None:
    fastapi_module = FakeFastAPIModule()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )

    def fake_build_sse_response(execute) -> object:
        captured["response"] = "stream"
        captured["execute"] = execute
        return "stream"

    monkeypatch.setattr(
        "ollm.server.routes.build_sse_response", fake_build_sse_response
    )

    app = cast(FakeFastAPIApp, create_server_app(build_application_service()))
    stream_handler = cast(
        Callable[[PromptRequestModel], object],
        app.routes[("POST", "/v1/prompt/stream")],
    )
    payload = stream_handler(
        PromptRequestModel.model_validate(
            {"prompt": "stream this", "runtime": {"model_reference": "llama3-1B-chat"}}
        )
    )

    assert payload == "stream"
    assert captured["response"] == "stream"


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


def test_session_routes_create_inspect_and_prompt(monkeypatch) -> None:
    fastapi_module = FakeFastAPIModule()
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )

    app = cast(FakeFastAPIApp, create_server_app(build_application_service()))
    create_session = cast(
        Callable[[SessionCreateRequestModel], SessionResponseModel],
        app.routes[("POST", "/v1/sessions")],
    )
    get_session = cast(
        Callable[[str], SessionResponseModel],
        app.routes[("GET", "/v1/sessions/{session_id}")],
    )
    prompt_session = cast(
        Callable[[str, SessionPromptRequestModel], PromptResponseModel],
        app.routes[("POST", "/v1/sessions/{session_id}/prompt")],
    )

    created = create_session(
        SessionCreateRequestModel.model_validate(
            {"runtime": {"model_reference": "llama3-1B-chat"}}
        )
    )
    fetched = get_session(created.session_id)
    response = prompt_session(
        created.session_id,
        SessionPromptRequestModel.model_validate({"prompt": "hello session"}),
    )
    updated = get_session(created.session_id)

    assert fetched.session_id == created.session_id
    assert response.text == "echo:hello session"
    assert updated.messages


def test_session_stream_route_uses_sse_builder(monkeypatch) -> None:
    fastapi_module = FakeFastAPIModule()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )

    def fake_build_sse_response(execute) -> object:
        captured["response"] = "session-stream"
        captured["execute"] = execute
        return "session-stream"

    monkeypatch.setattr(
        "ollm.server.routes.build_sse_response", fake_build_sse_response
    )

    app = cast(FakeFastAPIApp, create_server_app(build_application_service()))
    create_session = cast(
        Callable[[SessionCreateRequestModel], SessionResponseModel],
        app.routes[("POST", "/v1/sessions")],
    )
    session_stream = cast(
        Callable[[str, SessionPromptRequestModel], object],
        app.routes[("POST", "/v1/sessions/{session_id}/prompt/stream")],
    )

    created = create_session(
        SessionCreateRequestModel.model_validate(
            {"runtime": {"model_reference": "llama3-1B-chat"}}
        )
    )
    payload = session_stream(
        created.session_id,
        SessionPromptRequestModel.model_validate({"prompt": "stream session"}),
    )

    assert payload == "session-stream"
    assert captured["response"] == "session-stream"


def test_session_routes_report_missing_session_as_bad_request(monkeypatch) -> None:
    fastapi_module = FakeFastAPIModule()
    monkeypatch.setattr(
        "ollm.server.runtime._load_fastapi_module",
        lambda: fastapi_module,
    )

    app = cast(FakeFastAPIApp, create_server_app(build_application_service()))
    get_session = cast(
        Callable[[str], SessionResponseModel],
        app.routes[("GET", "/v1/sessions/{session_id}")],
    )

    try:
        get_session("missing-session")
    except FakeHTTPException as exc:
        assert exc.status_code == 400
        assert "does not exist" in exc.detail
    else:
        raise AssertionError("missing session should return a 400-style HTTP exception")

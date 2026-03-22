from typing import cast

import pytest

from ollm.server.runtime import create_server_app
from tests.server_openai_support import (
    JsonResponseProtocol,
    build_test_client,
    json_object,
    payload_dict,
    payload_list,
)
from tests.server_support import build_application_service

pytest.importorskip("fastapi")


def test_openai_models_routes_use_openai_shapes() -> None:
    app = create_server_app(build_application_service())
    client = build_test_client(app)

    list_response = cast(JsonResponseProtocol, client.get("/v1/models"))
    detail_response = cast(
        JsonResponseProtocol, client.get("/v1/models/llama3-1B-chat")
    )

    list_payload = json_object(list_response)
    detail_payload = json_object(detail_response)
    model_entries = payload_list(list_payload["data"])
    first_entry = payload_dict(model_entries[0])
    assert list_response.status_code == 200
    assert list_payload["object"] == "list"
    assert model_entries
    assert first_entry["object"] == "model"
    assert detail_response.status_code == 200
    assert detail_payload["object"] == "model"
    assert detail_payload["id"] == "llama3-1B-chat"

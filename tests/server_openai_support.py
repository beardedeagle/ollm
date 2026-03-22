import json
from collections.abc import Awaitable, Callable, Iterator, MutableMapping
from importlib import import_module
from typing import Protocol, Self, cast

from ollm.runtime.settings import load_app_settings


class ServerTestClientProtocol(Protocol):
    def post(self, url: str, **kwargs) -> object: ...

    def get(self, url: str, **kwargs) -> object: ...

    def delete(self, url: str, **kwargs) -> object: ...

    def stream(self, method: str, url: str, **kwargs) -> object: ...


class StreamResponseProtocol(Protocol):
    status_code: int

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type, exc, tb) -> bool | None: ...

    def iter_lines(self) -> Iterator[str]: ...


class JsonResponseProtocol(Protocol):
    status_code: int

    def json(self) -> object: ...


class AsgiAppProtocol(Protocol):
    def __call__(
        self,
        scope: MutableMapping[str, object],
        receive: Callable[[], Awaitable[MutableMapping[str, object]]],
        send: Callable[[MutableMapping[str, object]], Awaitable[None]],
        /,
    ) -> Awaitable[None]: ...


def build_test_client(app: object) -> ServerTestClientProtocol:
    testclient_module = import_module("fastapi.testclient")
    return cast(
        ServerTestClientProtocol,
        testclient_module.TestClient(cast(AsgiAppProtocol, app)),
    )


def json_object(response: JsonResponseProtocol) -> dict[str, object]:
    return cast(dict[str, object], response.json())


def payload_dict(value: object) -> dict[str, object]:
    return cast(dict[str, object], value)


def payload_list(value: object) -> list[object]:
    return cast(list[object], value)


def configure_response_store(monkeypatch, *, backend: str) -> None:
    settings = load_app_settings()
    monkeypatch.setattr(
        "ollm.server.runtime.load_app_settings",
        lambda: settings.model_copy(
            update={
                "server": settings.server.model_copy(
                    update={"response_store_backend": backend}
                )
            }
        ),
    )


def decode_stream_lines(lines: list[str]) -> tuple[list[str], list[dict[str, object]]]:
    events: list[str] = []
    payloads: list[dict[str, object]] = []
    for index in range(0, len(lines), 2):
        events.append(lines[index].removeprefix("event: "))
        payloads.append(
            payload_dict(json.loads(lines[index + 1].removeprefix("data: ")))
        )
    return events, payloads

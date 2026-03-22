"""Configurable storage backends for OpenAI-compatible response objects."""

from dataclasses import dataclass, field
from importlib import import_module
from threading import RLock
from typing import Protocol, cast

from ollm.app.types import Message
from ollm.runtime.settings import ServerSettings
from ollm.server.openai_response_models import OpenAIResponseResponseModel


@dataclass(frozen=True, slots=True)
class StoredOpenAIResponseFunctionCall:
    """Stored function-call output item used for response chaining."""

    item_id: str
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class StoredOpenAIResponseFunctionCallOutput:
    """Stored tool-result input item used for response chaining."""

    call_id: str
    output_text: str


@dataclass(slots=True)
class StoredOpenAIResponse:
    """Stored response payload plus conversation history for chaining."""

    response: OpenAIResponseResponseModel
    conversation_items: list[
        Message
        | StoredOpenAIResponseFunctionCall
        | StoredOpenAIResponseFunctionCallOutput
    ]


class OpenAIResponseStore(Protocol):
    """Storage boundary for response retrieval and chaining."""

    enabled: bool

    def put(self, stored_response: StoredOpenAIResponse) -> None: ...

    def get(self, response_id: str) -> StoredOpenAIResponse | None: ...

    def require(self, response_id: str) -> StoredOpenAIResponse: ...


@dataclass(slots=True)
class DisabledOpenAIResponseStore:
    """No-op response store used when persistence is disabled."""

    enabled: bool = False

    def put(self, stored_response: StoredOpenAIResponse) -> None:
        del stored_response

    def get(self, response_id: str) -> StoredOpenAIResponse | None:
        del response_id
        return None

    def require(self, response_id: str) -> StoredOpenAIResponse:
        del response_id
        raise ValueError("Responses storage is disabled for this server")


@dataclass(slots=True)
class MemoryOpenAIResponseStore:
    """Thread-safe in-memory response store."""

    enabled: bool = True
    _responses: dict[str, StoredOpenAIResponse] = field(
        default_factory=dict, repr=False
    )
    _lock: RLock = field(default_factory=RLock, repr=False)

    def put(self, stored_response: StoredOpenAIResponse) -> None:
        with self._lock:
            self._responses[stored_response.response.id] = stored_response

    def get(self, response_id: str) -> StoredOpenAIResponse | None:
        with self._lock:
            return self._responses.get(response_id)

    def require(self, response_id: str) -> StoredOpenAIResponse:
        stored_response = self.get(response_id)
        if stored_response is None:
            raise ValueError(f"Response '{response_id}' does not exist")
        return stored_response


def build_openai_response_store(server_settings: ServerSettings) -> OpenAIResponseStore:
    """Build the configured response-store backend for the local server."""
    backend = server_settings.response_store_backend
    if backend == "none":
        return DisabledOpenAIResponseStore()
    if backend == "memory":
        return MemoryOpenAIResponseStore()
    return _load_plugin_store(server_settings)


def _load_plugin_store(server_settings: ServerSettings) -> OpenAIResponseStore:
    factory_path = server_settings.response_store_factory
    if factory_path is None:
        raise ValueError(
            "response_store_factory is required when response_store_backend=plugin"
        )
    module_name, separator, attribute_name = factory_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "response_store_factory must use the form 'package.module:factory'"
        )
    module = import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if factory is None or not callable(factory):
        raise ValueError(
            f"response_store_factory '{factory_path}' did not resolve to a callable"
        )
    return cast(OpenAIResponseStoreFactory, factory)(server_settings)


class OpenAIResponseStoreFactory(Protocol):
    """Callable factory contract for plugin-backed response stores."""

    def __call__(self, server_settings: ServerSettings) -> OpenAIResponseStore: ...

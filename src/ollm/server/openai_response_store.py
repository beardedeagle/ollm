"""Thread-safe in-memory store for OpenAI-compatible response objects."""

from dataclasses import dataclass, field
from threading import RLock

from ollm.app.types import Message
from ollm.server.openai_response_models import OpenAIResponseResponseModel


@dataclass(slots=True)
class StoredOpenAIResponse:
    """Stored response payload plus conversation history for chaining."""

    response: OpenAIResponseResponseModel
    conversation_messages: list[Message]


@dataclass(slots=True)
class OpenAIResponseStore:
    """Manage in-memory Responses API objects with explicit locking."""

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

"""Thread-safe in-memory server session store for oLLM."""

from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4

from ollm.app.service import ApplicationService
from ollm.app.session import ChatSession
from ollm.app.types import PromptResponse
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.streaming import StreamSink


@dataclass(slots=True)
class ServerSessionHandle:
    """Own one in-memory chat session plus its execution lock."""

    session_id: str
    session: ChatSession
    _lock: RLock = field(default_factory=RLock, repr=False)

    def prompt_text(
        self,
        prompt: str,
        *,
        sink: StreamSink | None = None,
    ) -> PromptResponse:
        with self._lock:
            return self.session.prompt_text(prompt, sink=sink)


@dataclass(slots=True)
class ServerSessionStore:
    """Manage server-side chat sessions with explicit locking."""

    _sessions: dict[str, ServerSessionHandle] = field(default_factory=dict, repr=False)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def create(
        self,
        application_service: ApplicationService,
        *,
        runtime_config: RuntimeConfig,
        generation_config: GenerationConfig,
        session_name: str,
        system_prompt: str,
    ) -> ServerSessionHandle:
        session_id = str(uuid4())
        session = application_service.create_session(
            runtime_config=runtime_config,
            generation_config=generation_config,
            session_name=session_name,
            system_prompt=system_prompt,
        )
        handle = ServerSessionHandle(session_id=session_id, session=session)
        with self._lock:
            self._sessions[session_id] = handle
        return handle

    def get(self, session_id: str) -> ServerSessionHandle | None:
        with self._lock:
            return self._sessions.get(session_id)

    def require(self, session_id: str) -> ServerSessionHandle:
        handle = self.get(session_id)
        if handle is None:
            raise ValueError(f"Session '{session_id}' does not exist")
        return handle

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

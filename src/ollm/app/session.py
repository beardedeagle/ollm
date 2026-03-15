from dataclasses import dataclass, field
from pathlib import Path

from ollm.app.history import TRANSCRIPT_VERSION, load_transcript, save_transcript
from ollm.app.types import (
    ContentKind,
    ContentPart,
    Message,
    MessageRole,
    PromptRequest,
    PromptResponse,
    Transcript,
)
from ollm.runtime.config import DEFAULT_SYSTEM_PROMPT, GenerationConfig, RuntimeConfig
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import LoadedRuntime, RuntimeLoader
from ollm.runtime.streaming import StreamSink


@dataclass(slots=True)
class ChatSession:
    runtime_loader: RuntimeLoader
    runtime_executor: RuntimeExecutor
    runtime_config: RuntimeConfig
    generation_config: GenerationConfig
    session_name: str = "default"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    messages: list[Message] = field(default_factory=list)
    autosave_path: Path | None = None
    _loaded_runtime: LoadedRuntime | None = field(init=False, default=None, repr=False)

    def clear(self) -> None:
        self.messages.clear()
        self._autosave()

    def reset(self) -> None:
        self.messages.clear()
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self._autosave()

    def prompt_text(self, text: str, sink: StreamSink | None = None) -> PromptResponse:
        return self.prompt_parts([ContentPart.text(text)], sink=sink)

    def prompt_parts(
        self, parts: list[ContentPart], sink: StreamSink | None = None
    ) -> PromptResponse:
        if not parts:
            raise ValueError("A chat prompt requires at least one content part")

        requires_multimodal = any(part.kind is not ContentKind.TEXT for part in parts)
        runtime = self._ensure_runtime(requires_multimodal)
        user_message = Message(role=MessageRole.USER, content=parts)
        request = PromptRequest(
            runtime_config=runtime.config,
            generation_config=self.generation_config,
            messages=self._request_messages(user_message),
        )
        response = self.runtime_executor.execute(runtime, request, sink=sink)
        self.messages.append(user_message)
        self.messages.append(response.assistant_message)
        self._autosave()
        return response

    def retry_last(self, sink: StreamSink | None = None) -> PromptResponse:
        if len(self.messages) < 2:
            raise ValueError("There is no previous exchange to retry")
        assistant_message = self.messages.pop()
        user_message = self.messages.pop()
        if (
            assistant_message.role is not MessageRole.ASSISTANT
            or user_message.role is not MessageRole.USER
        ):
            raise ValueError(
                "The current session does not end with a retryable user/assistant exchange"
            )
        return self.prompt_parts(user_message.content, sink=sink)

    def undo_last_exchange(self) -> None:
        if len(self.messages) < 2:
            raise ValueError("There is no previous exchange to undo")
        assistant_message = self.messages.pop()
        user_message = self.messages.pop()
        if (
            assistant_message.role is not MessageRole.ASSISTANT
            or user_message.role is not MessageRole.USER
        ):
            raise ValueError(
                "The current session does not end with a removable user/assistant exchange"
            )
        self._autosave()

    def save(self, path: Path) -> None:
        save_transcript(path, self.transcript())
        self.autosave_path = path

    def load(self, path: Path) -> None:
        transcript = load_transcript(path)
        self.session_name = transcript.session_name
        self.system_prompt = transcript.system_prompt
        self.runtime_config.model_reference = transcript.model_reference
        self.messages = list(transcript.messages)
        self.autosave_path = path
        self._loaded_runtime = None

    def transcript(self) -> Transcript:
        return Transcript(
            version=TRANSCRIPT_VERSION,
            session_name=self.session_name,
            model_reference=self.runtime_config.model_reference,
            system_prompt=self.system_prompt,
            messages=list(self.messages),
        )

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        self._autosave()

    def set_model(self, model_reference: str) -> None:
        self.runtime_config.model_reference = model_reference
        self.runtime_config.multimodal = False
        self._loaded_runtime = None
        self._autosave()

    def _ensure_runtime(self, requires_multimodal: bool) -> LoadedRuntime:
        if requires_multimodal and not self.runtime_config.multimodal:
            self.runtime_config.multimodal = True
            self._loaded_runtime = None
        if self._loaded_runtime is None:
            self._loaded_runtime = self.runtime_loader.load(self.runtime_config)
        return self._loaded_runtime

    def _request_messages(self, user_message: Message) -> list[Message]:
        request_messages: list[Message] = []
        if self.system_prompt:
            request_messages.append(Message.system_text(self.system_prompt))
        request_messages.extend(self.messages)
        request_messages.append(user_message)
        return request_messages

    def _autosave(self) -> None:
        if self.autosave_path is None:
            return
        save_transcript(self.autosave_path, self.transcript())

from dataclasses import dataclass, field
from enum import Enum

from ollm.runtime.config import GenerationConfig, RuntimeConfig


class ContentKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ContentPart:
    kind: ContentKind
    value: str

    @classmethod
    def text(cls, value: str):
        return cls(kind=ContentKind.TEXT, value=value)

    @classmethod
    def image(cls, value: str):
        return cls(kind=ContentKind.IMAGE, value=value)

    @classmethod
    def audio(cls, value: str):
        return cls(kind=ContentKind.AUDIO, value=value)

    def as_transformers_content(self) -> dict[str, str]:
        if self.kind is ContentKind.TEXT:
            return {"type": "text", "text": self.value}
        if self.kind is ContentKind.IMAGE:
            return {"type": "image", "image": self.value}
        return {"type": "audio", "url": self.value}

    def as_dict(self) -> dict[str, str]:
        return {"type": self.kind.value, "value": self.value}

    @classmethod
    def from_dict(cls, payload: dict[str, str]):
        kind = ContentKind(payload["type"])
        value = payload["value"]
        if kind is ContentKind.TEXT:
            return cls.text(value)
        if kind is ContentKind.IMAGE:
            return cls.image(value)
        return cls.audio(value)


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: list[ContentPart]

    @classmethod
    def system_text(cls, text: str):
        return cls(role=MessageRole.SYSTEM, content=[ContentPart.text(text)])

    @classmethod
    def user_text(cls, text: str):
        return cls(role=MessageRole.USER, content=[ContentPart.text(text)])

    @classmethod
    def assistant_text(cls, text: str):
        return cls(role=MessageRole.ASSISTANT, content=[ContentPart.text(text)])

    def as_transformers_message(self) -> dict[str, object]:
        if len(self.content) == 1 and self.content[0].kind is ContentKind.TEXT:
            return {"role": self.role.value, "content": self.content[0].value}
        return {
            "role": self.role.value,
            "content": [part.as_transformers_content() for part in self.content],
        }

    def text_content(self) -> str:
        return "".join(part.value for part in self.content if part.kind is ContentKind.TEXT)

    def contains_non_text(self) -> bool:
        return any(part.kind is not ContentKind.TEXT for part in self.content)

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "content": [part.as_dict() for part in self.content],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        role = MessageRole(str(payload["role"]))
        parts = [ContentPart.from_dict(part) for part in payload["content"]]
        return cls(role=role, content=parts)


@dataclass(slots=True)
class PromptRequest:
    runtime_config: RuntimeConfig
    generation_config: GenerationConfig
    messages: list[Message]


@dataclass(slots=True)
class PromptResponse:
    text: str
    assistant_message: Message
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Transcript:
    version: int
    session_name: str
    model_reference: str
    system_prompt: str
    messages: list[Message]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "session_name": self.session_name,
            "model_reference": self.model_reference,
            "system_prompt": self.system_prompt,
            "messages": [message.as_dict() for message in self.messages],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]):
        version = int(payload["version"])
        session_name = str(payload["session_name"])
        model_reference = str(payload["model_reference"])
        system_prompt = str(payload.get("system_prompt", ""))
        messages = [Message.from_dict(message) for message in payload.get("messages", [])]
        return cls(
            version=version,
            session_name=session_name,
            model_reference=model_reference,
            system_prompt=system_prompt,
            messages=messages,
        )

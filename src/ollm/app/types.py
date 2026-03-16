from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Self, cast

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
    def text(cls, value: str) -> Self:
        return cls(kind=ContentKind.TEXT, value=value)

    @classmethod
    def image(cls, value: str) -> Self:
        return cls(kind=ContentKind.IMAGE, value=value)

    @classmethod
    def audio(cls, value: str) -> Self:
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
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        kind = ContentKind(_require_string(payload, "type"))
        value = _require_string(payload, "value")
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
    def system_text(cls, text: str) -> Self:
        return cls(role=MessageRole.SYSTEM, content=[ContentPart.text(text)])

    @classmethod
    def user_text(cls, text: str) -> Self:
        return cls(role=MessageRole.USER, content=[ContentPart.text(text)])

    @classmethod
    def assistant_text(cls, text: str) -> Self:
        return cls(role=MessageRole.ASSISTANT, content=[ContentPart.text(text)])

    def as_transformers_message(self) -> dict[str, object]:
        if len(self.content) == 1 and self.content[0].kind is ContentKind.TEXT:
            return {"role": self.role.value, "content": self.content[0].value}
        return {
            "role": self.role.value,
            "content": [part.as_transformers_content() for part in self.content],
        }

    def text_content(self) -> str:
        return "".join(
            part.value for part in self.content if part.kind is ContentKind.TEXT
        )

    def contains_non_text(self) -> bool:
        return any(part.kind is not ContentKind.TEXT for part in self.content)

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "content": [part.as_dict() for part in self.content],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        role = MessageRole(_require_string(payload, "role"))
        parts = [
            ContentPart.from_dict(_require_mapping(part, f"content[{index}]"))
            for index, part in enumerate(_require_sequence(payload, "content"))
        ]
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
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        version = _require_int(payload, "version")
        session_name = _require_string(payload, "session_name")
        model_reference = _require_string(payload, "model_reference")
        system_prompt = _optional_string(payload.get("system_prompt"), default="")
        messages = [
            Message.from_dict(_require_mapping(message, f"messages[{index}]"))
            for index, message in enumerate(
                _optional_sequence(payload.get("messages"), default=())
            )
        ]
        return cls(
            version=version,
            session_name=session_name,
            model_reference=model_reference,
            system_prompt=system_prompt,
            messages=messages,
        )


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _require_sequence(
    payload: Mapping[str, object], field_name: str
) -> Sequence[object]:
    value = payload.get(field_name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list")
    return value


def _optional_sequence(value: object, *, default: Sequence[object]) -> Sequence[object]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("messages must be a list")
    return value


def _require_string(payload: Mapping[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _optional_string(value: object, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("system_prompt must be a string")
    return value


def _require_int(payload: Mapping[str, object], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
    raise ValueError(f"{field_name} must be an integer")

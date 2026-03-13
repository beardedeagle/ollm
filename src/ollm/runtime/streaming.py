from dataclasses import dataclass, field
from typing import Protocol

from transformers import TextStreamer


class StreamSink(Protocol):
    def on_status(self, message: str) -> None:
        ...

    def on_text(self, text: str) -> None:
        ...

    def on_complete(self, text: str) -> None:
        ...


class NullStreamSink:
    def on_status(self, message: str) -> None:
        del message

    def on_text(self, text: str) -> None:
        del text

    def on_complete(self, text: str) -> None:
        del text


@dataclass(slots=True)
class CollectingStreamSink:
    statuses: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    completed_text: str = ""

    def on_status(self, message: str) -> None:
        self.statuses.append(message)

    def on_text(self, text: str) -> None:
        self.chunks.append(text)

    def on_complete(self, text: str) -> None:
        self.completed_text = text


class BufferedTextStreamer(TextStreamer):
    def __init__(self, tokenizer, sink: StreamSink, skip_prompt: bool = True, skip_special_tokens: bool = False):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self._sink = sink
        self._chunks: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self._chunks)

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        if text:
            self._chunks.append(text)
            self._sink.on_text(text)
        if stream_end:
            self._sink.on_complete(self.text)


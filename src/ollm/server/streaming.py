"""SSE streaming helpers for the oLLM server surface."""

import json
from collections.abc import Callable, Iterable, Iterator
from importlib import import_module
from queue import Queue
from threading import Thread
from typing import Protocol, cast

from ollm.runtime.streaming import StreamSink
from ollm.server.dependencies import SERVER_EXTRA_INSTALL_HINT, ServerDependenciesError


class StreamingResponseFactory(Protocol):
    """Protocol for the imported StreamingResponse constructor."""

    def __call__(self, content: Iterable[str], *, media_type: str) -> object: ...


class EventStreamSink(StreamSink):
    """Stream sink that serializes runtime callbacks into SSE events."""

    def __init__(self, queue: Queue[str | None]) -> None:
        self._queue = queue

    def on_status(self, message: str) -> None:
        self._queue.put(_sse_event("status", {"message": message}))

    def on_text(self, text: str) -> None:
        self._queue.put(_sse_event("text", {"delta": text}))

    def on_complete(self, text: str) -> None:
        self._queue.put(_sse_event("complete", {"text": text}))


def _load_streaming_response_factory() -> StreamingResponseFactory:
    try:
        responses_module = import_module("fastapi.responses")
    except ModuleNotFoundError as exc:
        raise ServerDependenciesError(SERVER_EXTRA_INSTALL_HINT) from exc
    return cast(StreamingResponseFactory, responses_module.StreamingResponse)


def _sse_event(event: str, payload: dict[str, str]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _event_iterator(execute: Callable[[StreamSink], None]) -> Iterator[str]:
    queue: Queue[str | None] = Queue()
    sink = EventStreamSink(queue)

    def run() -> None:
        try:
            execute(sink)
        except Exception as exc:
            queue.put(_sse_event("error", {"detail": str(exc)}))
        finally:
            queue.put(None)

    Thread(target=run, daemon=True).start()

    while True:
        item = queue.get()
        if item is None:
            return
        yield item


def build_sse_response(execute: Callable[[StreamSink], None]) -> object:
    """Build an SSE response around a blocking runtime execution callback."""
    streaming_response = _load_streaming_response_factory()
    return streaming_response(
        _event_iterator(execute),
        media_type="text/event-stream",
    )

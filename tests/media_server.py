import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(frozen=True, slots=True)
class MediaResponse:
    body: bytes
    content_type: str
    status_code: int = 200


@dataclass(slots=True)
class MediaFixtureServer:
    responses: dict[str, MediaResponse]
    request_paths: list[str] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = field(init=False, default=None, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)

    @property
    def base_url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server = server
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _handler(self):
        fixture = self

        class MediaHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                fixture.request_paths.append(self.path)
                response = fixture.responses.get(self.path)
                if response is None:
                    _write_response(
                        handler=self,
                        status_code=404,
                        body=b"missing",
                        content_type="text/plain",
                    )
                    return
                _write_response(
                    handler=self,
                    status_code=response.status_code,
                    body=response.body,
                    content_type=response.content_type,
                )

            def log_message(self, format: str, *args) -> None:
                del format, args

        return MediaHandler


def _write_response(
    *,
    handler: BaseHTTPRequestHandler,
    status_code: int,
    body: bytes,
    content_type: str,
) -> None:
    handler.send_response(status_code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)

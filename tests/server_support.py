from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

from ollm.app.doctor import DoctorService
from ollm.app.service import ApplicationService
from ollm.client import RuntimeClient
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader
from tests.fakes import FakeDoctorService, FakeRuntimeExecutor, FakeRuntimeLoader


class FakeFastAPIApp:
    def __init__(self) -> None:
        self.state = SimpleNamespace()
        self.routes: dict[tuple[str, str], Callable[..., object]] = {}

    def get(
        self,
        path: str,
        *,
        response_model: type[object],
        summary: str,
        tags: list[str],
    ):
        del response_model, summary, tags
        return self._register("GET", path)

    def post(
        self,
        path: str,
        *,
        response_model: type[object],
        summary: str,
        tags: list[str],
    ):
        del response_model, summary, tags
        return self._register("POST", path)

    def _register(self, method: str, path: str):
        def decorator(handler: Callable[..., object]) -> Callable[..., object]:
            self.routes[(method, path)] = handler
            return handler

        return decorator


class FakeHTTPException(Exception):
    def __init__(self, *, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeFastAPIModule:
    def __init__(self) -> None:
        self.apps: list[tuple[str, str, str, FakeFastAPIApp]] = []
        self.HTTPException = FakeHTTPException

    def FastAPI(
        self,
        *,
        title: str,
        description: str,
        version: str,
    ) -> FakeFastAPIApp:
        app = FakeFastAPIApp()
        self.apps.append((title, description, version, app))
        return app


class FakeUvicornServer:
    def __init__(self, runs: list[object], config: object) -> None:
        self._runs = runs
        self._config = config

    def run(self) -> None:
        self._runs.append(self._config)


class FakeUvicornModule:
    def __init__(self) -> None:
        self.configs: list[dict[str, object]] = []
        self.runs: list[object] = []

    def Config(
        self,
        app: object,
        *,
        host: str,
        port: int,
        reload: bool,
        log_level: str,
        factory: bool = False,
    ) -> object:
        config = {
            "app": app,
            "host": host,
            "port": port,
            "reload": reload,
            "log_level": log_level,
            "factory": factory,
        }
        self.configs.append(config)
        return config

    def Server(self, config: object) -> FakeUvicornServer:
        return FakeUvicornServer(self.runs, config)


def build_application_service() -> ApplicationService:
    loader = FakeRuntimeLoader()
    executor = FakeRuntimeExecutor()
    doctor = FakeDoctorService()
    return ApplicationService(
        runtime_client=RuntimeClient(
            runtime_loader=cast(RuntimeLoader, loader),
            runtime_executor=cast(RuntimeExecutor, executor),
        ),
        doctor_service=cast(DoctorService, doctor),
    )

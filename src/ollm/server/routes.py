"""REST route registration for the oLLM server surface."""

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, cast

from ollm.app.service import ApplicationService
from ollm.app.types import ContentPart
from ollm.runtime.catalog import list_model_catalog
from ollm.runtime.inspection import merged_runtime_payload
from ollm.runtime.settings import (
    GenerationConfigOverrides,
    RuntimeConfigOverrides,
    load_app_settings,
    resolve_generation_config,
    resolve_runtime_config,
)
from ollm.server.models import (
    HealthResponseModel,
    ModelInfoResponseModel,
    ModelsListResponseModel,
    PlanRequestModel,
    PlanResponseModel,
    PromptRequestModel,
    PromptResponseModel,
)


class HTTPExceptionFactory(Protocol):
    """Protocol for FastAPI HTTPException construction."""

    def __call__(self, *, status_code: int, detail: str) -> Exception: ...


RouteDecorator = Callable[[Callable[..., object]], Callable[..., object]]


class RouteRegistryApp(Protocol):
    """Protocol for the route registration surface used by FastAPI."""

    state: object

    def get(
        self,
        path: str,
        *,
        response_model: type[object],
        summary: str,
        tags: list[str],
    ) -> RouteDecorator: ...

    def post(
        self,
        path: str,
        *,
        response_model: type[object],
        summary: str,
        tags: list[str],
    ) -> RouteDecorator: ...


def _package_version() -> str:
    try:
        return version("ollm")
    except PackageNotFoundError:
        return "0.0.0"


def _http_bad_request(
    http_exception: HTTPExceptionFactory,
    exc: ValueError,
) -> Exception:
    return http_exception(status_code=400, detail=str(exc))


def _build_runtime_config(
    request_runtime,
):
    settings = load_app_settings()
    return resolve_runtime_config(
        settings.runtime,
        RuntimeConfigOverrides(
            model_reference=request_runtime.model_reference,
            models_dir=request_runtime.models_dir,
            device=request_runtime.device,
            backend=request_runtime.backend,
            adapter_dir=request_runtime.adapter_dir,
            multimodal=request_runtime.multimodal,
            use_specialization=request_runtime.use_specialization,
            cache_dir=request_runtime.cache_dir,
            use_cache=request_runtime.use_cache,
            offload_cpu_layers=request_runtime.offload_cpu_layers,
            offload_gpu_layers=request_runtime.offload_gpu_layers,
            force_download=request_runtime.force_download,
            stats=request_runtime.stats,
            verbose=request_runtime.verbose,
            quiet=request_runtime.quiet,
        ),
    )


def _build_generation_config(request_generation):
    settings = load_app_settings()
    return resolve_generation_config(
        settings.generation,
        GenerationConfigOverrides(
            max_new_tokens=request_generation.max_new_tokens,
            temperature=request_generation.temperature,
            top_p=request_generation.top_p,
            top_k=request_generation.top_k,
            seed=request_generation.seed,
            stream=False,
        ),
    )


def _model_dir(models_dir: str | None) -> Path:
    settings = load_app_settings()
    base_dir = settings.runtime.models_dir if models_dir is None else Path(models_dir)
    return base_dir.expanduser().resolve()


def _model_info_payload(
    application_service: ApplicationService,
    *,
    model_reference: str,
    model_dir: Path,
    backend: str | None,
    multimodal: bool | None,
    use_specialization: bool | None,
    discovery_source: str | None,
) -> ModelInfoResponseModel:
    settings = load_app_settings()
    runtime_config = resolve_runtime_config(
        settings.runtime,
        RuntimeConfigOverrides(
            model_reference=model_reference,
            models_dir=model_dir,
            backend=backend,
            multimodal=multimodal,
            use_specialization=use_specialization,
        ),
    )
    resolved_model = application_service.resolve_model(
        model_reference, runtime_config.resolved_models_dir()
    )
    materialized = bool(
        resolved_model.model_path is not None and resolved_model.model_path.exists()
    )
    runtime_plan = application_service.plan(runtime_config)
    payload = merged_runtime_payload(
        resolved_model,
        runtime_plan,
        materialized=materialized,
    )
    return ModelInfoResponseModel.model_validate(
        {
            **payload,
            "discovery_source": discovery_source,
        }
    )


def register_rest_routes(
    app: RouteRegistryApp,
    http_exception: HTTPExceptionFactory,
) -> None:
    """Register the first REST API surface on a FastAPI application."""
    application_service = cast(
        ApplicationService,
        getattr(app.state, "application_service"),
    )

    @app.get(
        "/v1/health",
        response_model=HealthResponseModel,
        summary="Health check",
        tags=["system"],
    )
    def health() -> HealthResponseModel:
        return HealthResponseModel(
            ok=True,
            service="ollm",
            version=_package_version(),
            server_mode=cast(str, getattr(app.state, "server_mode")),
        )

    @app.get(
        "/v1/models",
        response_model=ModelsListResponseModel,
        summary="List known and local models",
        tags=["models"],
    )
    def list_models(
        installed: bool = False,
        backend: str | None = None,
        no_specialization: bool | None = None,
        models_dir: str | None = None,
    ) -> ModelsListResponseModel:
        try:
            model_dir = _model_dir(models_dir)
            entries: list[ModelInfoResponseModel] = []
            seen_paths: set[str] = set()
            use_specialization = (
                None if no_specialization is None else not no_specialization
            )

            for entry in list_model_catalog():
                payload = _model_info_payload(
                    application_service,
                    model_reference=entry.model_id,
                    model_dir=model_dir,
                    backend=backend,
                    multimodal=None,
                    use_specialization=use_specialization,
                    discovery_source="built-in",
                )
                if installed and not payload.materialized:
                    continue
                entries.append(payload)
                if payload.path is not None:
                    seen_paths.add(payload.path)

            for resolved_model in application_service.discover_local_models(model_dir):
                payload = _model_info_payload(
                    application_service,
                    model_reference=resolved_model.reference.raw,
                    model_dir=model_dir,
                    backend=backend,
                    multimodal=None,
                    use_specialization=use_specialization,
                    discovery_source="discovered-local",
                )
                if payload.path is not None and payload.path in seen_paths:
                    continue
                if installed and not payload.materialized:
                    continue
                entries.append(payload)

            entries.sort(key=lambda item: (item.source_kind, item.model_reference))
            return ModelsListResponseModel(models=entries)
        except ValueError as exc:
            raise _http_bad_request(http_exception, exc) from exc

    @app.get(
        "/v1/models/{model_reference:path}",
        response_model=ModelInfoResponseModel,
        summary="Inspect one model reference",
        tags=["models"],
    )
    def model_info(
        model_reference: str,
        models_dir: str | None = None,
        backend: str | None = None,
        multimodal: bool | None = None,
        no_specialization: bool | None = None,
    ) -> ModelInfoResponseModel:
        try:
            return _model_info_payload(
                application_service,
                model_reference=model_reference,
                model_dir=_model_dir(models_dir),
                backend=backend,
                multimodal=multimodal,
                use_specialization=(
                    None if no_specialization is None else not no_specialization
                ),
                discovery_source=None,
            )
        except ValueError as exc:
            raise _http_bad_request(http_exception, exc) from exc

    @app.post(
        "/v1/plan",
        response_model=PlanResponseModel,
        summary="Inspect a runtime plan",
        tags=["runtime"],
    )
    def plan(request: PlanRequestModel) -> PlanResponseModel:
        try:
            payload = application_service.describe_plan(
                _build_runtime_config(request.runtime)
            )
        except ValueError as exc:
            raise _http_bad_request(http_exception, exc) from exc
        return PlanResponseModel.model_validate(payload)

    @app.post(
        "/v1/prompt",
        response_model=PromptResponseModel,
        summary="Execute a prompt",
        tags=["runtime"],
    )
    def prompt(request: PromptRequestModel) -> PromptResponseModel:
        try:
            response = application_service.prompt_parts(
                [ContentPart.text(request.prompt)],
                runtime_config=_build_runtime_config(request.runtime),
                generation_config=_build_generation_config(request.generation),
                system_prompt=request.system_prompt,
            )
        except ValueError as exc:
            raise _http_bad_request(http_exception, exc) from exc
        return PromptResponseModel(text=response.text, metadata=dict(response.metadata))

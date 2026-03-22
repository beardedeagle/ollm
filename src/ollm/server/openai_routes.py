"""OpenAI-compatible route registration for the local oLLM server."""

from collections.abc import Callable

from ollm.app.service import ApplicationService
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.settings import (
    GenerationConfigOverrides,
    RuntimeConfigOverrides,
    load_app_settings,
    resolve_generation_config,
    resolve_runtime_config,
)
from ollm.server.models import (
    ModelInfoResponseModel,
    OpenAIChatCompletionRequestModel,
    OpenAIChatCompletionResponseModel,
    OpenAIModelResponseModel,
    OpenAIModelsListResponseModel,
)
from ollm.server.openai_compat import (
    build_openai_chat_completion_response,
    build_openai_chat_sse_response,
    build_openai_error_response,
    build_openai_model_payload,
    build_openai_models_list_payload,
    parse_openai_chat_messages,
)


def register_openai_compat_routes(
    app,
    *,
    application_service: ApplicationService,
    list_model_entries: Callable[[], list[ModelInfoResponseModel]],
    load_model_info: Callable[[str], ModelInfoResponseModel],
) -> None:
    """Register the OpenAI-compatible routes on the FastAPI app."""

    @app.get(
        "/v1/models",
        response_model=OpenAIModelsListResponseModel,
        summary="List models (OpenAI-compatible)",
        tags=["openai-compatible"],
    )
    def openai_list_models() -> OpenAIModelsListResponseModel | object:
        try:
            return build_openai_models_list_payload(list_model_entries())
        except ValueError as exc:
            return build_openai_error_response(status_code=400, message=str(exc))

    @app.get(
        "/v1/models/{model_id:path}",
        response_model=OpenAIModelResponseModel,
        summary="Inspect one model (OpenAI-compatible)",
        tags=["openai-compatible"],
    )
    def openai_model_info(model_id: str) -> OpenAIModelResponseModel | object:
        try:
            return build_openai_model_payload(load_model_info(model_id))
        except ValueError as exc:
            return build_openai_error_response(status_code=400, message=str(exc))

    @app.post(
        "/v1/chat/completions",
        response_model=OpenAIChatCompletionResponseModel,
        summary="Create a chat completion (OpenAI-compatible)",
        tags=["openai-compatible"],
    )
    def chat_completions(
        request: OpenAIChatCompletionRequestModel,
    ) -> OpenAIChatCompletionResponseModel | object:
        try:
            runtime_config = _build_openai_runtime_config(request.model)
            generation_config = _build_openai_generation_config(request)
            history, prompt_parts = parse_openai_chat_messages(request.messages)
        except ValueError as exc:
            return build_openai_error_response(status_code=400, message=str(exc))

        if request.stream:
            return build_openai_chat_sse_response(
                lambda sink: application_service.prompt_parts(
                    prompt_parts,
                    runtime_config=runtime_config,
                    generation_config=generation_config,
                    system_prompt="",
                    history=history,
                    sink=sink,
                ),
                model=request.model,
            )

        try:
            response = application_service.prompt_parts(
                prompt_parts,
                runtime_config=runtime_config,
                generation_config=generation_config,
                system_prompt="",
                history=history,
            )
        except ValueError as exc:
            return build_openai_error_response(status_code=400, message=str(exc))

        return build_openai_chat_completion_response(
            model=request.model,
            text=response.text,
        )


def _build_openai_runtime_config(model_reference: str) -> RuntimeConfig:
    settings = load_app_settings()
    return resolve_runtime_config(
        settings.runtime,
        RuntimeConfigOverrides(model_reference=model_reference),
    )


def _build_openai_generation_config(
    request: OpenAIChatCompletionRequestModel,
) -> GenerationConfig:
    settings = load_app_settings()
    return resolve_generation_config(
        settings.generation,
        GenerationConfigOverrides(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
            stream=request.stream,
        ),
    )

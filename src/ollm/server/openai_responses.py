"""OpenAI-compatible Responses API route registration."""

import time
import uuid

from ollm.app.service import ApplicationService
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.settings import (
    GenerationConfigOverrides,
    RuntimeConfigOverrides,
    load_app_settings,
    resolve_generation_config,
    resolve_runtime_config,
)
from ollm.server.openai_compat import build_openai_error_response
from ollm.server.openai_response_execution import (
    build_conversation_items,
    build_response_payload,
    build_response_prompt,
    new_output_message_id,
    resolve_response_system_prompt,
)
from ollm.server.openai_response_models import (
    OpenAIDeletedResponseModel,
    OpenAIResponseCreateRequestModel,
)
from ollm.server.openai_response_store import OpenAIResponseStore, StoredOpenAIResponse
from ollm.server.openai_response_streaming import (
    response_event_iterator,
    structured_response_event_iterator,
)
from ollm.server.openai_response_tooling import (
    normalize_tool_choice,
    parse_model_output,
)
from ollm.server.streaming import build_sse_response_from_iterator


def register_openai_responses_routes(
    app,
    *,
    application_service: ApplicationService,
    response_store: OpenAIResponseStore,
) -> None:
    """Register the OpenAI-compatible Responses API routes."""

    @app.post(
        "/v1/responses",
        response_model=object,
        summary="Create a response (OpenAI-compatible)",
        tags=["openai-compatible"],
    )
    def create_response(request: OpenAIResponseCreateRequestModel) -> object:
        try:
            runtime_config = _build_runtime_config(request.model)
            generation_config = _build_generation_config(request)
            tool_choice = normalize_tool_choice(
                request.tool_choice,
                tools=request.tools,
            )
            prepared_request = build_response_prompt(
                request.input,
                previous_response_id=request.previous_response_id,
                response_store=response_store,
            )
        except ValueError as exc:
            return build_openai_error_response(status_code=400, message=str(exc))

        created_at = int(time.time())
        response_id = _new_response_id()
        output_message_id = new_output_message_id()
        system_prompt = resolve_response_system_prompt(
            instructions=request.instructions,
            tools=request.tools,
            tool_choice=tool_choice,
        )

        if request.stream:
            if request.tools:
                return build_sse_response_from_iterator(
                    structured_response_event_iterator(
                        lambda: application_service.prompt_parts(
                            prepared_request.prompt_parts,
                            runtime_config=runtime_config,
                            generation_config=generation_config,
                            system_prompt=system_prompt,
                            history=prepared_request.history_messages,
                        ),
                        model=request.model,
                        response_id=response_id,
                        created_at=created_at,
                        request_input=request.input,
                        instructions=request.instructions,
                        max_output_tokens=request.max_output_tokens,
                        previous_response_id=request.previous_response_id,
                        response_store=response_store,
                        conversation_items=prepared_request.conversation_items,
                        temperature=request.temperature,
                        tools=request.tools,
                        tool_choice=tool_choice,
                        top_p=request.top_p,
                        parallel_tool_calls=request.parallel_tool_calls,
                    )
                )
            return build_sse_response_from_iterator(
                response_event_iterator(
                    lambda sink: application_service.prompt_parts(
                        prepared_request.prompt_parts,
                        runtime_config=runtime_config,
                        generation_config=generation_config,
                        system_prompt=system_prompt,
                        history=prepared_request.history_messages,
                        sink=sink,
                    ),
                    model=request.model,
                    response_id=response_id,
                    output_message_id=output_message_id,
                    created_at=created_at,
                    request_input=request.input,
                    instructions=request.instructions,
                    max_output_tokens=request.max_output_tokens,
                    previous_response_id=request.previous_response_id,
                    response_store=response_store,
                    conversation_items=prepared_request.conversation_items,
                    temperature=request.temperature,
                    tools=request.tools,
                    tool_choice=tool_choice,
                    top_p=request.top_p,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
            )

        try:
            prompt_response = application_service.prompt_parts(
                prepared_request.prompt_parts,
                runtime_config=runtime_config,
                generation_config=generation_config,
                system_prompt=system_prompt,
                history=prepared_request.history_messages,
            )
            parsed_output = parse_model_output(
                prompt_response.text,
                tools=request.tools,
                tool_choice=tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
            )
        except ValueError as exc:
            status_code = 500 if request.tools else 400
            error_type = "server_error" if request.tools else "invalid_request_error"
            code = "responses_tool_contract_failed" if request.tools else None
            return build_openai_error_response(
                status_code=status_code,
                message=str(exc),
                error_type=error_type,
                code=code,
            )

        completed_at = int(time.time())
        response = build_response_payload(
            response_id=response_id,
            output_message_id=output_message_id,
            created_at=created_at,
            completed_at=completed_at,
            model=request.model,
            parsed_output=parsed_output,
            input_items=request.input,
            instructions=request.instructions,
            max_output_tokens=request.max_output_tokens,
            previous_response_id=request.previous_response_id,
            store=response_store.enabled,
            temperature=request.temperature,
            tools=request.tools,
            tool_choice=tool_choice,
            top_p=request.top_p,
            parallel_tool_calls=request.parallel_tool_calls,
        )
        response_store.put(
            StoredOpenAIResponse(
                response=response,
                conversation_items=build_conversation_items(
                    base_items=prepared_request.conversation_items,
                    output_items=response.output,
                ),
            )
        )
        return response

    @app.get(
        "/v1/responses/{response_id}",
        response_model=object,
        summary="Retrieve a response (OpenAI-compatible)",
        tags=["openai-compatible"],
    )
    def get_response(response_id: str) -> object:
        if not response_store.enabled:
            return build_openai_error_response(
                status_code=501,
                message="Responses retrieval requires a configured response-store backend",
                error_type="server_error",
                code="responses_storage_disabled",
            )
        try:
            return response_store.require(response_id).response
        except ValueError as exc:
            return build_openai_error_response(
                status_code=404,
                message=str(exc),
                error_type="not_found_error",
            )

    @app.delete(
        "/v1/responses/{response_id}",
        response_model=OpenAIDeletedResponseModel,
        summary="Delete a response (OpenAI-compatible)",
        tags=["openai-compatible"],
    )
    def delete_response(response_id: str) -> OpenAIDeletedResponseModel | object:
        if not response_store.enabled:
            return build_openai_error_response(
                status_code=501,
                message="Responses deletion requires a configured response-store backend",
                error_type="server_error",
                code="responses_storage_disabled",
            )
        deleted = response_store.delete(response_id)
        if not deleted:
            return build_openai_error_response(
                status_code=404,
                message=f"Response '{response_id}' does not exist",
                error_type="not_found_error",
            )
        return OpenAIDeletedResponseModel(id=response_id)


def _build_runtime_config(model_reference: str) -> RuntimeConfig:
    settings = load_app_settings()
    return resolve_runtime_config(
        settings.runtime,
        RuntimeConfigOverrides(model_reference=model_reference),
    )


def _build_generation_config(
    request: OpenAIResponseCreateRequestModel,
) -> GenerationConfig:
    settings = load_app_settings()
    return resolve_generation_config(
        settings.generation,
        GenerationConfigOverrides(
            max_new_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
            stream=request.stream,
        ),
    )


def _new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"

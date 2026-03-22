"""Pydantic transport models for the native and OpenAI-compatible server APIs."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ollm.runtime.config import DEFAULT_SYSTEM_PROMPT


class HealthResponseModel(BaseModel):
    """Health response for the local-only server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    service: str
    version: str
    server_mode: str


class RuntimeConfigResponseModel(BaseModel):
    """Serialized runtime configuration for native API responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_reference: str
    models_dir: str
    device: str
    backend: str | None
    adapter_dir: str | None
    multimodal: bool
    use_specialization: bool
    cache_dir: str
    use_cache: bool
    kv_cache_strategy: str
    strategy_selector_profile: str
    kv_cache_lifecycle: str
    kv_cache_adaptation_mode: str
    kv_cache_window_tokens: int | None
    dense_projection_chunk_rows: int | None
    offload_cpu_layers: int
    offload_cpu_policy: str
    offload_gpu_layers: int
    force_download: bool
    stats: bool
    verbose: bool
    quiet: bool


class ResolvedModelResponseModel(BaseModel):
    """Serialized resolved-model payload for native API responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_reference: str
    normalized_name: str
    source_kind: str
    support_level: str
    modalities: list[str]
    requires_processor: bool
    supports_disk_cache: bool
    supports_specialization: bool
    repo_id: str | None
    revision: str | None
    path: str | None
    native_family: str | None
    architecture: str | None
    model_type: str | None
    generic_model_kind: str | None
    resolution_message: str


class RuntimePlanResponseModel(BaseModel):
    """Serialized runtime-plan payload for native API responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend_id: str | None
    modalities: list[str]
    requires_processor: bool
    supports_disk_cache: bool
    supports_cpu_offload: bool
    supports_gpu_offload: bool
    specialization_provider_id: str | None
    specialization_enabled: bool
    specialization_state: str
    planned_specialization_pass_ids: list[str]
    reason: str
    details: dict[str, str]


class PlanResponseModel(BaseModel):
    """Structured runtime plan response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_config: RuntimeConfigResponseModel
    resolved_model: ResolvedModelResponseModel
    runtime_plan: RuntimePlanResponseModel


class ModelInfoResponseModel(BaseModel):
    """Serialized native model discovery/info payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_reference: str
    normalized_name: str
    source_kind: str
    support_level: str
    modalities: list[str]
    requires_processor: bool
    supports_disk_cache: bool
    supports_specialization: bool
    repo_id: str | None
    revision: str | None
    path: str | None
    native_family: str | None
    architecture: str | None
    model_type: str | None
    generic_model_kind: str | None
    resolution_message: str
    materialized: bool
    resolved_support_level: str
    resolved_modalities: list[str]
    resolved_requires_processor: bool
    resolved_supports_disk_cache: bool
    resolved_resolution_message: str
    runtime_plan: RuntimePlanResponseModel
    discovery_source: str | None = None


class ModelsListResponseModel(BaseModel):
    """Response model for the native models list endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    models: list[ModelInfoResponseModel]


class RuntimeRequestModel(BaseModel):
    """Request overrides for native runtime configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_reference: str | None = None
    models_dir: Path | None = None
    device: str | None = None
    backend: str | None = None
    adapter_dir: Path | None = None
    multimodal: bool | None = None
    use_specialization: bool | None = None
    cache_dir: Path | None = None
    use_cache: bool | None = None
    kv_cache_strategy: str | None = None
    strategy_selector_profile: str | None = None
    kv_cache_lifecycle: str | None = None
    kv_cache_adaptation_mode: str | None = None
    kv_cache_window_tokens: int | None = Field(default=None, gt=0)
    dense_projection_chunk_rows: int | None = Field(default=None, gt=0)
    offload_cpu_layers: int | None = Field(default=None, ge=0)
    offload_cpu_policy: str | None = None
    offload_gpu_layers: int | None = Field(default=None, ge=0)
    force_download: bool | None = None
    stats: bool | None = None
    verbose: bool | None = None
    quiet: bool | None = None


class GenerationRequestModel(BaseModel):
    """Request overrides for native generation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, gt=0)
    seed: int | None = None


class PlanRequestModel(BaseModel):
    """Request model for runtime plan inspection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: RuntimeRequestModel = Field(default_factory=RuntimeRequestModel)


class PromptRequestModel(BaseModel):
    """Request model for native prompt execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    runtime: RuntimeRequestModel = Field(default_factory=RuntimeRequestModel)
    generation: GenerationRequestModel = Field(default_factory=GenerationRequestModel)


class PromptResponseModel(BaseModel):
    """Response model for native prompt execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    metadata: dict[str, str]


class ContentPartResponseModel(BaseModel):
    """Serialized content part for native session responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    value: str


class MessageResponseModel(BaseModel):
    """Serialized message payload for native session responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    content: list[ContentPartResponseModel]


class SessionCreateRequestModel(BaseModel):
    """Request model for server-side session creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_name: str = "default"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    runtime: RuntimeRequestModel = Field(default_factory=RuntimeRequestModel)
    generation: GenerationRequestModel = Field(default_factory=GenerationRequestModel)


class SessionResponseModel(BaseModel):
    """Serialized server-side session state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    session_name: str
    model_reference: str
    system_prompt: str
    messages: list[MessageResponseModel]


class SessionPromptRequestModel(BaseModel):
    """Request model for session prompt execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str


class OpenAIModelResponseModel(BaseModel):
    """OpenAI-compatible model object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    object: str = "model"
    created: int
    owned_by: str


class OpenAIModelsListResponseModel(BaseModel):
    """OpenAI-compatible model list response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object: str = "list"
    data: list[OpenAIModelResponseModel]


class OpenAIChatMessageContentPartRequestModel(BaseModel):
    """Structured content part for chat-completions messages."""

    model_config = ConfigDict(extra="allow", frozen=True)

    type: str
    text: str | None = None


class OpenAIChatMessageRequestModel(BaseModel):
    """OpenAI-compatible request message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    content: str | list[OpenAIChatMessageContentPartRequestModel]
    name: str | None = None


class OpenAIChatCompletionRequestModel(BaseModel):
    """OpenAI-compatible chat-completions request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    messages: list[OpenAIChatMessageRequestModel]
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None


class OpenAIChatCompletionMessageResponseModel(BaseModel):
    """OpenAI-compatible response message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = "assistant"
    content: str


class OpenAIChatCompletionChoiceResponseModel(BaseModel):
    """OpenAI-compatible non-streaming choice payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    message: OpenAIChatCompletionMessageResponseModel
    finish_reason: str | None = None


class OpenAIUsageResponseModel(BaseModel):
    """OpenAI-compatible usage payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatCompletionResponseModel(BaseModel):
    """OpenAI-compatible non-streaming chat completion response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatCompletionChoiceResponseModel]
    usage: OpenAIUsageResponseModel | None = None


class OpenAIChatCompletionDeltaResponseModel(BaseModel):
    """OpenAI-compatible streaming delta payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str | None = None
    content: str | None = None


class OpenAIChatCompletionChunkChoiceResponseModel(BaseModel):
    """OpenAI-compatible streaming choice payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    delta: OpenAIChatCompletionDeltaResponseModel
    finish_reason: str | None = None


class OpenAIChatCompletionChunkResponseModel(BaseModel):
    """OpenAI-compatible streaming chat completion chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIChatCompletionChunkChoiceResponseModel]


class OpenAIErrorResponseModel(BaseModel):
    """OpenAI-compatible error object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorEnvelopeResponseModel(BaseModel):
    """OpenAI-compatible top-level error envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    error: OpenAIErrorResponseModel

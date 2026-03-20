"""Pydantic transport models for the oLLM REST server surface."""

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
    """Serialized runtime configuration for API responses."""

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
    kv_cache_lifecycle: str
    kv_cache_adaptation_mode: str
    kv_cache_window_tokens: int | None
    offload_cpu_layers: int
    offload_gpu_layers: int
    force_download: bool
    stats: bool
    verbose: bool
    quiet: bool


class ResolvedModelResponseModel(BaseModel):
    """Serialized resolved-model payload for API responses."""

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
    """Serialized runtime-plan payload for API responses."""

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


class PlanResponseModel(BaseModel):
    """Structured runtime plan response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_config: RuntimeConfigResponseModel
    resolved_model: ResolvedModelResponseModel
    runtime_plan: RuntimePlanResponseModel


class ModelInfoResponseModel(BaseModel):
    """Serialized model discovery/info payload for API responses."""

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
    """Response model for the models list endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    models: list[ModelInfoResponseModel]


class RuntimeRequestModel(BaseModel):
    """Request overrides for runtime configuration."""

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
    kv_cache_lifecycle: str | None = None
    kv_cache_adaptation_mode: str | None = None
    kv_cache_window_tokens: int | None = Field(default=None, gt=0)
    offload_cpu_layers: int | None = Field(default=None, ge=0)
    offload_gpu_layers: int | None = Field(default=None, ge=0)
    force_download: bool | None = None
    stats: bool | None = None
    verbose: bool | None = None
    quiet: bool | None = None


class GenerationRequestModel(BaseModel):
    """Request overrides for generation configuration."""

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
    """Request model for prompt execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    runtime: RuntimeRequestModel = Field(default_factory=RuntimeRequestModel)
    generation: GenerationRequestModel = Field(default_factory=GenerationRequestModel)


class PromptResponseModel(BaseModel):
    """Response model for prompt execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    metadata: dict[str, str]


class ContentPartResponseModel(BaseModel):
    """Serialized content part for session responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    value: str


class MessageResponseModel(BaseModel):
    """Serialized message payload for session responses."""

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

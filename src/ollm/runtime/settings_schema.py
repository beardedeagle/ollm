"""Settings schema models and validation rules for oLLM."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ollm.dense_projection_chunking import normalize_dense_projection_chunk_rows
from ollm.kv_cache.matrix import (
    DEFAULT_KV_CACHE_ADAPTATION_MODE,
    DEFAULT_KV_CACHE_LIFECYCLE,
    normalize_kv_cache_adaptation_mode,
    normalize_kv_cache_window_tokens,
    resolve_kv_cache_lifecycle,
    resolve_kv_cache_window_tokens,
)
from ollm.kv_cache.strategy import (
    DEFAULT_KV_CACHE_STRATEGY,
    normalize_kv_cache_strategy,
)
from ollm.runtime.config import (
    DEFAULT_DEVICE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_REFERENCE,
    _window_strategy_for_validation,
    normalize_backend,
)
from ollm.runtime.offload_policy import (
    DEFAULT_CPU_OFFLOAD_POLICY,
    normalize_cpu_offload_policy,
)
from ollm.runtime.server_settings_support import (
    DEFAULT_RESPONSE_STORE_BACKEND,
    normalize_response_store_backend,
)
from ollm.runtime.strategy_selector import (
    DEFAULT_STRATEGY_SELECTOR_PROFILE,
    normalize_strategy_selector_profile,
)

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000
DEFAULT_SETTINGS_FILE = Path("ollm.toml")


class SettingsPrecedenceLayer(StrEnum):
    """Named layers for the canonical settings-precedence contract."""

    CLI = "cli"
    ENVIRONMENT = "env"
    CONFIG_FILE = "config_file"
    DEFAULTS = "defaults"


SETTINGS_PRECEDENCE = (
    SettingsPrecedenceLayer.CLI,
    SettingsPrecedenceLayer.ENVIRONMENT,
    SettingsPrecedenceLayer.CONFIG_FILE,
    SettingsPrecedenceLayer.DEFAULTS,
)


class RuntimeSettings(BaseModel):
    """Default runtime settings resolved before execution-specific overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_reference: str = DEFAULT_MODEL_REFERENCE
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    device: str = DEFAULT_DEVICE
    backend: str | None = None
    adapter_dir: Path | None = None
    multimodal: bool = False
    use_specialization: bool = True
    cache_dir: Path = Field(default_factory=lambda: Path("kv_cache"))
    use_cache: bool = True
    kv_cache_strategy: str | None = None
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE
    kv_cache_lifecycle: str = DEFAULT_KV_CACHE_LIFECYCLE
    kv_cache_adaptation_mode: str = DEFAULT_KV_CACHE_ADAPTATION_MODE
    kv_cache_window_tokens: int | None = Field(default=None, gt=0)
    dense_projection_chunk_rows: int | None = Field(default=None, gt=0)
    offload_cpu_layers: int = Field(default=0, ge=0)
    offload_cpu_policy: str = DEFAULT_CPU_OFFLOAD_POLICY
    offload_gpu_layers: int = Field(default=0, ge=0)
    force_download: bool = False
    stats: bool = False
    verbose: bool = False
    quiet: bool = False

    @field_validator("backend")
    @classmethod
    def _normalize_backend(cls, backend: str | None) -> str | None:
        return normalize_backend(backend)

    @field_validator("kv_cache_strategy")
    @classmethod
    def _normalize_kv_cache_strategy(cls, strategy: str | None) -> str | None:
        return normalize_kv_cache_strategy(strategy)

    @field_validator("strategy_selector_profile")
    @classmethod
    def _normalize_strategy_selector_profile(cls, profile: str) -> str:
        normalized_profile = normalize_strategy_selector_profile(profile)
        if normalized_profile is None:
            raise ValueError("strategy_selector_profile cannot be empty")
        return normalized_profile

    @field_validator("kv_cache_lifecycle")
    @classmethod
    def _normalize_kv_cache_lifecycle(cls, lifecycle: str) -> str:
        return resolve_kv_cache_lifecycle(DEFAULT_KV_CACHE_STRATEGY, lifecycle)

    @field_validator("kv_cache_adaptation_mode")
    @classmethod
    def _normalize_kv_cache_adaptation_mode(cls, mode: str) -> str:
        normalized_mode = normalize_kv_cache_adaptation_mode(mode)
        if normalized_mode is None:
            raise ValueError("kv_cache_adaptation_mode cannot be empty")
        return normalized_mode

    @field_validator("kv_cache_window_tokens")
    @classmethod
    def _normalize_kv_cache_window_tokens(cls, window_tokens: int | None) -> int | None:
        return normalize_kv_cache_window_tokens(window_tokens)

    @field_validator("dense_projection_chunk_rows")
    @classmethod
    def _normalize_dense_projection_chunk_rows(cls, rows: int | None) -> int | None:
        return normalize_dense_projection_chunk_rows(rows)

    @field_validator("offload_cpu_policy")
    @classmethod
    def _normalize_offload_cpu_policy(cls, policy: str) -> str:
        normalized_policy = normalize_cpu_offload_policy(policy)
        if normalized_policy is None:
            raise ValueError("offload_cpu_policy cannot be empty")
        return normalized_policy

    @model_validator(mode="after")
    def _validate_window_strategy_pair(self):
        resolve_kv_cache_lifecycle(
            self.kv_cache_strategy,
            self.kv_cache_lifecycle,
        )
        resolve_kv_cache_window_tokens(
            _window_strategy_for_validation(
                self.kv_cache_strategy,
                self.strategy_selector_profile,
                self.kv_cache_window_tokens,
            ),
            self.kv_cache_window_tokens,
        )
        if self.offload_cpu_layers > 0 and self.device == "cpu":
            raise ValueError(
                "--offload-cpu-layers requires an accelerator runtime device"
            )
        if self.offload_cpu_layers > 0 and self.offload_gpu_layers > 0:
            raise ValueError(
                "--offload-cpu-layers cannot be combined with "
                "--offload-gpu-layers in this runtime"
            )
        return self


class GenerationSettings(BaseModel):
    """Default generation settings resolved before request-level overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stream: bool = True


class ServerSettings(BaseModel):
    """Default headless-server settings for future server surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = DEFAULT_SERVER_HOST
    port: int = Field(default=DEFAULT_SERVER_PORT, gt=0, le=65535)
    reload: bool = False
    log_level: str = "info"
    response_store_backend: str = DEFAULT_RESPONSE_STORE_BACKEND
    response_store_factory: str | None = None

    @field_validator("response_store_backend")
    @classmethod
    def _normalize_response_store_backend(cls, backend: str) -> str:
        normalized_backend = normalize_response_store_backend(backend)
        if normalized_backend is None:
            raise ValueError("response_store_backend cannot be empty")
        return normalized_backend

    @field_validator("response_store_factory")
    @classmethod
    def _normalize_response_store_factory(cls, factory: str | None) -> str | None:
        if factory is None:
            return None
        normalized_factory = factory.strip()
        if not normalized_factory:
            raise ValueError("response_store_factory cannot be empty")
        return normalized_factory

    @model_validator(mode="after")
    def _validate_response_store_pair(self):
        if (
            self.response_store_backend == "plugin"
            and self.response_store_factory is None
        ):
            raise ValueError(
                "response_store_factory is required when response_store_backend=plugin"
            )
        if self.response_store_backend != "plugin" and self.response_store_factory:
            raise ValueError(
                "response_store_factory is only valid when response_store_backend=plugin"
            )
        return self


class AppSettings(BaseSettings):
    """Top-level application settings schema for CLI and future server surfaces."""

    model_config = SettingsConfigDict(
        env_prefix="OLLM_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)


class RuntimeConfigOverrides(BaseModel):
    """Explicit runtime overrides layered on top of runtime settings."""

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

    @field_validator("backend")
    @classmethod
    def _normalize_backend(cls, backend: str | None) -> str | None:
        return normalize_backend(backend)

    @field_validator("kv_cache_strategy")
    @classmethod
    def _normalize_kv_cache_strategy(cls, strategy: str | None) -> str | None:
        return normalize_kv_cache_strategy(strategy)

    @field_validator("strategy_selector_profile")
    @classmethod
    def _normalize_strategy_selector_profile(cls, profile: str | None) -> str | None:
        return normalize_strategy_selector_profile(profile)

    @field_validator("kv_cache_lifecycle")
    @classmethod
    def _normalize_kv_cache_lifecycle(cls, lifecycle: str | None) -> str | None:
        if lifecycle is None:
            return None
        return resolve_kv_cache_lifecycle(DEFAULT_KV_CACHE_STRATEGY, lifecycle)

    @field_validator("kv_cache_adaptation_mode")
    @classmethod
    def _normalize_kv_cache_adaptation_mode(cls, mode: str | None) -> str | None:
        return normalize_kv_cache_adaptation_mode(mode)

    @field_validator("kv_cache_window_tokens")
    @classmethod
    def _normalize_kv_cache_window_tokens(cls, window_tokens: int | None) -> int | None:
        return normalize_kv_cache_window_tokens(window_tokens)

    @field_validator("dense_projection_chunk_rows")
    @classmethod
    def _normalize_dense_projection_chunk_rows(cls, rows: int | None) -> int | None:
        return normalize_dense_projection_chunk_rows(rows)

    @field_validator("offload_cpu_policy")
    @classmethod
    def _normalize_offload_cpu_policy(cls, policy: str | None) -> str | None:
        return normalize_cpu_offload_policy(policy)

    @model_validator(mode="after")
    def _validate_window_strategy_pair(self):
        strategy = self.kv_cache_strategy
        resolve_kv_cache_lifecycle(
            DEFAULT_KV_CACHE_STRATEGY if strategy is None else strategy,
            self.kv_cache_lifecycle,
        )
        resolve_kv_cache_window_tokens(
            _window_strategy_for_validation(
                strategy,
                self.strategy_selector_profile,
                self.kv_cache_window_tokens,
            ),
            self.kv_cache_window_tokens,
        )
        offload_cpu_layers = (
            0 if self.offload_cpu_layers is None else self.offload_cpu_layers
        )
        offload_gpu_layers = (
            0 if self.offload_gpu_layers is None else self.offload_gpu_layers
        )
        resolved_device = DEFAULT_DEVICE if self.device is None else self.device
        if offload_cpu_layers > 0 and resolved_device == "cpu":
            raise ValueError(
                "--offload-cpu-layers requires an accelerator runtime device"
            )
        if offload_cpu_layers > 0 and offload_gpu_layers > 0:
            raise ValueError(
                "--offload-cpu-layers cannot be combined with "
                "--offload-gpu-layers in this runtime"
            )
        return self


class GenerationConfigOverrides(BaseModel):
    """Explicit generation overrides layered on top of generation settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stream: bool | None = None


class ServerSettingsOverrides(BaseModel):
    """Explicit server overrides layered on top of server settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str | None = None
    port: int | None = Field(default=None, gt=0, le=65535)
    reload: bool | None = None
    log_level: str | None = None
    response_store_backend: str | None = None
    response_store_factory: str | None = None

    @field_validator("response_store_backend")
    @classmethod
    def _normalize_response_store_backend(cls, backend: str | None) -> str | None:
        return normalize_response_store_backend(backend)

    @field_validator("response_store_factory")
    @classmethod
    def _normalize_response_store_factory(cls, factory: str | None) -> str | None:
        if factory is None:
            return None
        normalized_factory = factory.strip()
        if not normalized_factory:
            raise ValueError("response_store_factory cannot be empty")
        return normalized_factory

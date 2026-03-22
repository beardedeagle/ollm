"""Runtime and generation configuration types used by the CLI and library APIs."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ollm.dense_projection_chunking import normalize_dense_projection_chunk_rows
from ollm.kv_cache.matrix import (
    DEFAULT_KV_CACHE_ADAPTATION_MODE,
    DEFAULT_KV_CACHE_LIFECYCLE,
    normalize_kv_cache_adaptation_mode,
    resolve_kv_cache_lifecycle,
    resolve_kv_cache_window_tokens,
)
from ollm.kv_cache.strategy import (
    DEFAULT_KV_CACHE_STRATEGY,
    normalize_kv_cache_strategy,
)
from ollm.runtime.offload_policy import (
    DEFAULT_CPU_OFFLOAD_POLICY,
    normalize_cpu_offload_policy,
    resolve_cpu_offload_policy,
)
from ollm.runtime.strategy_selector import (
    DEFAULT_STRATEGY_SELECTOR_PROFILE,
    StrategySelectorProfile,
    normalize_strategy_selector_profile,
    resolve_strategy_selector_profile,
)

DEFAULT_MODEL_REFERENCE = "llama3-1B-chat"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_NEW_TOKENS = 500


def _platform_default_device() -> str:
    """Return a sensible default device based on the host platform."""
    if sys.platform == "darwin":
        return "mps"
    if sys.platform in ("linux", "win32"):
        return "cuda:0"
    return "cpu"


DEFAULT_DEVICE = _platform_default_device()
KNOWN_BACKEND_IDS = ("optimized-native", "transformers-generic")


def normalize_backend(backend: str | None) -> str | None:
    """Validate and normalize a backend override identifier."""
    if backend is None:
        return None
    normalized_backend = backend.strip().lower()
    if not normalized_backend:
        raise ValueError("--backend cannot be empty")
    if normalized_backend not in KNOWN_BACKEND_IDS:
        allowed_backends = ", ".join(KNOWN_BACKEND_IDS)
        raise ValueError(f"--backend must be one of: {allowed_backends}")
    return normalized_backend


def _window_strategy_for_validation(
    strategy: str | None,
    strategy_selector_profile: str | None,
    window_tokens: int | None,
) -> str | None:
    resolved_profile = resolve_strategy_selector_profile(strategy_selector_profile)
    if strategy is not None:
        return strategy
    if (
        window_tokens is not None
        or resolved_profile == StrategySelectorProfile.BOUNDED_WINDOW.value
    ):
        return "sliding-window-ring-buffer"
    return strategy


@dataclass(slots=True)
class RuntimeConfig:
    """Describe how a model reference should be resolved and executed.

    This is the shared execution contract used by the CLI, the library, and the
    local server. Field annotations remain the source of truth for supported
    options, while the helper methods normalize and validate those fields for
    planning and execution.
    """

    model_reference: str = DEFAULT_MODEL_REFERENCE
    models_dir: Path = field(default_factory=lambda: Path("models"))
    device: str = DEFAULT_DEVICE
    backend: str | None = None
    adapter_dir: Path | None = None
    multimodal: bool = False
    use_specialization: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("kv_cache"))
    use_cache: bool = True
    kv_cache_strategy: str | None = None
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE
    kv_cache_lifecycle: str = DEFAULT_KV_CACHE_LIFECYCLE
    kv_cache_adaptation_mode: str = DEFAULT_KV_CACHE_ADAPTATION_MODE
    kv_cache_window_tokens: int | None = None
    dense_projection_chunk_rows: int | None = None
    offload_cpu_layers: int = 0
    offload_cpu_policy: str = DEFAULT_CPU_OFFLOAD_POLICY
    offload_gpu_layers: int = 0
    force_download: bool = False
    stats: bool = False
    verbose: bool = False
    quiet: bool = False

    def resolved_models_dir(self) -> Path:
        """Return the absolute local models directory."""
        return self.models_dir.expanduser().resolve()

    def resolved_backend(self) -> str | None:
        """Return the normalized backend override when provided."""
        return normalize_backend(self.backend)

    def resolved_cache_dir(self) -> Path:
        """Return the absolute cache directory."""
        return self.cache_dir.expanduser().resolve()

    def requested_kv_cache_strategy(self) -> str | None:
        """Return the normalized explicit KV strategy override when one exists."""

        return normalize_kv_cache_strategy(self.kv_cache_strategy)

    def resolved_strategy_selector_profile(self) -> str:
        """Return the normalized selector profile."""

        return resolve_strategy_selector_profile(self.strategy_selector_profile)

    def resolved_kv_cache_strategy(self) -> str:
        """Return the normalized KV cache strategy."""
        normalized_strategy = self.requested_kv_cache_strategy()
        if normalized_strategy is None:
            return DEFAULT_KV_CACHE_STRATEGY
        return normalized_strategy

    def resolved_kv_cache_lifecycle(self) -> str:
        """Return the normalized cache lifecycle."""
        return resolve_kv_cache_lifecycle(
            self.kv_cache_strategy,
            self.kv_cache_lifecycle,
        )

    def resolved_kv_cache_adaptation_mode(self) -> str:
        """Return the normalized cache adaptation mode."""

        normalized_mode = normalize_kv_cache_adaptation_mode(
            self.kv_cache_adaptation_mode
        )
        if normalized_mode is None:
            return DEFAULT_KV_CACHE_ADAPTATION_MODE
        return normalized_mode

    def resolved_kv_cache_window_tokens(self) -> int | None:
        """Return the normalized sliding-window token budget."""

        return resolve_kv_cache_window_tokens(
            _window_strategy_for_validation(
                self.kv_cache_strategy,
                self.strategy_selector_profile,
                self.kv_cache_window_tokens,
            ),
            self.kv_cache_window_tokens,
        )

    def resolved_dense_projection_chunk_rows(self) -> int | None:
        """Return the normalized explicit dense-projection chunk row budget."""

        return normalize_dense_projection_chunk_rows(self.dense_projection_chunk_rows)

    def resolved_offload_cpu_policy(self) -> str:
        """Return the normalized CPU offload policy."""

        return resolve_cpu_offload_policy(self.offload_cpu_policy)

    def resolved_adapter_dir(self) -> Path | None:
        """Return the absolute adapter directory when one is configured."""
        if self.adapter_dir is None:
            return None
        return self.adapter_dir.expanduser().resolve()

    def validate(self) -> None:
        """Validate the configuration before planning or execution.

        Raises:
            ValueError: Raised when any runtime option is structurally invalid,
                contradictory, or unsupported for the current execution model.
        """
        if not self.model_reference.strip():
            raise ValueError("--model cannot be empty")
        if self.backend is not None:
            normalize_backend(self.backend)
        normalize_kv_cache_strategy(self.kv_cache_strategy)
        normalize_strategy_selector_profile(self.strategy_selector_profile)
        resolve_kv_cache_lifecycle(
            self.kv_cache_strategy,
            self.kv_cache_lifecycle,
        )
        normalize_kv_cache_adaptation_mode(self.kv_cache_adaptation_mode)
        resolve_kv_cache_window_tokens(
            _window_strategy_for_validation(
                self.kv_cache_strategy,
                self.strategy_selector_profile,
                self.kv_cache_window_tokens,
            ),
            self.kv_cache_window_tokens,
        )
        normalize_dense_projection_chunk_rows(self.dense_projection_chunk_rows)
        normalize_cpu_offload_policy(self.offload_cpu_policy)
        if self.verbose and self.quiet:
            raise ValueError("--verbose and --quiet cannot be used together")
        if (
            not self.use_specialization
            and self.resolved_backend() == "optimized-native"
        ):
            raise ValueError(
                "--backend optimized-native cannot be combined with --no-specialization"
            )
        if self.offload_cpu_layers < 0:
            raise ValueError("--offload-cpu-layers must be zero or greater")
        if self.offload_cpu_layers > 0 and self.device == "cpu":
            raise ValueError(
                "--offload-cpu-layers requires an accelerator runtime device"
            )
        if self.offload_cpu_layers > 0 and self.offload_gpu_layers > 0:
            raise ValueError(
                "--offload-cpu-layers cannot be combined with "
                "--offload-gpu-layers in this runtime"
            )
        if self.offload_gpu_layers < 0:
            raise ValueError("--offload-gpu-layers must be zero or greater")


@dataclass(slots=True)
class GenerationConfig:
    """Describe generation-time sampling and streaming behavior.

    Field annotations remain the source of truth for supported sampling controls.
    Use :meth:`validate` before execution when constructing this type directly.
    """

    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stream: bool = True

    def validate(self) -> None:
        """Validate sampling and generation limits.

        Raises:
            ValueError: Raised when token or sampling limits fall outside the
                supported runtime range.
        """
        if self.max_new_tokens <= 0:
            raise ValueError("--max-new-tokens must be greater than zero")
        if self.temperature < 0:
            raise ValueError("--temperature must be zero or greater")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("--top-p must be within (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("--top-k must be greater than zero")

    def sampling_enabled(self) -> bool:
        """Return whether stochastic sampling is enabled.

        Returns:
            bool: ``True`` when ``temperature`` enables stochastic sampling.
        """
        return self.temperature > 0

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_MODEL_REFERENCE = "llama3-1B-chat"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_NEW_TOKENS = 500
KNOWN_BACKEND_IDS = (
    "optimized-native",
    "transformers-generic",
    "ollama",
    "openai-compatible",
)


def normalize_provider_endpoint(provider_endpoint: str | None) -> str | None:
    if provider_endpoint is None:
        return None
    normalized_endpoint = provider_endpoint.strip().rstrip("/")
    if not normalized_endpoint:
        raise ValueError("--provider-endpoint cannot be empty")
    parsed_endpoint = urlparse(normalized_endpoint)
    if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
        raise ValueError("--provider-endpoint must be an absolute http or https URL")
    if parsed_endpoint.username is not None or parsed_endpoint.password is not None:
        raise ValueError("--provider-endpoint must not include credentials")
    return normalized_endpoint


def normalize_backend(backend: str | None) -> str | None:
    if backend is None:
        return None
    normalized_backend = backend.strip().lower()
    if not normalized_backend:
        raise ValueError("--backend cannot be empty")
    if normalized_backend not in KNOWN_BACKEND_IDS:
        allowed_backends = ", ".join(KNOWN_BACKEND_IDS)
        raise ValueError(f"--backend must be one of: {allowed_backends}")
    return normalized_backend


@dataclass(slots=True)
class RuntimeConfig:
    model_reference: str = DEFAULT_MODEL_REFERENCE
    models_dir: Path = field(default_factory=lambda: Path("models"))
    device: str = "cuda:0"
    backend: str | None = None
    provider_endpoint: str | None = None
    adapter_dir: Path | None = None
    multimodal: bool = False
    use_specialization: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("kv_cache"))
    use_cache: bool = True
    offload_cpu_layers: int = 0
    offload_gpu_layers: int = 0
    force_download: bool = False
    stats: bool = False
    verbose: bool = False
    quiet: bool = False

    def resolved_models_dir(self) -> Path:
        return self.models_dir.expanduser().resolve()

    def resolved_backend(self) -> str | None:
        return normalize_backend(self.backend)

    def resolved_provider_endpoint(self) -> str | None:
        return normalize_provider_endpoint(self.provider_endpoint)

    def resolved_cache_dir(self) -> Path:
        return self.cache_dir.expanduser().resolve()

    def resolved_adapter_dir(self) -> Path | None:
        if self.adapter_dir is None:
            return None
        return self.adapter_dir.expanduser().resolve()

    def validate(self) -> None:
        if not self.model_reference.strip():
            raise ValueError("--model cannot be empty")
        if self.backend is not None:
            normalize_backend(self.backend)
        if self.provider_endpoint is not None:
            normalize_provider_endpoint(self.provider_endpoint)
        if self.verbose and self.quiet:
            raise ValueError("--verbose and --quiet cannot be used together")
        if not self.use_specialization and self.resolved_backend() == "optimized-native":
            raise ValueError("--backend optimized-native cannot be combined with --no-specialization")
        if self.offload_cpu_layers < 0:
            raise ValueError("--offload-cpu-layers must be zero or greater")
        if self.offload_gpu_layers < 0:
            raise ValueError("--offload-gpu-layers must be zero or greater")


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stream: bool = True

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("--max-new-tokens must be greater than zero")
        if self.temperature < 0:
            raise ValueError("--temperature must be zero or greater")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("--top-p must be within (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("--top-k must be greater than zero")

    def sampling_enabled(self) -> bool:
        return self.temperature > 0

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_REFERENCE = "llama3-1B-chat"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_NEW_TOKENS = 500


@dataclass(slots=True)
class RuntimeConfig:
    model_reference: str = DEFAULT_MODEL_REFERENCE
    models_dir: Path = field(default_factory=lambda: Path("models"))
    device: str = "cuda:0"
    adapter_dir: Path | None = None
    multimodal: bool = False
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

    def resolved_cache_dir(self) -> Path:
        return self.cache_dir.expanduser().resolve()

    def resolved_adapter_dir(self) -> Path | None:
        if self.adapter_dir is None:
            return None
        return self.adapter_dir.expanduser().resolve()

    def validate(self) -> None:
        if not self.model_reference.strip():
            raise ValueError("--model cannot be empty")
        if self.verbose and self.quiet:
            raise ValueError("--verbose and --quiet cannot be used together")
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

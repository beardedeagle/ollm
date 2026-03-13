import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from ollm.runtime.config import GenerationConfig, RuntimeConfig


def build_console(no_color: bool = False) -> Console:
    return Console(no_color=no_color)


def build_runtime_config(
    model: str,
    models_dir: Path,
    device: str,
    adapter_dir: Path | None,
    multimodal: bool,
    cache_dir: Path,
    no_cache: bool,
    offload_cpu_layers: int,
    offload_gpu_layers: int,
    force_download: bool,
    stats: bool,
    verbose: bool,
    quiet: bool,
) -> RuntimeConfig:
    config = RuntimeConfig(
        model_id=model,
        models_dir=models_dir,
        device=device,
        adapter_dir=adapter_dir,
        multimodal=multimodal,
        cache_dir=cache_dir,
        use_cache=not no_cache,
        offload_cpu_layers=offload_cpu_layers,
        offload_gpu_layers=offload_gpu_layers,
        force_download=force_download,
        stats=stats,
        verbose=verbose,
        quiet=quiet,
    )
    config.validate()
    return config


def build_generation_config(
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    stream: bool,
) -> GenerationConfig:
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        stream=stream,
    )
    config.validate()
    return config


def ensure_interactive_terminal() -> None:
    if sys.stdin.isatty() and sys.stdout.isatty():
        return
    raise typer.BadParameter("Interactive chat requires a TTY. Use `ollm prompt` for non-interactive usage.")


def print_json(console: Console, payload: dict[str, object]) -> None:
    del console
    typer.echo(json.dumps(payload, indent=2))


def config_as_dict(runtime_config: RuntimeConfig, generation_config: GenerationConfig) -> dict[str, object]:
    return {
        "runtime": {
            "model_id": runtime_config.model_id,
            "models_dir": str(runtime_config.resolved_models_dir()),
            "device": runtime_config.device,
            "adapter_dir": None if runtime_config.resolved_adapter_dir() is None else str(runtime_config.resolved_adapter_dir()),
            "multimodal": runtime_config.multimodal,
            "cache_dir": str(runtime_config.resolved_cache_dir()),
            "use_cache": runtime_config.use_cache,
            "offload_cpu_layers": runtime_config.offload_cpu_layers,
            "offload_gpu_layers": runtime_config.offload_gpu_layers,
            "force_download": runtime_config.force_download,
            "stats": runtime_config.stats,
            "verbose": runtime_config.verbose,
            "quiet": runtime_config.quiet,
        },
        "generation": {
            "max_new_tokens": generation_config.max_new_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "top_k": generation_config.top_k,
            "seed": generation_config.seed,
            "stream": generation_config.stream,
        },
    }

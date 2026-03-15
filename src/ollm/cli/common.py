import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.inspection import runtime_config_payload


def build_console(no_color: bool = False) -> Console:
    return Console(no_color=no_color)


def build_runtime_config(
    model: str,
    models_dir: Path,
    device: str,
    backend: str | None,
    provider_endpoint: str | None,
    adapter_dir: Path | None,
    multimodal: bool,
    no_specialization: bool,
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
        model_reference=model,
        models_dir=models_dir,
        device=device,
        backend=backend,
        provider_endpoint=provider_endpoint,
        adapter_dir=adapter_dir,
        multimodal=multimodal,
        use_specialization=not no_specialization,
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


def print_json(console: Console, payload: object) -> None:
    del console
    typer.echo(json.dumps(payload, indent=2))


def config_as_dict(runtime_config: RuntimeConfig, generation_config: GenerationConfig) -> dict[str, object]:
    return {
        "runtime": runtime_config_payload(runtime_config),
        "generation": {
            "max_new_tokens": generation_config.max_new_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "top_k": generation_config.top_k,
            "seed": generation_config.seed,
            "stream": generation_config.stream,
        },
    }

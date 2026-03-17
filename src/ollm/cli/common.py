import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.inspection import runtime_config_payload
from ollm.runtime.settings import (
    AppSettings,
    GenerationConfigOverrides,
    RuntimeConfigOverrides,
    load_app_settings,
    resolve_generation_config,
    resolve_runtime_config,
)


def build_console(no_color: bool = False) -> Console:
    return Console(no_color=no_color)


def build_runtime_config(
    model: str | None,
    models_dir: Path | None,
    device: str | None,
    backend: str | None,
    adapter_dir: Path | None,
    multimodal: bool | None,
    no_specialization: bool | None,
    cache_dir: Path | None,
    no_cache: bool | None,
    offload_cpu_layers: int | None,
    offload_gpu_layers: int | None,
    force_download: bool | None,
    stats: bool | None,
    verbose: bool | None,
    quiet: bool | None,
    settings: AppSettings | None = None,
) -> RuntimeConfig:
    resolved_settings = load_app_settings() if settings is None else settings
    return resolve_runtime_config(
        resolved_settings.runtime,
        RuntimeConfigOverrides(
            model_reference=model,
            models_dir=models_dir,
            device=device,
            backend=backend,
            adapter_dir=adapter_dir,
            multimodal=multimodal,
            use_specialization=(
                None if no_specialization is None else not no_specialization
            ),
            cache_dir=cache_dir,
            use_cache=None if no_cache is None else not no_cache,
            offload_cpu_layers=offload_cpu_layers,
            offload_gpu_layers=offload_gpu_layers,
            force_download=force_download,
            stats=stats,
            verbose=verbose,
            quiet=quiet,
        ),
    )


def build_generation_config(
    max_new_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    stream: bool | None,
    settings: AppSettings | None = None,
) -> GenerationConfig:
    resolved_settings = load_app_settings() if settings is None else settings
    return resolve_generation_config(
        resolved_settings.generation,
        GenerationConfigOverrides(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stream=stream,
        ),
    )


def ensure_interactive_terminal() -> None:
    if sys.stdin.isatty() and sys.stdout.isatty():
        return
    raise typer.BadParameter(
        "Interactive chat requires a TTY. Use `ollm prompt` for non-interactive usage."
    )


def print_json(console: Console, payload: object) -> None:
    del console
    typer.echo(json.dumps(payload, indent=2))


def config_as_dict(
    runtime_config: RuntimeConfig, generation_config: GenerationConfig
) -> dict[str, object]:
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

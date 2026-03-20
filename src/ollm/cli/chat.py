from pathlib import Path

import typer

from ollm.cli.common import (
    build_console,
    build_generation_config,
    build_runtime_config,
    config_as_dict,
    ensure_interactive_terminal,
    print_json,
)
from ollm.cli.services import CommandServices
from ollm.runtime.settings import load_app_settings
from ollm.ui.chat_shell import InteractiveChatShell


def run_chat_command(
    services: CommandServices,
    model: str | None,
    models_dir: Path | None,
    device: str | None,
    backend: str | None,
    adapter_dir: Path | None,
    multimodal: bool | None,
    no_specialization: bool | None,
    cache_dir: Path | None,
    no_cache: bool | None,
    kv_cache_strategy: str | None,
    kv_cache_window_tokens: int | None,
    offload_cpu_layers: int | None,
    offload_gpu_layers: int | None,
    force_download: bool | None,
    max_new_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    stats: bool | None,
    verbose: bool | None,
    quiet: bool | None,
    system: str,
    resume: Path | None,
    save: Path | None,
    history_file: Path | None,
    session_name: str,
    stream: bool | None,
    plain: bool,
    no_color: bool,
    print_config_flag: bool,
    plan_json_flag: bool,
) -> None:
    settings = load_app_settings()
    runtime_config = build_runtime_config(
        model=model,
        models_dir=models_dir,
        device=device,
        backend=backend,
        adapter_dir=adapter_dir,
        multimodal=multimodal,
        no_specialization=no_specialization,
        cache_dir=cache_dir,
        no_cache=no_cache,
        kv_cache_strategy=kv_cache_strategy,
        kv_cache_window_tokens=kv_cache_window_tokens,
        offload_cpu_layers=offload_cpu_layers,
        offload_gpu_layers=offload_gpu_layers,
        force_download=force_download,
        stats=stats,
        verbose=verbose,
        quiet=quiet,
        settings=settings,
    )
    generation_config = build_generation_config(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        stream=stream,
        settings=settings,
    )

    console = build_console(no_color=no_color)
    if plan_json_flag:
        if print_config_flag:
            raise typer.BadParameter(
                "--plan-json cannot be combined with --print-config"
            )
        print_json(console, services.application_service.describe_plan(runtime_config))
        return
    ensure_interactive_terminal()
    if print_config_flag:
        print_json(console, config_as_dict(runtime_config, generation_config))

    transcript_path = None if save is None else save.expanduser().resolve()
    history_path = (
        history_file.expanduser().resolve() if history_file is not None else None
    )
    session = services.application_service.create_session(
        runtime_config=runtime_config,
        generation_config=generation_config,
        session_name=session_name,
        system_prompt=system,
        autosave_path=transcript_path,
    )

    if resume is not None:
        session.load(resume.expanduser().resolve())
        if transcript_path is not None:
            session.autosave_path = transcript_path

    shell = InteractiveChatShell(
        session=session,
        console=console,
        history_file=history_path,
        plain=plain,
    )
    shell.run()


def register_chat_surfaces(app: typer.Typer, services: CommandServices) -> None:
    @app.callback(invoke_without_command=True)
    @app.command("chat")
    def chat_command(
        ctx: typer.Context,
        model: str | None = typer.Option(
            None, "--model", help="Model reference to resolve."
        ),
        models_dir: Path | None = typer.Option(
            None, "--models-dir", help="Directory containing model data."
        ),
        device: str | None = typer.Option(
            None, "--device", help="Torch device string."
        ),
        backend: str | None = typer.Option(None, "--backend", help="Backend override."),
        adapter_dir: Path | None = typer.Option(
            None, "--adapter-dir", help="Optional PEFT adapter directory."
        ),
        multimodal: bool | None = typer.Option(
            None,
            "--multimodal/--no-multimodal",
            help="Enable multimodal processor support.",
        ),
        no_specialization: bool | None = typer.Option(
            None,
            "--no-specialization",
            help="Disable optimized specialization selection.",
        ),
        cache_dir: Path | None = typer.Option(
            None, "--cache-dir", help="KV cache directory."
        ),
        no_cache: bool | None = typer.Option(
            None, "--no-cache", help="Disable disk KV cache."
        ),
        kv_cache_strategy: str | None = typer.Option(
            None,
            "--kv-cache-strategy",
            help=(
                "Disk KV strategy: chunked, paged, streamed-segmented, "
                "log-structured-journal, sliding-window-ring-buffer, "
                "quantized-cold-tier, or tiered-write-back."
            ),
        ),
        kv_cache_window_tokens: int | None = typer.Option(
            None,
            "--kv-cache-window-tokens",
            min=1,
            help=("Recent-context token budget for sliding-window-ring-buffer mode."),
        ),
        offload_cpu_layers: int | None = typer.Option(
            None,
            "--offload-cpu-layers",
            min=0,
            help="Number of layers to offload to CPU.",
        ),
        offload_gpu_layers: int | None = typer.Option(
            None,
            "--offload-gpu-layers",
            min=0,
            help="Number of layers to keep on GPU when using mixed offload.",
        ),
        force_download: bool | None = typer.Option(
            None, "--force-download", help="Force redownload of the selected model."
        ),
        max_new_tokens: int | None = typer.Option(
            None, "--max-new-tokens", min=1, help="Maximum generated tokens."
        ),
        temperature: float | None = typer.Option(
            None, "--temperature", min=0.0, help="Sampling temperature."
        ),
        top_p: float | None = typer.Option(
            None, "--top-p", min=0.0, max=1.0, help="Top-p sampling cutoff."
        ),
        top_k: int | None = typer.Option(
            None, "--top-k", min=1, help="Top-k sampling cutoff."
        ),
        seed: int | None = typer.Option(
            None, "--seed", help="Random seed for sampling."
        ),
        stats: bool | None = typer.Option(
            None, "--stats", help="Enable runtime stats collection."
        ),
        verbose: bool | None = typer.Option(
            None, "--verbose", help="Enable verbose runtime logging."
        ),
        quiet: bool | None = typer.Option(
            None, "--quiet", help="Suppress non-essential runtime output."
        ),
        system: str = typer.Option(
            "You are a helpful assistant.",
            "--system",
            help="System prompt for the session.",
        ),
        resume: Path | None = typer.Option(
            None, "--resume", help="Load a saved transcript before starting."
        ),
        save: Path | None = typer.Option(
            None, "--save", help="Autosave transcript path."
        ),
        history_file: Path | None = typer.Option(
            None, "--history-file", help="Prompt history file for interactive input."
        ),
        session_name: str = typer.Option(
            "default", "--session-name", help="Session name stored in transcripts."
        ),
        stream: bool | None = typer.Option(
            None,
            "--stream/--no-stream",
            help="Stream assistant output while generating.",
        ),
        plain: bool = typer.Option(
            False, "--plain", help="Use plain output without styled chrome."
        ),
        no_color: bool = typer.Option(
            False, "--no-color", help="Disable ANSI color output."
        ),
        print_config_flag: bool = typer.Option(
            False,
            "--print-config",
            help="Print resolved runtime config before starting.",
        ),
        plan_json_flag: bool = typer.Option(
            False,
            "--plan-json",
            help="Print the resolved runtime plan as JSON and exit.",
        ),
    ) -> None:
        if ctx.invoked_subcommand is not None and ctx.info_name != "chat":
            return
        run_chat_command(
            services=services,
            model=model,
            models_dir=models_dir,
            device=device,
            backend=backend,
            adapter_dir=adapter_dir,
            multimodal=multimodal,
            no_specialization=no_specialization,
            cache_dir=cache_dir,
            no_cache=no_cache,
            kv_cache_strategy=kv_cache_strategy,
            kv_cache_window_tokens=kv_cache_window_tokens,
            offload_cpu_layers=offload_cpu_layers,
            offload_gpu_layers=offload_gpu_layers,
            force_download=force_download,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stats=stats,
            verbose=verbose,
            quiet=quiet,
            system=system,
            resume=resume,
            save=save,
            history_file=history_file,
            session_name=session_name,
            stream=stream,
            plain=plain,
            no_color=no_color,
            print_config_flag=print_config_flag,
            plan_json_flag=plan_json_flag,
        )

from pathlib import Path

import typer

from ollm.app.session import ChatSession
from ollm.cli.common import build_console, build_generation_config, build_runtime_config, config_as_dict, ensure_interactive_terminal, print_json
from ollm.cli.services import CommandServices
from ollm.ui.chat_shell import InteractiveChatShell


def run_chat_command(
    services: CommandServices,
    model: str,
    models_dir: Path,
    device: str,
    provider_endpoint: str | None,
    adapter_dir: Path | None,
    multimodal: bool,
    cache_dir: Path,
    no_cache: bool,
    offload_cpu_layers: int,
    offload_gpu_layers: int,
    force_download: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    stats: bool,
    verbose: bool,
    quiet: bool,
    system: str,
    resume: Path | None,
    save: Path | None,
    history_file: Path | None,
    session_name: str,
    stream: bool,
    plain: bool,
    no_color: bool,
    print_config_flag: bool,
) -> None:
    ensure_interactive_terminal()
    runtime_config = build_runtime_config(
        model=model,
        models_dir=models_dir,
        device=device,
        provider_endpoint=provider_endpoint,
        adapter_dir=adapter_dir,
        multimodal=multimodal,
        cache_dir=cache_dir,
        no_cache=no_cache,
        offload_cpu_layers=offload_cpu_layers,
        offload_gpu_layers=offload_gpu_layers,
        force_download=force_download,
        stats=stats,
        verbose=verbose,
        quiet=quiet,
    )
    generation_config = build_generation_config(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        stream=stream,
    )

    console = build_console(no_color=no_color)
    if print_config_flag:
        print_json(console, config_as_dict(runtime_config, generation_config))

    transcript_path = None if save is None else save.expanduser().resolve()
    history_path = history_file.expanduser().resolve() if history_file is not None else None
    session = ChatSession(
        runtime_loader=services.runtime_loader,
        runtime_executor=services.runtime_executor,
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
        model: str = typer.Option("llama3-1B-chat", "--model", help="Model reference to resolve."),
        models_dir: Path = typer.Option(Path("models"), "--models-dir", help="Directory containing model data."),
        device: str = typer.Option("cuda:0", "--device", help="Torch device string."),
        provider_endpoint: str | None = typer.Option(None, "--provider-endpoint", help="Provider API root URL."),
        adapter_dir: Path | None = typer.Option(None, "--adapter-dir", help="Optional PEFT adapter directory."),
        multimodal: bool = typer.Option(False, "--multimodal/--no-multimodal", help="Enable multimodal processor support."),
        cache_dir: Path = typer.Option(Path("kv_cache"), "--cache-dir", help="KV cache directory."),
        no_cache: bool = typer.Option(False, "--no-cache", help="Disable disk KV cache."),
        offload_cpu_layers: int = typer.Option(0, "--offload-cpu-layers", min=0, help="Number of layers to offload to CPU."),
        offload_gpu_layers: int = typer.Option(0, "--offload-gpu-layers", min=0, help="Number of layers to keep on GPU when using mixed offload."),
        force_download: bool = typer.Option(False, "--force-download", help="Force redownload of the selected model."),
        max_new_tokens: int = typer.Option(500, "--max-new-tokens", min=1, help="Maximum generated tokens."),
        temperature: float = typer.Option(0.0, "--temperature", min=0.0, help="Sampling temperature."),
        top_p: float | None = typer.Option(None, "--top-p", min=0.0, max=1.0, help="Top-p sampling cutoff."),
        top_k: int | None = typer.Option(None, "--top-k", min=1, help="Top-k sampling cutoff."),
        seed: int | None = typer.Option(None, "--seed", help="Random seed for sampling."),
        stats: bool = typer.Option(False, "--stats", help="Enable runtime stats collection."),
        verbose: bool = typer.Option(False, "--verbose", help="Enable verbose runtime logging."),
        quiet: bool = typer.Option(False, "--quiet", help="Suppress non-essential runtime output."),
        system: str = typer.Option("You are a helpful assistant.", "--system", help="System prompt for the session."),
        resume: Path | None = typer.Option(None, "--resume", help="Load a saved transcript before starting."),
        save: Path | None = typer.Option(None, "--save", help="Autosave transcript path."),
        history_file: Path | None = typer.Option(None, "--history-file", help="Prompt history file for interactive input."),
        session_name: str = typer.Option("default", "--session-name", help="Session name stored in transcripts."),
        stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream assistant output while generating."),
        plain: bool = typer.Option(False, "--plain", help="Use plain output without styled chrome."),
        no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
        print_config_flag: bool = typer.Option(False, "--print-config", help="Print resolved runtime config before starting."),
    ) -> None:
        if ctx.invoked_subcommand is not None and ctx.info_name != "chat":
            return
        run_chat_command(
            services=services,
            model=model,
            models_dir=models_dir,
            device=device,
            provider_endpoint=provider_endpoint,
            adapter_dir=adapter_dir,
            multimodal=multimodal,
            cache_dir=cache_dir,
            no_cache=no_cache,
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
        )

import json
import sys
from pathlib import Path

import typer

from ollm.app.history import write_private_text
from ollm.app.types import ContentKind, ContentPart
from ollm.cli.common import (
    build_console,
    build_generation_config,
    build_runtime_config,
    config_as_dict,
    print_json,
)
from ollm.cli.services import CommandServices
from ollm.runtime.settings import load_app_settings
from ollm.runtime.streaming import StreamSink


class PromptStreamSink(StreamSink):
    def __init__(self, console):
        self._console = console
        self._line_open = False

    def on_status(self, message: str) -> None:
        del message

    def on_text(self, text: str) -> None:
        if not self._line_open:
            self._line_open = True
        self._console.print(text, end="")

    def on_complete(self, text: str) -> None:
        del text
        if self._line_open:
            self._console.print()
            self._line_open = False


def register_prompt_command(app: typer.Typer, services: CommandServices) -> None:
    @app.command("prompt")
    def prompt_command(
        prompt: str | None = typer.Argument(None, help="Prompt text."),
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
            help="System prompt for the request.",
        ),
        file: Path | None = typer.Option(
            None, "--file", help="Read prompt text from a file."
        ),
        stdin: bool = typer.Option(
            False, "--stdin", help="Read prompt text from stdin."
        ),
        image: list[str] | None = typer.Option(
            None, "--image", help="Image path or URL. Repeatable."
        ),
        audio: list[str] | None = typer.Option(
            None, "--audio", help="Audio path or URL. Repeatable."
        ),
        stream: bool | None = typer.Option(
            None,
            "--stream/--no-stream",
            help="Stream assistant output while generating.",
        ),
        output: Path | None = typer.Option(
            None, "--output", help="Write the final response to a file."
        ),
        format: str = typer.Option(
            "text", "--format", help="Output format: text or json."
        ),
        show_prompt: bool = typer.Option(
            False, "--show-prompt", help="Print the resolved request before execution."
        ),
        print_config_flag: bool = typer.Option(
            False,
            "--print-config",
            help="Print resolved runtime config before running.",
        ),
        plan_json_flag: bool = typer.Option(
            False,
            "--plan-json",
            help="Print the resolved runtime plan as JSON and exit.",
        ),
        no_color: bool = typer.Option(
            False, "--no-color", help="Disable ANSI color output."
        ),
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
            if output is not None:
                raise typer.BadParameter("--plan-json cannot be combined with --output")
            if print_config_flag:
                raise typer.BadParameter(
                    "--plan-json cannot be combined with --print-config"
                )
            print_json(
                console, services.application_service.describe_plan(runtime_config)
            )
            return
        if print_config_flag:
            print_json(console, config_as_dict(runtime_config, generation_config))

        prompt_text = _resolve_prompt_text(prompt, file, stdin)
        parts = [ContentPart.text(prompt_text)]
        if image:
            parts.extend(ContentPart.image(item) for item in image)
        if audio:
            parts.extend(ContentPart.audio(item) for item in audio)

        if (
            any(part.kind is not ContentKind.TEXT for part in parts[1:])
            and not runtime_config.multimodal
        ):
            raise typer.BadParameter("--image and --audio require --multimodal")
        if format not in {"text", "json"}:
            raise typer.BadParameter("--format must be either 'text' or 'json'")
        if format == "json" and generation_config.stream:
            raise typer.BadParameter("--format json cannot be combined with --stream")

        if show_prompt:
            console.print(f"[bold]system:[/bold] {system}")
            console.print(f"[bold]prompt:[/bold] {prompt_text}")

        if any(part.kind is not ContentKind.TEXT for part in parts):
            runtime_config.multimodal = True

        try:
            sink = PromptStreamSink(console) if generation_config.stream else None
            response = services.application_service.prompt_parts(
                parts,
                runtime_config=runtime_config,
                generation_config=generation_config,
                system_prompt=system,
                sink=sink,
            )
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1)

        if format == "json":
            payload: dict[str, object] = {
                "text": response.text,
                "metadata": dict(response.metadata),
            }
            if output is not None:
                write_private_text(output, json.dumps(payload, indent=2) + "\n")
            else:
                print_json(console, payload)
            return

        if output is not None:
            write_private_text(output, response.text)
            return
        if not generation_config.stream:
            console.print(response.text)
        if "stats" in response.metadata and not quiet:
            console.print(f"[dim]{response.metadata['stats']}[/dim]")


def _resolve_prompt_text(
    prompt: str | None, file: Path | None, stdin_flag: bool
) -> str:
    provided_sources = 0
    if prompt is not None:
        provided_sources += 1
    if file is not None:
        provided_sources += 1
    if stdin_flag:
        provided_sources += 1
    if provided_sources > 1:
        raise typer.BadParameter("PROMPT, --file, and --stdin are mutually exclusive")

    if prompt is not None:
        return prompt
    if file is not None:
        return file.expanduser().resolve().read_text(encoding="utf-8")
    if stdin_flag or not sys.stdin.isatty():
        return sys.stdin.read()
    raise typer.BadParameter("Provide PROMPT text, --file, or --stdin")

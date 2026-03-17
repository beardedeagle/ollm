from pathlib import Path

import typer

from ollm.cli.common import build_console, build_runtime_config, print_json
from ollm.cli.services import CommandServices
from ollm.runtime.settings import load_app_settings


def register_doctor_command(app: typer.Typer, services: CommandServices) -> None:
    @app.command("doctor")
    def doctor_command(
        model: str | None = typer.Option(
            None, "--model", help="Model reference to inspect."
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
            help="Enable multimodal processor support for plan checks.",
        ),
        no_specialization: bool | None = typer.Option(
            None,
            "--no-specialization",
            help="Disable optimized specialization selection.",
        ),
        imports: bool = typer.Option(False, "--imports", help="Run import checks."),
        runtime: bool = typer.Option(
            False, "--runtime", help="Run runtime availability checks."
        ),
        paths: bool = typer.Option(
            False, "--paths", help="Run path and writability checks."
        ),
        download: bool = typer.Option(
            False, "--download", help="Include download-readiness details."
        ),
        json_output: bool = typer.Option(False, "--json", help="Output JSON."),
        plan_json_flag: bool = typer.Option(
            False,
            "--plan-json",
            help="Print the resolved runtime plan as JSON and exit.",
        ),
        verbose: bool | None = typer.Option(
            None, "--verbose", help="Include verbose output."
        ),
        no_color: bool = typer.Option(
            False, "--no-color", help="Disable ANSI color output."
        ),
    ) -> None:
        if not any((imports, runtime, paths, download)):
            imports = True
            runtime = True
            paths = True

        runtime_config = build_runtime_config(
            model=model,
            models_dir=models_dir,
            device=device,
            backend=backend,
            adapter_dir=adapter_dir,
            multimodal=multimodal,
            no_specialization=no_specialization,
            cache_dir=None,
            no_cache=None,
            offload_cpu_layers=None,
            offload_gpu_layers=None,
            force_download=None,
            stats=None,
            verbose=verbose,
            quiet=None,
            settings=load_app_settings(),
        )
        console = build_console(no_color=no_color)
        if plan_json_flag:
            print_json(
                console, services.application_service.describe_plan(runtime_config)
            )
            raise typer.Exit(code=0)
        report = services.application_service.run_doctor(
            runtime_config=runtime_config,
            include_imports=imports,
            include_runtime=runtime,
            include_paths=paths,
            include_download=download,
        )
        if json_output:
            print_json(console, report.as_dict())
            raise typer.Exit(code=0 if report.ok() else 1)

        for check in report.checks:
            status = "[green]OK[/green]" if check.ok else "[red]FAIL[/red]"
            console.print(f"{status} {check.name}: {check.message}")
            if verbose and check.details:
                for key, value in check.details.items():
                    console.print(f"  {key}: {value}")
        raise typer.Exit(code=0 if report.ok() else 1)

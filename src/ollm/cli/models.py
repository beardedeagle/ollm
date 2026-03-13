from pathlib import Path

import typer

from ollm.cli.common import build_console, print_json
from ollm.cli.services import CommandServices
from ollm.runtime.catalog import get_model_catalog_entry, list_model_catalog


def register_models_command(app: typer.Typer, services: CommandServices) -> None:
    models_app = typer.Typer(help="Inspect and manage built-in model ids.")

    @models_app.command("list")
    def list_models(
        installed: bool = typer.Option(False, "--installed", help="Show only installed models."),
        json_output: bool = typer.Option(False, "--json", help="Output JSON."),
        models_dir: Path = typer.Option(Path("models"), "--models-dir", help="Directory containing model data."),
        no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
    ) -> None:
        model_dir = models_dir.expanduser().resolve()
        entries = []
        for entry in list_model_catalog():
            model_path = model_dir / entry.model_id
            is_installed = model_path.exists()
            if installed and not is_installed:
                continue
            entries.append(
                {
                    "model_id": entry.model_id,
                    "summary": entry.summary,
                    "modalities": [modality.value for modality in entry.modalities],
                    "installed": is_installed,
                    "path": str(model_path),
                }
            )

        console = build_console(no_color=no_color)
        if json_output:
            print_json(console, {"models": entries})
            return

        for entry in entries:
            status = "installed" if entry["installed"] else "not-installed"
            console.print(f"{entry['model_id']} [{status}] - {entry['summary']}")

    @models_app.command("info")
    def model_info(
        model: str = typer.Argument(..., help="Model id."),
        models_dir: Path = typer.Option(Path("models"), "--models-dir", help="Directory containing model data."),
        json_output: bool = typer.Option(False, "--json", help="Output JSON."),
        no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
    ) -> None:
        entry = get_model_catalog_entry(model)
        model_path = models_dir.expanduser().resolve() / entry.model_id
        payload = {
            "model_id": entry.model_id,
            "summary": entry.summary,
            "repo_id": entry.repo_id,
            "modalities": [modality.value for modality in entry.modalities],
            "requires_processor": entry.requires_processor,
            "supports_disk_cache": entry.supports_disk_cache,
            "installed": model_path.exists(),
            "path": str(model_path),
        }
        console = build_console(no_color=no_color)
        if json_output:
            print_json(console, payload)
            return
        console.print(f"model: {payload['model_id']}")
        console.print(f"repo: {payload['repo_id']}")
        console.print(f"modalities: {', '.join(payload['modalities'])}")
        console.print(f"path: {payload['path']}")
        console.print(f"installed: {payload['installed']}")
        console.print(payload["summary"])

    @models_app.command("download")
    def download_model(
        model: str = typer.Argument(..., help="Model id."),
        models_dir: Path = typer.Option(Path("models"), "--models-dir", help="Directory to store model data."),
        force: bool = typer.Option(False, "--force", help="Force redownload."),
        no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
    ) -> None:
        model_path = services.runtime_loader.download(model, models_dir.expanduser().resolve(), force_download=force)
        console = build_console(no_color=no_color)
        console.print(f"Downloaded {model} to {model_path}")

    @models_app.command("path")
    def model_path(
        model: str = typer.Argument(..., help="Model id."),
        models_dir: Path = typer.Option(Path("models"), "--models-dir", help="Directory containing model data."),
    ) -> None:
        entry = get_model_catalog_entry(model)
        typer.echo(str(models_dir.expanduser().resolve() / entry.model_id))

    app.add_typer(models_app, name="models")

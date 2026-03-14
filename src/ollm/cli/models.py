from pathlib import Path

import typer

from ollm.cli.common import build_console, print_json
from ollm.cli.services import CommandServices
from ollm.runtime.catalog import list_model_catalog


def _resolved_model_payload(resolved_model) -> dict[str, object]:
    return {
        "model_reference": resolved_model.reference.raw,
        "normalized_name": resolved_model.normalized_name,
        "source_kind": resolved_model.source_kind.value,
        "support_level": resolved_model.capabilities.support_level.value,
        "modalities": [modality.value for modality in resolved_model.capabilities.modalities],
        "requires_processor": resolved_model.capabilities.requires_processor,
        "supports_disk_cache": resolved_model.capabilities.supports_disk_cache,
        "supports_specialization": resolved_model.capabilities.supports_specialization,
        "repo_id": resolved_model.repo_id,
        "revision": resolved_model.revision,
        "path": None if resolved_model.model_path is None else str(resolved_model.model_path),
        "native_family": None if resolved_model.native_family is None else resolved_model.native_family.value,
        "architecture": resolved_model.architecture,
        "model_type": resolved_model.model_type,
        "generic_model_kind": None if resolved_model.generic_model_kind is None else resolved_model.generic_model_kind.value,
        "resolution_message": resolved_model.resolution_message,
    }


def register_models_command(app: typer.Typer, services: CommandServices) -> None:
    models_app = typer.Typer(help="Inspect built-in aliases and discovered model references.")

    @models_app.command("list")
    def list_models(
        installed: bool = typer.Option(False, "--installed", help="Show only materialized local models."),
        json_output: bool = typer.Option(False, "--json", help="Output JSON."),
        models_dir: Path = typer.Option(Path("models"), "--models-dir", help="Directory containing model data."),
        no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
    ) -> None:
        model_dir = models_dir.expanduser().resolve()
        entries: list[dict[str, object]] = []
        seen_paths: set[str] = set()

        for entry in list_model_catalog():
            resolved_model = services.runtime_loader.resolve(entry.model_id, model_dir)
            payload = _resolved_model_payload(resolved_model)
            payload["known"] = True
            payload["installed"] = bool(resolved_model.model_path is not None and resolved_model.model_path.exists())
            if installed and not payload["installed"]:
                continue
            entries.append(payload)
            if resolved_model.model_path is not None:
                seen_paths.add(str(resolved_model.model_path))

        for resolved_model in services.runtime_loader.discover_local_models(model_dir):
            path = None if resolved_model.model_path is None else str(resolved_model.model_path)
            if path is not None and path in seen_paths:
                continue
            payload = _resolved_model_payload(resolved_model)
            payload["known"] = False
            payload["installed"] = bool(resolved_model.model_path is not None and resolved_model.model_path.exists())
            if installed and not payload["installed"]:
                continue
            entries.append(payload)

        entries.sort(key=lambda item: (str(item["source_kind"]), str(item["model_reference"])))
        console = build_console(no_color=no_color)
        if json_output:
            print_json(console, {"models": entries})
            return

        for entry in entries:
            status = "installed" if entry["installed"] else "not-installed"
            known = "built-in" if entry["known"] else "discovered"
            console.print(
                f"{entry['model_reference']} [{known} / {status}] - "
                f"{entry['support_level']} - {entry['resolution_message']}"
            )

    @models_app.command("info")
    def model_info(
        model: str = typer.Argument(..., help="Model reference."),
        models_dir: Path = typer.Option(Path("models"), "--models-dir", help="Directory containing model data."),
        json_output: bool = typer.Option(False, "--json", help="Output JSON."),
        no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
    ) -> None:
        resolved_model = services.runtime_loader.resolve(model, models_dir.expanduser().resolve())
        payload = _resolved_model_payload(resolved_model)
        payload["installed"] = bool(resolved_model.model_path is not None and resolved_model.model_path.exists())
        console = build_console(no_color=no_color)
        if json_output:
            print_json(console, payload)
            return
        console.print(f"reference: {payload['model_reference']}")
        console.print(f"normalized: {payload['normalized_name']}")
        console.print(f"source: {payload['source_kind']}")
        console.print(f"support: {payload['support_level']}")
        console.print(f"specialization: {payload['supports_specialization']}")
        if payload["generic_model_kind"] is not None:
            console.print(f"generic-kind: {payload['generic_model_kind']}")
        if payload["architecture"] is not None:
            console.print(f"architecture: {payload['architecture']}")
        if payload["model_type"] is not None:
            console.print(f"model-type: {payload['model_type']}")
        if payload["repo_id"] is not None:
            console.print(f"repo: {payload['repo_id']}")
        if payload["revision"] is not None:
            console.print(f"revision: {payload['revision']}")
        if payload["path"] is not None:
            console.print(f"path: {payload['path']}")
        console.print(f"modalities: {', '.join(payload['modalities'])}")
        console.print(payload["resolution_message"])

    @models_app.command("download")
    def download_model(
        model: str = typer.Argument(..., help="Model reference."),
        models_dir: Path = typer.Option(Path("models"), "--models-dir", help="Directory to store model data."),
        force: bool = typer.Option(False, "--force", help="Force redownload."),
        no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
    ) -> None:
        model_path = services.runtime_loader.download(model, models_dir.expanduser().resolve(), force_download=force)
        console = build_console(no_color=no_color)
        console.print(f"Materialized {model} to {model_path}")

    @models_app.command("path")
    def model_path(
        model: str = typer.Argument(..., help="Model reference."),
        models_dir: Path = typer.Option(Path("models"), "--models-dir", help="Directory containing model data."),
    ) -> None:
        resolved_model = services.runtime_loader.resolve(model, models_dir.expanduser().resolve())
        if resolved_model.model_path is None:
            raise typer.BadParameter(
                f"Model reference '{model}' does not resolve to a local materialization path"
            )
        typer.echo(str(resolved_model.model_path))

    app.add_typer(models_app, name="models")

from pathlib import Path

import typer

from ollm.cli.common import build_console, print_json
from ollm.cli.services import CommandServices
from ollm.runtime.catalog import list_model_catalog
from ollm.runtime.inspection import (
    MergedRuntimePayload,
    RuntimePlanPayload,
    merged_runtime_payload,
    plan_json_payload,
)
from ollm.runtime.settings import (
    RuntimeConfigOverrides,
    default_app_settings,
    resolve_runtime_config,
)


def _discovery_entry(
    payload: MergedRuntimePayload,
    *,
    discovery_source: str,
) -> dict[str, object]:
    entry: dict[str, object] = {key: value for key, value in payload.items()}
    entry["discovery_source"] = discovery_source
    return entry


def register_models_command(app: typer.Typer, services: CommandServices) -> None:
    models_app = typer.Typer(
        help="Inspect built-in aliases and discovered local model materializations."
    )

    @models_app.command("list")
    def list_models(
        installed: bool = typer.Option(
            False, "--installed", help="Show only materialized local model references."
        ),
        backend: str | None = typer.Option(
            None, "--backend", help="Backend override for runtime planning output."
        ),
        no_specialization: bool = typer.Option(
            False,
            "--no-specialization",
            help="Disable optimized specialization selection for runtime planning output.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Output JSON."),
        models_dir: Path = typer.Option(
            Path("models"), "--models-dir", help="Directory containing model data."
        ),
        no_color: bool = typer.Option(
            False, "--no-color", help="Disable ANSI color output."
        ),
    ) -> None:
        model_dir = models_dir.expanduser().resolve()
        entries: list[dict[str, object]] = []
        seen_paths: set[str] = set()

        for entry in list_model_catalog():
            resolved_model = services.runtime_loader.resolve(entry.model_id, model_dir)
            installed_entry = bool(
                resolved_model.model_path is not None
                and resolved_model.model_path.exists()
            )
            runtime_plan = services.runtime_loader.plan(
                resolve_runtime_config(
                    default_app_settings().runtime,
                    RuntimeConfigOverrides(
                        model_reference=entry.model_id,
                        models_dir=model_dir,
                        backend=backend,
                        use_specialization=not no_specialization,
                    ),
                )
            )
            payload = _discovery_entry(
                merged_runtime_payload(
                    resolved_model,
                    runtime_plan,
                    materialized=installed_entry,
                ),
                discovery_source="built-in",
            )
            if installed and not bool(payload["materialized"]):
                continue
            entries.append(payload)
            if resolved_model.model_path is not None:
                seen_paths.add(str(resolved_model.model_path))

        for resolved_model in services.runtime_loader.discover_local_models(model_dir):
            path = (
                None
                if resolved_model.model_path is None
                else str(resolved_model.model_path)
            )
            if path is not None and path in seen_paths:
                continue
            installed_entry = bool(
                resolved_model.model_path is not None
                and resolved_model.model_path.exists()
            )
            runtime_plan = services.runtime_loader.plan(
                resolve_runtime_config(
                    default_app_settings().runtime,
                    RuntimeConfigOverrides(
                        model_reference=resolved_model.reference.raw,
                        models_dir=model_dir,
                        backend=backend,
                        use_specialization=not no_specialization,
                    ),
                )
            )
            payload = _discovery_entry(
                merged_runtime_payload(
                    resolved_model,
                    runtime_plan,
                    materialized=installed_entry,
                ),
                discovery_source="discovered-local",
            )
            if installed and not bool(payload["materialized"]):
                continue
            entries.append(payload)

        entries.sort(
            key=lambda item: (str(item["source_kind"]), str(item["model_reference"]))
        )
        console = build_console(no_color=no_color)
        if json_output:
            print_json(console, {"models": entries})
            return

        for entry in entries:
            known = entry["discovery_source"]
            console.print(
                f"{entry['model_reference']} [{known}] - {entry['support_level']} - "
                f"{entry['resolution_message']}"
            )

    @models_app.command("info")
    def model_info(
        model: str = typer.Argument(..., help="Model reference."),
        models_dir: Path = typer.Option(
            Path("models"), "--models-dir", help="Directory containing model data."
        ),
        backend: str | None = typer.Option(None, "--backend", help="Backend override."),
        multimodal: bool = typer.Option(
            False,
            "--multimodal/--no-multimodal",
            help="Enable multimodal processor support for runtime planning.",
        ),
        no_specialization: bool = typer.Option(
            False,
            "--no-specialization",
            help="Disable optimized specialization selection.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Output JSON."),
        plan_json_flag: bool = typer.Option(
            False, "--plan-json", help="Output structured runtime plan JSON."
        ),
        no_color: bool = typer.Option(
            False, "--no-color", help="Disable ANSI color output."
        ),
    ) -> None:
        runtime_config = resolve_runtime_config(
            default_app_settings().runtime,
            RuntimeConfigOverrides(
                model_reference=model,
                models_dir=models_dir.expanduser().resolve(),
                backend=backend,
                multimodal=multimodal,
                use_specialization=not no_specialization,
            ),
        )
        resolved_model = services.runtime_loader.resolve(
            model, runtime_config.resolved_models_dir()
        )
        materialized = bool(
            resolved_model.model_path is not None and resolved_model.model_path.exists()
        )
        runtime_plan = services.runtime_loader.plan(runtime_config)
        payload: MergedRuntimePayload = merged_runtime_payload(
            resolved_model,
            runtime_plan,
            materialized=materialized,
        )
        console = build_console(no_color=no_color)
        if plan_json_flag:
            print_json(console, plan_json_payload(runtime_config, runtime_plan))
            return
        if json_output:
            print_json(console, payload)
            return
        console.print(f"reference: {payload['model_reference']}")
        console.print(f"normalized: {payload['normalized_name']}")
        console.print(f"source: {payload['source_kind']}")
        console.print(f"materialized: {payload['materialized']}")
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
        runtime_plan_payload: RuntimePlanPayload = payload["runtime_plan"]
        console.print(f"backend: {runtime_plan_payload['backend_id']}")
        console.print(
            f"specialization-state: {runtime_plan_payload['specialization_state']}"
        )
        console.print(
            f"planned-passes: {', '.join(runtime_plan_payload['planned_specialization_pass_ids'])}"
        )
        console.print(f"modalities: {', '.join(payload['modalities'])}")
        console.print(payload["resolution_message"])

    @models_app.command("download")
    def download_model(
        model: str = typer.Argument(..., help="Model reference."),
        models_dir: Path = typer.Option(
            Path("models"),
            "--models-dir",
            help="Directory to store materialized runtime artifacts.",
        ),
        force: bool = typer.Option(False, "--force", help="Force redownload."),
        no_color: bool = typer.Option(
            False, "--no-color", help="Disable ANSI color output."
        ),
    ) -> None:
        model_path = services.runtime_loader.download(
            model, models_dir.expanduser().resolve(), force_download=force
        )
        console = build_console(no_color=no_color)
        console.print(f"Materialized runtime artifacts for {model} to {model_path}")

    @models_app.command("path")
    def model_path(
        model: str = typer.Argument(..., help="Model reference."),
        models_dir: Path = typer.Option(
            Path("models"), "--models-dir", help="Directory containing model data."
        ),
    ) -> None:
        resolved_model = services.runtime_loader.resolve(
            model, models_dir.expanduser().resolve()
        )
        if resolved_model.model_path is None:
            raise typer.BadParameter(
                f"Model reference '{model}' does not resolve to a local materialization path"
            )
        typer.echo(str(resolved_model.model_path))

    app.add_typer(models_app, name="models")

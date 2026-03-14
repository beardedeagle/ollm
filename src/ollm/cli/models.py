from pathlib import Path

import typer

from ollm.cli.common import build_console, print_json
from ollm.cli.services import CommandServices
from ollm.runtime.catalog import list_model_catalog
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel


def _resolved_model_payload(resolved_model: ResolvedModel) -> dict[str, object]:
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


def _runtime_plan_payload(runtime_plan: RuntimePlan) -> dict[str, object]:
    return {
        "backend_id": runtime_plan.backend_id,
        "supports_disk_cache": runtime_plan.supports_disk_cache,
        "supports_cpu_offload": runtime_plan.supports_cpu_offload,
        "supports_gpu_offload": runtime_plan.supports_gpu_offload,
        "specialization_provider_id": runtime_plan.specialization_provider_id,
        "specialization_state": runtime_plan.specialization_state.value,
        "planned_specialization_pass_ids": [
            pass_id.value for pass_id in runtime_plan.specialization_pass_ids
        ],
        "reason": runtime_plan.reason,
    }


def _merge_runtime_plan_payload(
    payload: dict[str, object],
    runtime_plan: RuntimePlan,
) -> dict[str, object]:
    merged_payload = dict(payload)
    merged_payload["resolved_support_level"] = payload["support_level"]
    merged_payload["resolved_supports_disk_cache"] = payload["supports_disk_cache"]
    merged_payload["resolved_resolution_message"] = payload["resolution_message"]
    merged_payload["support_level"] = runtime_plan.support_level.value
    merged_payload["supports_disk_cache"] = runtime_plan.supports_disk_cache
    merged_payload["resolution_message"] = runtime_plan.reason
    merged_payload["runtime_plan"] = _runtime_plan_payload(runtime_plan)
    return merged_payload


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
            if payload["installed"]:
                runtime_plan = services.runtime_loader.plan(
                    RuntimeConfig(model_reference=entry.model_id, models_dir=model_dir)
                )
                payload = _merge_runtime_plan_payload(payload, runtime_plan)
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
            if payload["installed"]:
                runtime_plan = services.runtime_loader.plan(
                    RuntimeConfig(model_reference=resolved_model.reference.raw, models_dir=model_dir)
                )
                payload = _merge_runtime_plan_payload(payload, runtime_plan)
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
        provider_endpoint: str | None = typer.Option(None, "--provider-endpoint", help="Provider API root URL."),
        multimodal: bool = typer.Option(False, "--multimodal/--no-multimodal", help="Enable multimodal processor support for runtime planning."),
        json_output: bool = typer.Option(False, "--json", help="Output JSON."),
        no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
    ) -> None:
        resolved_model = services.runtime_loader.resolve(model, models_dir.expanduser().resolve())
        payload = _resolved_model_payload(resolved_model)
        payload["installed"] = bool(resolved_model.model_path is not None and resolved_model.model_path.exists())
        runtime_plan = services.runtime_loader.plan(
            RuntimeConfig(
                model_reference=model,
                models_dir=models_dir.expanduser().resolve(),
                provider_endpoint=provider_endpoint,
                multimodal=multimodal,
            )
        )
        if resolved_model.source_kind is ModelSourceKind.PROVIDER and runtime_plan.is_executable():
            payload["installed"] = True
        payload = _merge_runtime_plan_payload(payload, runtime_plan)
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
        runtime_plan = payload.get("runtime_plan")
        if runtime_plan is not None:
            console.print(f"backend: {runtime_plan['backend_id']}")
            console.print(f"specialization-state: {runtime_plan['specialization_state']}")
            console.print(f"planned-passes: {', '.join(runtime_plan['planned_specialization_pass_ids'])}")
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

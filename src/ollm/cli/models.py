from pathlib import Path

import typer

from ollm.cli.common import build_console, print_json
from ollm.cli.services import CommandServices
from ollm.runtime.catalog import list_model_catalog
from ollm.runtime.config import RuntimeConfig, normalize_provider_endpoint
from ollm.runtime.inspection import merged_runtime_payload, plan_json_payload
from ollm.runtime.loader import DiscoveredRuntimeModel


def _availability_label(entry: dict[str, object]) -> str:
    return str(entry["availability_status"])


def _provider_discovery_names(
    discover_provider: list[str] | None,
    provider_endpoint: str | None,
) -> tuple[str, ...]:
    if provider_endpoint is not None:
        normalize_provider_endpoint(provider_endpoint)
    if not discover_provider:
        return ("ollama", "lmstudio")

    normalized_providers: list[str] = []
    for provider_name in discover_provider:
        normalized_name = provider_name.strip().lower()
        if normalized_name not in {"ollama", "lmstudio", "openai-compatible", "msty"}:
            raise typer.BadParameter(
                "--discover-provider must be one of: ollama, lmstudio, openai-compatible, msty"
            )
        if normalized_name in {"openai-compatible", "msty"} and provider_endpoint is None:
            raise typer.BadParameter(
                f"--discover-provider {normalized_name} requires --provider-endpoint"
            )
        if normalized_name not in normalized_providers:
            normalized_providers.append(normalized_name)
    return tuple(normalized_providers)


def _provider_runtime_config(
    discovered_model: DiscoveredRuntimeModel,
    models_dir: Path,
    backend: str | None = None,
    no_specialization: bool = False,
) -> RuntimeConfig:
    return RuntimeConfig(
        model_reference=discovered_model.model_reference,
        models_dir=models_dir,
        backend=backend,
        provider_endpoint=discovered_model.provider_endpoint,
        use_specialization=not no_specialization,
    )


def register_models_command(app: typer.Typer, services: CommandServices) -> None:
    models_app = typer.Typer(
        help="Inspect built-in aliases, local materializations, and provider-discovered model references."
    )

    @models_app.command("list")
    def list_models(
        installed: bool = typer.Option(False, "--installed", help="Show only materialized local model references."),
        backend: str | None = typer.Option(None, "--backend", help="Backend override for runtime planning output."),
        no_specialization: bool = typer.Option(False, "--no-specialization", help="Disable optimized specialization selection for runtime planning output."),
        discover_provider: list[str] | None = typer.Option(
            None,
            "--discover-provider",
            help=(
                "Provider to probe for discovered models. Repeatable. "
                "Defaults to ollama and lmstudio."
            ),
        ),
        provider_endpoint: str | None = typer.Option(
            None,
            "--provider-endpoint",
            help="Provider API root URL for msty/openai-compatible discovery or lmstudio override.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Output JSON."),
        models_dir: Path = typer.Option(Path("models"), "--models-dir", help="Directory containing model data."),
        no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
    ) -> None:
        model_dir = models_dir.expanduser().resolve()
        entries: list[dict[str, object]] = []
        seen_paths: set[str] = set()
        try:
            provider_names = _provider_discovery_names(discover_provider, provider_endpoint)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        for entry in list_model_catalog():
            resolved_model = services.runtime_loader.resolve(entry.model_id, model_dir)
            installed_entry = bool(resolved_model.model_path is not None and resolved_model.model_path.exists())
            runtime_plan = services.runtime_loader.plan(
                RuntimeConfig(
                    model_reference=entry.model_id,
                    models_dir=model_dir,
                    backend=backend,
                    use_specialization=not no_specialization,
                )
            )
            payload = merged_runtime_payload(
                resolved_model,
                runtime_plan,
                materialized=installed_entry,
            )
            payload["discovery_source"] = "built-in"
            if installed and not payload["materialized"]:
                continue
            entries.append(payload)
            if resolved_model.model_path is not None:
                seen_paths.add(str(resolved_model.model_path))

        for resolved_model in services.runtime_loader.discover_local_models(model_dir):
            path = None if resolved_model.model_path is None else str(resolved_model.model_path)
            if path is not None and path in seen_paths:
                continue
            installed_entry = bool(resolved_model.model_path is not None and resolved_model.model_path.exists())
            runtime_plan = services.runtime_loader.plan(
                RuntimeConfig(
                    model_reference=resolved_model.reference.raw,
                    models_dir=model_dir,
                    backend=backend,
                    use_specialization=not no_specialization,
                )
            )
            payload = merged_runtime_payload(
                resolved_model,
                runtime_plan,
                materialized=installed_entry,
            )
            payload["discovery_source"] = "discovered-local"
            if installed and not payload["materialized"]:
                continue
            entries.append(payload)

        try:
            discovered_provider_models = services.runtime_loader.discover_provider_models(
                model_dir,
                provider_names,
                provider_endpoint,
                strict=bool(discover_provider),
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        for discovered_model in discovered_provider_models:
            runtime_plan = services.runtime_loader.plan(
                _provider_runtime_config(
                    discovered_model,
                    model_dir,
                    backend=backend,
                    no_specialization=no_specialization,
                )
            )
            payload = merged_runtime_payload(
                discovered_model.resolved_model,
                runtime_plan,
                materialized=False,
            )
            payload["discovery_source"] = "discovered-provider"
            payload["provider_endpoint"] = discovered_model.provider_endpoint
            if installed and not payload["materialized"]:
                continue
            entries.append(payload)

        entries.sort(key=lambda item: (str(item["source_kind"]), str(item["model_reference"])))
        console = build_console(no_color=no_color)
        if json_output:
            print_json(console, {"models": entries})
            return

        for entry in entries:
            status = _availability_label(entry)
            known = entry["discovery_source"]
            console.print(
                f"{entry['model_reference']} [{known} / {status}] - "
                f"{entry['support_level']} - {entry['resolution_message']}"
            )

    @models_app.command("info")
    def model_info(
        model: str = typer.Argument(..., help="Model reference."),
        models_dir: Path = typer.Option(Path("models"), "--models-dir", help="Directory containing model data."),
        backend: str | None = typer.Option(None, "--backend", help="Backend override."),
        provider_endpoint: str | None = typer.Option(None, "--provider-endpoint", help="Provider API root URL."),
        multimodal: bool = typer.Option(False, "--multimodal/--no-multimodal", help="Enable multimodal processor support for runtime planning."),
        no_specialization: bool = typer.Option(False, "--no-specialization", help="Disable optimized specialization selection."),
        json_output: bool = typer.Option(False, "--json", help="Output JSON."),
        plan_json_flag: bool = typer.Option(False, "--plan-json", help="Output structured runtime plan JSON."),
        no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color output."),
    ) -> None:
        resolved_model = services.runtime_loader.resolve(model, models_dir.expanduser().resolve())
        materialized = bool(resolved_model.model_path is not None and resolved_model.model_path.exists())
        runtime_plan = services.runtime_loader.plan(
            RuntimeConfig(
                model_reference=model,
                models_dir=models_dir.expanduser().resolve(),
                backend=backend,
                provider_endpoint=provider_endpoint,
                multimodal=multimodal,
                use_specialization=not no_specialization,
            )
        )
        payload = merged_runtime_payload(
            resolved_model,
            runtime_plan,
            materialized=materialized,
        )
        console = build_console(no_color=no_color)
        if plan_json_flag:
            print_json(
                console,
                plan_json_payload(
                    RuntimeConfig(
                        model_reference=model,
                        models_dir=models_dir.expanduser().resolve(),
                        backend=backend,
                        provider_endpoint=provider_endpoint,
                        multimodal=multimodal,
                        use_specialization=not no_specialization,
                    ),
                    runtime_plan,
                ),
            )
            return
        if json_output:
            print_json(console, payload)
            return
        console.print(f"reference: {payload['model_reference']}")
        console.print(f"normalized: {payload['normalized_name']}")
        console.print(f"source: {payload['source_kind']}")
        if payload["provider_name"] is not None:
            console.print(f"provider: {payload['provider_name']}")
            console.print(f"availability: {payload['availability_status']}")
        else:
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
        runtime_plan = payload.get("runtime_plan")
        if runtime_plan is not None:
            console.print(f"backend: {runtime_plan['backend_id']}")
            if runtime_plan.get("audio_input_support"):
                console.print(f"audio-input: {runtime_plan['audio_input_support']}")
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

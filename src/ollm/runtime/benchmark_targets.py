"""Runtime benchmark target selection and probe composition helpers."""

from pathlib import Path

from ollm.client import RuntimeClient
from ollm.runtime.benchmark_commands import measure_runtime_probe
from ollm.runtime.benchmark_details import backend_pair_payload
from ollm.runtime.benchmark_probe_measurements import (
    measure_output_scaling_probe,
    measure_prompt_scaling_probe,
    measure_session_growth_probe,
    measure_warm_runtime_probe,
)
from ollm.runtime.benchmark_types import (
    DEFAULT_OUTPUT_TOKEN_TARGETS,
    DEFAULT_PROMPT_TOKEN_TARGETS,
    DEFAULT_SESSION_TURNS,
    CommandBenchmarkSpec,
    RuntimeComparisonTarget,
    unavailable_measurement,
)
from ollm.runtime.catalog import list_model_catalog


def build_runtime_probe_command(
    repo_root: Path,
    model_reference: str,
    *,
    models_dir: Path,
    device: str,
    backend: str,
    use_specialization: bool,
    probe_mode: str = "cold",
    prompt: str = "Say hi.",
    max_new_tokens: int = 4,
    iterations: int = 1,
    warmup_iterations: int = 0,
    prompt_token_targets: tuple[int, ...] = DEFAULT_PROMPT_TOKEN_TARGETS,
    output_token_targets: tuple[int, ...] = DEFAULT_OUTPUT_TOKEN_TARGETS,
    session_turns: int = DEFAULT_SESSION_TURNS,
) -> tuple[str, ...]:
    import sys

    command = [
        sys.executable,
        str((repo_root / "scripts" / "benchmark_runtime.py").resolve()),
        "--probe-runtime",
        "--probe-mode",
        probe_mode,
        "--model",
        model_reference,
        "--models-dir",
        str(models_dir),
        "--device",
        device,
        "--probe-backend",
        backend,
        "--probe-prompt",
        prompt,
        "--probe-max-new-tokens",
        str(max_new_tokens),
        "--probe-iterations",
        str(iterations),
        "--probe-warmup-iterations",
        str(warmup_iterations),
        "--probe-prompt-token-targets",
        ",".join(str(value) for value in prompt_token_targets),
        "--probe-output-token-targets",
        ",".join(str(value) for value in output_token_targets),
        "--probe-session-turns",
        str(session_turns),
    ]
    if not use_specialization:
        command.append("--probe-no-specialization")
    return tuple(command)


def build_current_supported_family_targets(
    models_dir: Path,
) -> tuple[RuntimeComparisonTarget, ...]:
    client = RuntimeClient()
    models_root = models_dir.expanduser().resolve()
    targets_by_family: dict[str, RuntimeComparisonTarget] = {}
    for entry in list_model_catalog():
        resolved_model = client.resolve(entry.model_id, models_root)
        if resolved_model.native_family is None:
            continue
        family_name = resolved_model.native_family.value
        model_path = resolved_model.model_path
        candidate = RuntimeComparisonTarget(
            family=family_name,
            model_reference=entry.model_id,
            is_materialized=model_path is not None and model_path.exists(),
            model_path=None if model_path is None else str(model_path),
        )
        existing = targets_by_family.get(family_name)
        if existing is None or (
            candidate.is_materialized and not existing.is_materialized
        ):
            targets_by_family[family_name] = candidate
    return tuple(
        targets_by_family[family_name] for family_name in sorted(targets_by_family)
    )


def benchmark_runtime_target(
    *,
    repo_root: Path,
    target: RuntimeComparisonTarget,
    models_dir: Path,
    device: str,
    iterations: int,
    warmup_iterations: int,
    include_extended_scenarios: bool = False,
    prompt_token_targets: tuple[int, ...] = DEFAULT_PROMPT_TOKEN_TARGETS,
    output_token_targets: tuple[int, ...] = DEFAULT_OUTPUT_TOKEN_TARGETS,
    session_turns: int = DEFAULT_SESSION_TURNS,
) -> dict[str, object]:
    client = RuntimeClient()
    if not target.is_materialized:
        unavailable = unavailable_measurement(
            name=f"{target.family}-runtime",
            details={
                "family": target.family,
                "model_reference": target.model_reference,
                "model_path": target.model_path,
                "reason": "model is not materialized locally",
            },
        )
        payload: dict[str, object] = {
            "family": target.family,
            "model_reference": target.model_reference,
            "materialized": target.is_materialized,
            "model_path": target.model_path,
            "cold_start": backend_pair_payload(unavailable, unavailable),
            "warm_runtime": backend_pair_payload(unavailable, unavailable),
        }
        if include_extended_scenarios:
            payload["prompt_length_scaling"] = {
                "generic": unavailable.to_dict(),
                "optimized_native": unavailable.to_dict(),
            }
            payload["output_length_scaling"] = {
                "generic": unavailable.to_dict(),
                "optimized_native": unavailable.to_dict(),
            }
            payload["session_growth"] = {
                "generic": unavailable.to_dict(),
                "optimized_native": unavailable.to_dict(),
            }
        return payload
    cold_generic = measure_runtime_probe(
        CommandBenchmarkSpec(
            name=f"{target.family}-generic-cold-start",
            command=build_runtime_probe_command(
                repo_root,
                target.model_reference,
                models_dir=models_dir,
                device=device,
                backend="transformers-generic",
                use_specialization=False,
                probe_mode="cold",
                iterations=max(1, iterations),
            ),
            timeout_seconds=240.0,
        ),
        iterations=max(1, iterations),
        warmup_iterations=0,
        cwd=repo_root,
    )
    cold_optimized = measure_runtime_probe(
        CommandBenchmarkSpec(
            name=f"{target.family}-optimized-cold-start",
            command=build_runtime_probe_command(
                repo_root,
                target.model_reference,
                models_dir=models_dir,
                device=device,
                backend="optimized-native",
                use_specialization=True,
                probe_mode="cold",
                iterations=max(1, iterations),
            ),
            timeout_seconds=240.0,
        ),
        iterations=max(1, iterations),
        warmup_iterations=0,
        cwd=repo_root,
    )
    warm_generic = measure_warm_runtime_probe(
        CommandBenchmarkSpec(
            name=f"{target.family}-generic-warm-runtime",
            command=build_runtime_probe_command(
                repo_root,
                target.model_reference,
                models_dir=models_dir,
                device=device,
                backend="transformers-generic",
                use_specialization=False,
                probe_mode="warm",
                iterations=max(1, iterations),
                warmup_iterations=warmup_iterations,
            ),
            timeout_seconds=240.0,
        ),
        cwd=repo_root,
    )
    warm_optimized = measure_warm_runtime_probe(
        CommandBenchmarkSpec(
            name=f"{target.family}-optimized-warm-runtime",
            command=build_runtime_probe_command(
                repo_root,
                target.model_reference,
                models_dir=models_dir,
                device=device,
                backend="optimized-native",
                use_specialization=True,
                probe_mode="warm",
                iterations=max(1, iterations),
                warmup_iterations=warmup_iterations,
            ),
            timeout_seconds=240.0,
        ),
        cwd=repo_root,
    )
    refreshed_target = refresh_runtime_target(
        client=client,
        target=target,
        models_dir=models_dir,
    )
    payload: dict[str, object] = {
        "family": target.family,
        "model_reference": target.model_reference,
        "materialized": refreshed_target.is_materialized,
        "model_path": refreshed_target.model_path,
        "cold_start": backend_pair_payload(cold_generic, cold_optimized),
        "warm_runtime": backend_pair_payload(warm_generic, warm_optimized),
    }
    if include_extended_scenarios:
        payload["prompt_length_scaling"] = {
            "generic": measure_prompt_scaling_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-generic-prompt-scaling",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="transformers-generic",
                        use_specialization=False,
                        probe_mode="prompt-scaling",
                        max_new_tokens=min(output_token_targets),
                        prompt_token_targets=prompt_token_targets,
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
            "optimized_native": measure_prompt_scaling_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-optimized-prompt-scaling",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="optimized-native",
                        use_specialization=True,
                        probe_mode="prompt-scaling",
                        max_new_tokens=min(output_token_targets),
                        prompt_token_targets=prompt_token_targets,
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
        }
        payload["output_length_scaling"] = {
            "generic": measure_output_scaling_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-generic-output-scaling",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="transformers-generic",
                        use_specialization=False,
                        probe_mode="output-scaling",
                        prompt="Explain KV cache in one sentence.",
                        output_token_targets=output_token_targets,
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
            "optimized_native": measure_output_scaling_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-optimized-output-scaling",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="optimized-native",
                        use_specialization=True,
                        probe_mode="output-scaling",
                        prompt="Explain KV cache in one sentence.",
                        output_token_targets=output_token_targets,
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
        }
        payload["session_growth"] = {
            "generic": measure_session_growth_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-generic-session-growth",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="transformers-generic",
                        use_specialization=False,
                        probe_mode="session-growth",
                        session_turns=session_turns,
                        max_new_tokens=min(output_token_targets),
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
            "optimized_native": measure_session_growth_probe(
                CommandBenchmarkSpec(
                    name=f"{target.family}-optimized-session-growth",
                    command=build_runtime_probe_command(
                        repo_root,
                        target.model_reference,
                        models_dir=models_dir,
                        device=device,
                        backend="optimized-native",
                        use_specialization=True,
                        probe_mode="session-growth",
                        session_turns=session_turns,
                        max_new_tokens=min(output_token_targets),
                    ),
                    timeout_seconds=300.0,
                ),
                cwd=repo_root,
            ).to_dict(),
        }
    return payload


def refresh_runtime_target(
    *,
    client: RuntimeClient,
    target: RuntimeComparisonTarget,
    models_dir: Path,
) -> RuntimeComparisonTarget:
    resolved_model = client.resolve(target.model_reference, models_dir)
    model_path = resolved_model.model_path
    return RuntimeComparisonTarget(
        family=target.family,
        model_reference=target.model_reference,
        is_materialized=model_path is not None and model_path.exists(),
        model_path=None if model_path is None else str(model_path),
    )

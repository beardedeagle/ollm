import argparse
import json
import sys
from pathlib import Path
from typing import cast

from ollm.runtime.benchmark_history import record_benchmark_history
from ollm.runtime.benchmark_metadata import (
    probe_comparison_key,
    report_comparison_key,
    resolve_history_codebase_label,
)
from ollm.runtime.benchmark_types import (
    DEFAULT_RUNTIME_BENCHMARK_PROFILE,
    KNOWN_RUNTIME_BENCHMARK_PROFILES,
    resolve_runtime_benchmark_profile,
)
from ollm.runtime.benchmarks import (
    build_runtime_benchmark_report,
    choose_default_device,
    render_output_scaling_probe_json,
    render_prompt_scaling_probe_json,
    render_reopen_session_growth_probe_json,
    render_report_json,
    render_runtime_probe_json,
    render_session_growth_probe_json,
    render_warm_runtime_probe_json,
    run_output_scaling_probe,
    run_prompt_scaling_probe,
    run_reopen_session_growth_probe,
    run_runtime_probe,
    run_session_growth_probe,
    run_warm_runtime_probe,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the oLLM runtime benchmark and perf-report suite."
    )
    parser.add_argument(
        "--profile",
        choices=KNOWN_RUNTIME_BENCHMARK_PROFILES,
        default=DEFAULT_RUNTIME_BENCHMARK_PROFILE,
        help="Benchmark profile: quick or full.",
    )
    parser.add_argument(
        "--model-reference",
        default="llama3-1B-chat",
        help="Model reference used for runtime comparison.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Models directory for resolver/runtime benchmarks.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Runtime device for comparisons. Defaults to the best local device.",
    )
    parser.add_argument(
        "--kv-cache-strategy",
        default="chunked",
        help="Disk KV strategy for optimized-native probes and comparisons.",
    )
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=None,
        help="Measured iterations per benchmark.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=non_negative_int,
        default=None,
        help="Warmup iterations per benchmark.",
    )
    parser.add_argument(
        "--output", default=None, help="Optional path to write the JSON report."
    )
    parser.add_argument(
        "--history-dir",
        default=None,
        help="Optional directory for persistent benchmark history records.",
    )
    parser.add_argument(
        "--history-codebase-label",
        default=None,
        help=(
            "Optional stable label for benchmark-history matching. "
            "Defaults to the normalized git origin remote."
        ),
    )
    parser.add_argument(
        "--no-record-history",
        action="store_true",
        help="Disable persistent benchmark history recording for this run.",
    )
    parser.add_argument(
        "--prompt-scale-tokens",
        default=None,
        help="Comma-separated prompt-token targets for the primary target scaling sweep.",
    )
    parser.add_argument(
        "--output-scale-tokens",
        default=None,
        help="Comma-separated output-token targets for the primary target scaling sweep.",
    )
    parser.add_argument(
        "--session-turns",
        type=positive_int,
        default=None,
        help="Measured turns for the primary target repeated-session growth sweep.",
    )
    parser.add_argument(
        "--probe-runtime",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--probe-mode", default="cold", help=argparse.SUPPRESS)
    parser.add_argument("--model", dest="probe_model", help=argparse.SUPPRESS)
    parser.add_argument("--probe-backend", help=argparse.SUPPRESS)
    parser.add_argument(
        "--probe-kv-cache-strategy",
        default="chunked",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--probe-prompt", help=argparse.SUPPRESS, default="Say hi.")
    parser.add_argument(
        "--probe-max-new-tokens",
        type=positive_int,
        default=4,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-no-specialization",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-iterations",
        type=positive_int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-warmup-iterations",
        type=non_negative_int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-prompt-token-targets",
        default="32,128,512",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-output-token-targets",
        default="16,64,128",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-session-turns",
        type=positive_int,
        default=4,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def parse_positive_int_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        raise SystemExit("expected a comma-separated list of positive integers")
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit(
            "expected a comma-separated list of positive integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise SystemExit("expected a comma-separated list of positive integers")
    return values


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    history_codebase_label = resolve_history_codebase_label(
        repo_root, override_label=args.history_codebase_label
    )
    prompt_scale_tokens = (
        None
        if args.prompt_scale_tokens is None
        else parse_positive_int_list(args.prompt_scale_tokens)
    )
    output_scale_tokens = (
        None
        if args.output_scale_tokens is None
        else parse_positive_int_list(args.output_scale_tokens)
    )
    profile = resolve_runtime_benchmark_profile(
        profile=args.profile,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        prompt_token_targets=prompt_scale_tokens,
        output_token_targets=output_scale_tokens,
        session_turns=args.session_turns,
    )
    if args.probe_runtime:
        if args.probe_model is None:
            raise SystemExit("--probe-runtime requires --model")
        if args.probe_backend is None:
            raise SystemExit("--probe-runtime requires --probe-backend")
        prompt_token_targets = parse_positive_int_list(args.probe_prompt_token_targets)
        output_token_targets = parse_positive_int_list(args.probe_output_token_targets)
        probe_device = choose_default_device() if args.device is None else args.device
        if args.probe_mode == "cold":
            probe = run_runtime_probe(
                model_reference=args.probe_model,
                models_dir=Path(args.models_dir),
                device=probe_device,
                backend=args.probe_backend,
                use_specialization=not args.probe_no_specialization,
                kv_cache_strategy=args.probe_kv_cache_strategy,
                prompt=args.probe_prompt,
                max_new_tokens=args.probe_max_new_tokens,
            )
            rendered_probe = render_runtime_probe_json(probe)
        elif args.probe_mode == "warm":
            probe = run_warm_runtime_probe(
                model_reference=args.probe_model,
                models_dir=Path(args.models_dir),
                device=probe_device,
                backend=args.probe_backend,
                use_specialization=not args.probe_no_specialization,
                kv_cache_strategy=args.probe_kv_cache_strategy,
                prompt=args.probe_prompt,
                max_new_tokens=args.probe_max_new_tokens,
                iterations=args.probe_iterations,
                warmup_iterations=args.probe_warmup_iterations,
            )
            rendered_probe = render_warm_runtime_probe_json(probe)
        elif args.probe_mode == "prompt-scaling":
            probe = run_prompt_scaling_probe(
                model_reference=args.probe_model,
                models_dir=Path(args.models_dir),
                device=probe_device,
                backend=args.probe_backend,
                use_specialization=not args.probe_no_specialization,
                kv_cache_strategy=args.probe_kv_cache_strategy,
                prompt_token_targets=prompt_token_targets,
                max_new_tokens=args.probe_max_new_tokens,
            )
            rendered_probe = render_prompt_scaling_probe_json(probe)
        elif args.probe_mode == "output-scaling":
            probe = run_output_scaling_probe(
                model_reference=args.probe_model,
                models_dir=Path(args.models_dir),
                device=probe_device,
                backend=args.probe_backend,
                use_specialization=not args.probe_no_specialization,
                kv_cache_strategy=args.probe_kv_cache_strategy,
                prompt=args.probe_prompt,
                output_token_targets=output_token_targets,
            )
            rendered_probe = render_output_scaling_probe_json(probe)
        elif args.probe_mode == "session-growth":
            probe = run_session_growth_probe(
                model_reference=args.probe_model,
                models_dir=Path(args.models_dir),
                device=probe_device,
                backend=args.probe_backend,
                use_specialization=not args.probe_no_specialization,
                kv_cache_strategy=args.probe_kv_cache_strategy,
                session_turns=args.probe_session_turns,
                max_new_tokens=args.probe_max_new_tokens,
            )
            rendered_probe = render_session_growth_probe_json(probe)
        elif args.probe_mode == "reopen-session-growth":
            probe = run_reopen_session_growth_probe(
                model_reference=args.probe_model,
                models_dir=Path(args.models_dir),
                device=probe_device,
                backend=args.probe_backend,
                use_specialization=not args.probe_no_specialization,
                kv_cache_strategy=args.probe_kv_cache_strategy,
                session_turns=args.probe_session_turns,
                max_new_tokens=args.probe_max_new_tokens,
            )
            rendered_probe = render_reopen_session_growth_probe_json(probe)
        else:
            raise SystemExit(f"Unsupported --probe-mode: {args.probe_mode}")
        if not args.no_record_history:
            payload = json.loads(rendered_probe)
            history_result = record_benchmark_history(
                repo_root=repo_root,
                payload=payload,
                run_kind=f"probe-{args.probe_mode}",
                history_dir=(
                    None if args.history_dir is None else Path(args.history_dir)
                ),
                codebase_label=history_codebase_label,
                comparison_key=probe_comparison_key(
                    codebase_label=history_codebase_label,
                    model_reference=args.probe_model,
                    device=probe_device,
                    backend=args.probe_backend,
                    kv_cache_strategy=args.probe_kv_cache_strategy,
                    probe_mode=args.probe_mode,
                    prompt=args.probe_prompt,
                    max_new_tokens=args.probe_max_new_tokens,
                    iterations=args.probe_iterations,
                    warmup_iterations=args.probe_warmup_iterations,
                    prompt_token_targets=prompt_token_targets,
                    output_token_targets=output_token_targets,
                    session_turns=args.probe_session_turns,
                ),
            )
            _emit_history_status(history_result)
        sys.stdout.write(rendered_probe)
        sys.stdout.write("\n")
        return 0

    device = choose_default_device() if args.device is None else args.device
    report = build_runtime_benchmark_report(
        repo_root=repo_root,
        benchmark_model_reference=args.model_reference,
        models_dir=Path(args.models_dir),
        device=device,
        kv_cache_strategy=args.kv_cache_strategy,
        iterations=profile.iterations,
        warmup_iterations=profile.warmup_iterations,
        prompt_token_targets=profile.prompt_token_targets,
        output_token_targets=profile.output_token_targets,
        session_turns=profile.session_turns,
        include_family_results=profile.include_family_results,
        include_primary_extended_scenarios=profile.include_primary_extended_scenarios,
        cold_timeout_seconds=profile.cold_timeout_seconds,
        warm_timeout_seconds=profile.warm_timeout_seconds,
        scaling_timeout_seconds=profile.scaling_timeout_seconds,
        session_timeout_seconds=profile.session_timeout_seconds,
    )
    rendered = render_report_json(report)
    if not args.no_record_history:
        history_result = record_benchmark_history(
            repo_root=repo_root,
            payload=json.loads(rendered),
            run_kind="report",
            history_dir=None if args.history_dir is None else Path(args.history_dir),
            codebase_label=history_codebase_label,
            comparison_key=report_comparison_key(
                codebase_label=history_codebase_label,
                benchmark_model_reference=args.model_reference,
                device=device,
                kv_cache_strategy=args.kv_cache_strategy,
                profile_id=profile.profile_id,
                prompt_token_targets=profile.prompt_token_targets,
                output_token_targets=profile.output_token_targets,
                session_turns=profile.session_turns,
            ),
        )
        _emit_history_status(history_result)
    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")
    sys.stdout.write(rendered)
    sys.stdout.write("\n")
    return 0


def _emit_history_status(history_result: dict[str, object]) -> None:
    record_path = history_result.get("record_path")
    if isinstance(record_path, str):
        codebase = history_result.get("codebase_label")
        if isinstance(codebase, str):
            print(
                f"benchmark history recorded [{codebase}]: {record_path}",
                file=sys.stderr,
            )
        else:
            print(f"benchmark history recorded: {record_path}", file=sys.stderr)
    comparison = history_result.get("comparison_to_previous")
    if not isinstance(comparison, dict):
        return
    comparison_payload = cast(dict[str, object], comparison)
    regressions = comparison_payload.get("potential_regressions")
    if not isinstance(regressions, list) or not regressions:
        return
    print(
        "potential benchmark regressions: "
        + ", ".join(str(item) for item in regressions),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())

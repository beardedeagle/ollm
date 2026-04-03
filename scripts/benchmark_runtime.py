import argparse
import json
import sys
from pathlib import Path

from benchmark_runtime_support import (
    emit_history_status,
    extract_probe_selector_result,
    non_negative_int,
    parse_positive_int_list,
    positive_int,
    resolve_report_selector_result,
)
from ollm.runtime.benchmark import (
    build_runtime_benchmark_report,
    choose_default_device,
    render_report_json,
)
from ollm.runtime.benchmark.history import record_benchmark_history
from ollm.runtime.benchmark.metadata import (
    probe_comparison_key,
    report_comparison_key,
    resolve_history_codebase_label,
)
from ollm.runtime.benchmark.probe_registry import (
    ProbeInvocation,
    ProbeMode,
    get_probe_definition,
    probe_mode_choices,
)
from ollm.runtime.benchmark.types import (
    DEFAULT_RUNTIME_BENCHMARK_PROFILE,
    KNOWN_RUNTIME_BENCHMARK_PROFILES,
    resolve_runtime_benchmark_profile,
)
from ollm.runtime.strategy_selector import DEFAULT_STRATEGY_SELECTOR_PROFILE


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
        default=None,
        help="Explicit KV cache strategy override for probes and comparisons.",
    )
    parser.add_argument(
        "--strategy-selector-profile",
        default=DEFAULT_STRATEGY_SELECTOR_PROFILE,
        help="Runtime strategy selector profile for probes and comparisons.",
    )
    parser.add_argument(
        "--kv-cache-window-tokens",
        type=positive_int,
        default=None,
        help="Sliding-window token budget for bounded KV strategies.",
    )
    parser.add_argument(
        "--offload-cpu-layers",
        type=non_negative_int,
        default=0,
        help="CPU offload layer budget for runtime probes and comparisons.",
    )
    parser.add_argument(
        "--offload-cpu-policy",
        default="auto",
        help="CPU offload placement policy.",
    )
    parser.add_argument(
        "--offload-gpu-layers",
        type=non_negative_int,
        default=0,
        help="GPU offload layer budget for runtime probes and comparisons.",
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
        "--session-max-new-tokens",
        type=positive_int,
        default=None,
        help=(
            "Measured max new tokens per turn for the repeated-session growth "
            "sweep. This is independent from --output-scale-tokens."
        ),
    )
    parser.add_argument("--probe-runtime", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--probe-mode",
        choices=probe_mode_choices(),
        default=ProbeMode.COLD.value,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--model", dest="probe_model", help=argparse.SUPPRESS)
    parser.add_argument("--probe-backend", help=argparse.SUPPRESS)
    parser.add_argument(
        "--probe-kv-cache-strategy",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-strategy-selector-profile",
        default=DEFAULT_STRATEGY_SELECTOR_PROFILE,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-kv-cache-window-tokens",
        type=positive_int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-offload-cpu-layers",
        type=non_negative_int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-offload-cpu-policy",
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-offload-gpu-layers",
        type=non_negative_int,
        default=0,
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
        session_max_new_tokens=args.session_max_new_tokens,
    )
    if args.probe_runtime:
        if args.probe_model is None:
            raise SystemExit("--probe-runtime requires --model")
        if args.probe_backend is None:
            raise SystemExit("--probe-runtime requires --probe-backend")
        probe_mode = ProbeMode(args.probe_mode)
        prompt_token_targets = parse_positive_int_list(args.probe_prompt_token_targets)
        output_token_targets = parse_positive_int_list(args.probe_output_token_targets)
        probe_device = choose_default_device() if args.device is None else args.device
        invocation = ProbeInvocation(
            model_reference=args.probe_model,
            models_dir=Path(args.models_dir),
            device=probe_device,
            backend=args.probe_backend,
            use_specialization=not args.probe_no_specialization,
            kv_cache_strategy=args.probe_kv_cache_strategy,
            strategy_selector_profile=args.probe_strategy_selector_profile,
            kv_cache_window_tokens=args.probe_kv_cache_window_tokens,
            offload_cpu_layers=args.probe_offload_cpu_layers,
            offload_cpu_policy=args.probe_offload_cpu_policy,
            offload_gpu_layers=args.probe_offload_gpu_layers,
            prompt=args.probe_prompt,
            max_new_tokens=args.probe_max_new_tokens,
            iterations=args.probe_iterations,
            warmup_iterations=args.probe_warmup_iterations,
            prompt_token_targets=prompt_token_targets,
            output_token_targets=output_token_targets,
            session_turns=args.probe_session_turns,
        )
        probe_definition = get_probe_definition(probe_mode)
        probe = probe_definition.runner(invocation)
        rendered_probe = probe_definition.renderer(probe)
        if not args.no_record_history:
            payload = json.loads(rendered_probe)
            selector_rule_id, selector_applied_strategy = extract_probe_selector_result(
                payload,
                probe_mode=probe_mode,
            )
            history_result = record_benchmark_history(
                repo_root=repo_root,
                payload=payload,
                run_kind=probe_definition.run_kind,
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
                    strategy_selector_profile=args.probe_strategy_selector_profile,
                    strategy_selector_rule_id=selector_rule_id,
                    strategy_selector_applied_kv_cache_strategy=(
                        selector_applied_strategy
                    ),
                    kv_cache_window_tokens=args.probe_kv_cache_window_tokens,
                    offload_cpu_layers=args.probe_offload_cpu_layers,
                    offload_cpu_policy=args.probe_offload_cpu_policy,
                    offload_gpu_layers=args.probe_offload_gpu_layers,
                    probe_mode=probe_mode.value,
                    prompt=args.probe_prompt,
                    max_new_tokens=args.probe_max_new_tokens,
                    iterations=args.probe_iterations,
                    warmup_iterations=args.probe_warmup_iterations,
                    prompt_token_targets=prompt_token_targets,
                    output_token_targets=output_token_targets,
                    session_turns=args.probe_session_turns,
                ),
            )
            emit_history_status(history_result)
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
        strategy_selector_profile=args.strategy_selector_profile,
        kv_cache_window_tokens=args.kv_cache_window_tokens,
        offload_cpu_layers=args.offload_cpu_layers,
        offload_cpu_policy=args.offload_cpu_policy,
        offload_gpu_layers=args.offload_gpu_layers,
        iterations=profile.iterations,
        warmup_iterations=profile.warmup_iterations,
        prompt_token_targets=profile.prompt_token_targets,
        output_token_targets=profile.output_token_targets,
        session_turns=profile.session_turns,
        session_max_new_tokens=profile.session_max_new_tokens,
        include_family_results=profile.include_family_results,
        include_primary_extended_scenarios=profile.include_primary_extended_scenarios,
        cold_timeout_seconds=profile.cold_timeout_seconds,
        warm_timeout_seconds=profile.warm_timeout_seconds,
        scaling_timeout_seconds=profile.scaling_timeout_seconds,
        session_timeout_seconds=profile.session_timeout_seconds,
    )
    rendered = render_report_json(report)
    if not args.no_record_history:
        selector_rule_id, selector_applied_strategy = resolve_report_selector_result(
            model_reference=args.model_reference,
            models_dir=Path(args.models_dir),
            device=device,
            kv_cache_strategy=args.kv_cache_strategy,
            strategy_selector_profile=args.strategy_selector_profile,
            kv_cache_window_tokens=args.kv_cache_window_tokens,
            offload_cpu_layers=args.offload_cpu_layers,
            offload_cpu_policy=args.offload_cpu_policy,
            offload_gpu_layers=args.offload_gpu_layers,
        )
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
                strategy_selector_profile=args.strategy_selector_profile,
                strategy_selector_rule_id=selector_rule_id,
                strategy_selector_applied_kv_cache_strategy=selector_applied_strategy,
                kv_cache_window_tokens=args.kv_cache_window_tokens,
                offload_cpu_layers=args.offload_cpu_layers,
                offload_cpu_policy=args.offload_cpu_policy,
                offload_gpu_layers=args.offload_gpu_layers,
                profile_id=profile.profile_id,
                prompt_token_targets=profile.prompt_token_targets,
                output_token_targets=profile.output_token_targets,
                session_turns=profile.session_turns,
                session_max_new_tokens=profile.session_max_new_tokens,
            ),
        )
        emit_history_status(history_result)
    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")
    sys.stdout.write(rendered)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

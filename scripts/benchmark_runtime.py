import argparse
from pathlib import Path
import sys

from ollm.runtime.benchmarks import (
    build_runtime_benchmark_report,
    choose_default_device,
    render_report_json,
    render_runtime_probe_json,
    run_runtime_probe,
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
        "--iterations",
        type=positive_int,
        default=5,
        help="Measured iterations per benchmark.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=non_negative_int,
        default=1,
        help="Warmup iterations per benchmark.",
    )
    parser.add_argument(
        "--output", default=None, help="Optional path to write the JSON report."
    )
    parser.add_argument(
        "--probe-runtime",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--model", dest="probe_model", help=argparse.SUPPRESS)
    parser.add_argument("--probe-backend", help=argparse.SUPPRESS)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.probe_runtime:
        if args.probe_model is None:
            raise SystemExit("--probe-runtime requires --model")
        if args.probe_backend is None:
            raise SystemExit("--probe-runtime requires --probe-backend")
        probe = run_runtime_probe(
            model_reference=args.probe_model,
            models_dir=Path(args.models_dir),
            device=choose_default_device() if args.device is None else args.device,
            backend=args.probe_backend,
            use_specialization=not args.probe_no_specialization,
            prompt=args.probe_prompt,
            max_new_tokens=args.probe_max_new_tokens,
        )
        sys.stdout.write(render_runtime_probe_json(probe))
        sys.stdout.write("\n")
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    device = choose_default_device() if args.device is None else args.device
    report = build_runtime_benchmark_report(
        repo_root=repo_root,
        benchmark_model_reference=args.model_reference,
        models_dir=Path(args.models_dir),
        device=device,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
    )
    rendered = render_report_json(report)
    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")
    sys.stdout.write(rendered)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

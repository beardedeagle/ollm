"""Run a reusable runtime smoke validation against a real oLLM runtime."""

import argparse
import sys
from pathlib import Path

from ollm.app.runtime_smoke import (
    DEFAULT_RUNTIME_SMOKE_CHAT_TURNS,
    DEFAULT_RUNTIME_SMOKE_EXPECTATIONS,
    DEFAULT_RUNTIME_SMOKE_PROMPT,
    DEFAULT_RUNTIME_SMOKE_SYSTEM_PROMPT,
    render_runtime_smoke_report_json,
    run_runtime_smoke,
)
from ollm.app.service import build_default_application_service
from ollm.async_io import path_mkdir, path_write_text
from ollm.cli.common import build_generation_config, build_runtime_config
from ollm.runtime.settings import load_app_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reusable one-shot and chat-session runtime smoke validation."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model reference to resolve.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing model data.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Backend override.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Optional PEFT adapter directory.",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Enable multimodal processor support.",
    )
    parser.add_argument(
        "--no-specialization",
        action="store_true",
        help="Disable optimized specialization selection.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="KV cache directory.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable KV cache.",
    )
    parser.add_argument(
        "--kv-cache-strategy",
        default=None,
        help="Explicit KV cache strategy override.",
    )
    parser.add_argument(
        "--strategy-selector-profile",
        default=None,
        help="Selector profile: balanced, latency, capacity, or bounded-window.",
    )
    parser.add_argument(
        "--kv-cache-window-tokens",
        type=int,
        default=None,
        help="Sliding-window token budget for bounded KV strategies.",
    )
    parser.add_argument(
        "--offload-cpu-layers",
        type=int,
        default=None,
        help="Number of layers to offload to CPU.",
    )
    parser.add_argument(
        "--offload-cpu-policy",
        default=None,
        help="CPU offload placement policy.",
    )
    parser.add_argument(
        "--offload-gpu-layers",
        type=int,
        default=None,
        help="Number of layers to keep on GPU when using mixed offload.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force redownload of the selected model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum generated tokens for the smoke prompts.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the smoke prompts.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling cutoff.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the smoke prompts.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_RUNTIME_SMOKE_SYSTEM_PROMPT,
        help="System prompt used for validation.",
    )
    parser.add_argument(
        "--prompt-text",
        default=DEFAULT_RUNTIME_SMOKE_PROMPT,
        help="Prompt used for the one-shot validation.",
    )
    parser.add_argument(
        "--chat-turn",
        action="append",
        default=[],
        help="Chat turn prompt. Provide more than once for multi-turn validation.",
    )
    parser.add_argument(
        "--expect-contains",
        action="append",
        default=[],
        help="Substring that must appear in each smoke response.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path for the JSON smoke report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_app_settings()
    runtime_config = build_runtime_config(
        model=args.model,
        models_dir=args.models_dir,
        device=args.device,
        backend=args.backend,
        adapter_dir=args.adapter_dir,
        multimodal=True if args.multimodal else None,
        no_specialization=True if args.no_specialization else None,
        cache_dir=args.cache_dir,
        no_cache=True if args.no_cache else None,
        kv_cache_strategy=args.kv_cache_strategy,
        strategy_selector_profile=args.strategy_selector_profile,
        kv_cache_window_tokens=args.kv_cache_window_tokens,
        offload_cpu_layers=args.offload_cpu_layers,
        offload_cpu_policy=args.offload_cpu_policy,
        offload_gpu_layers=args.offload_gpu_layers,
        force_download=True if args.force_download else None,
        stats=None,
        verbose=None,
        quiet=None,
        settings=settings,
    )
    generation_config = build_generation_config(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        stream=False,
        settings=settings,
    )
    chat_turns = (
        DEFAULT_RUNTIME_SMOKE_CHAT_TURNS
        if not args.chat_turn
        else tuple(args.chat_turn)
    )
    expected_contains = (
        DEFAULT_RUNTIME_SMOKE_EXPECTATIONS
        if not args.expect_contains
        else tuple(args.expect_contains)
    )
    report = run_runtime_smoke(
        service=build_default_application_service(),
        runtime_config=runtime_config,
        generation_config=generation_config,
        prompt_text=args.prompt_text,
        chat_turns=chat_turns,
        system_prompt=args.system_prompt,
        expected_contains=expected_contains,
    )
    rendered_report = render_runtime_smoke_report_json(report)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        path_mkdir(output_path.parent, parents=True, exist_ok=True)
        path_write_text(output_path, rendered_report, encoding="utf-8")
    print(rendered_report)
    if report.ok:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

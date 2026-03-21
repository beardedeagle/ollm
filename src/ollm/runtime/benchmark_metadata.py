"""Benchmark metadata and comparison-key helpers."""

import hashlib
import platform
from pathlib import Path
from subprocess import CompletedProcess, run
from urllib.parse import urlparse

from ollm.runtime.strategy_selector import DEFAULT_STRATEGY_SELECTOR_PROFILE


def build_history_host_summary() -> dict[str, object]:
    """Capture host identity for benchmark history records."""

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def build_history_codebase_summary(
    repo_root: Path, *, codebase_label: str
) -> dict[str, str | None]:
    """Capture the benchmark history codebase label and source remote."""

    return {
        "label": codebase_label,
        "origin_url": _git_stdout(repo_root, "remote", "get-url", "origin"),
    }


def build_git_summary(repo_root: Path) -> dict[str, str | None]:
    """Capture the current git branch and commit when available."""

    return {
        "branch": _git_stdout(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git_stdout(repo_root, "rev-parse", "HEAD"),
    }


def resolve_history_codebase_label(
    repo_root: Path, *, override_label: str | None
) -> str:
    """Resolve the stable codebase label used for benchmark history matching."""

    if override_label is not None:
        stripped = override_label.strip()
        if not stripped:
            raise ValueError("--history-codebase-label cannot be empty")
        return stripped
    remote_url = _git_stdout(repo_root, "remote", "get-url", "origin")
    normalized = normalize_git_remote_url(remote_url)
    if normalized is not None:
        return normalized
    return repo_root.resolve().name


def probe_comparison_key(
    *,
    codebase_label: str,
    model_reference: str,
    device: str,
    backend: str,
    kv_cache_strategy: str | None,
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE,
    strategy_selector_rule_id: str | None = None,
    strategy_selector_applied_kv_cache_strategy: str | None = None,
    kv_cache_window_tokens: int | None = None,
    offload_cpu_layers: int = 0,
    offload_cpu_policy: str = "auto",
    offload_gpu_layers: int = 0,
    probe_mode: str,
    prompt: str,
    max_new_tokens: int,
    iterations: int,
    warmup_iterations: int,
    prompt_token_targets: tuple[int, ...],
    output_token_targets: tuple[int, ...],
    session_turns: int,
) -> dict[str, object]:
    """Build a stable comparison key for one probe run shape."""

    return {
        "run_kind": f"probe-{probe_mode}",
        "codebase_label": codebase_label,
        "model_reference": model_reference,
        "device": device,
        "backend": backend,
        "kv_cache_strategy": _requested_strategy_key(kv_cache_strategy),
        "strategy_selector_profile": strategy_selector_profile,
        "strategy_selector_rule_id": strategy_selector_rule_id,
        "strategy_selector_applied_kv_cache_strategy": (
            strategy_selector_applied_kv_cache_strategy
        ),
        "kv_cache_window_tokens": kv_cache_window_tokens,
        "offload_cpu_layers": offload_cpu_layers,
        "offload_cpu_policy": offload_cpu_policy,
        "offload_gpu_layers": offload_gpu_layers,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "max_new_tokens": max_new_tokens,
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "prompt_token_targets": list(prompt_token_targets),
        "output_token_targets": list(output_token_targets),
        "session_turns": session_turns,
    }


def report_comparison_key(
    *,
    codebase_label: str,
    benchmark_model_reference: str,
    device: str,
    kv_cache_strategy: str | None,
    strategy_selector_profile: str = DEFAULT_STRATEGY_SELECTOR_PROFILE,
    strategy_selector_rule_id: str | None = None,
    strategy_selector_applied_kv_cache_strategy: str | None = None,
    kv_cache_window_tokens: int | None = None,
    offload_cpu_layers: int = 0,
    offload_cpu_policy: str = "auto",
    offload_gpu_layers: int = 0,
    profile_id: str,
    prompt_token_targets: tuple[int, ...],
    output_token_targets: tuple[int, ...],
    session_turns: int,
    session_max_new_tokens: int,
) -> dict[str, object]:
    """Build a stable comparison key for one report run shape."""

    return {
        "run_kind": "report",
        "codebase_label": codebase_label,
        "benchmark_model_reference": benchmark_model_reference,
        "device": device,
        "kv_cache_strategy": _requested_strategy_key(kv_cache_strategy),
        "strategy_selector_profile": strategy_selector_profile,
        "strategy_selector_rule_id": strategy_selector_rule_id,
        "strategy_selector_applied_kv_cache_strategy": (
            strategy_selector_applied_kv_cache_strategy
        ),
        "kv_cache_window_tokens": kv_cache_window_tokens,
        "offload_cpu_layers": offload_cpu_layers,
        "offload_cpu_policy": offload_cpu_policy,
        "offload_gpu_layers": offload_gpu_layers,
        "profile_id": profile_id,
        "prompt_token_targets": list(prompt_token_targets),
        "output_token_targets": list(output_token_targets),
        "session_turns": session_turns,
        "session_max_new_tokens": session_max_new_tokens,
    }


def _git_stdout(repo_root: Path, *args: str) -> str | None:
    completed: CompletedProcess[str] = run(
        ("git", *args),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def normalize_git_remote_url(remote_url: str | None) -> str | None:
    """Normalize a git remote into a stable host/owner/repo label."""

    if remote_url is None:
        return None
    stripped = remote_url.strip()
    if not stripped:
        return None
    if stripped.startswith("git@") and ":" in stripped:
        host_and_path = stripped.split("@", maxsplit=1)[1]
        host, repo_path = host_and_path.split(":", maxsplit=1)
        normalized_path = _normalize_repo_path(repo_path)
        if normalized_path is None:
            return None
        return f"{host}/{normalized_path}"
    parsed = urlparse(stripped)
    if parsed.scheme:
        host = parsed.hostname
        normalized_path = _normalize_repo_path(parsed.path)
        if host is None or normalized_path is None:
            return None
        return f"{host}/{normalized_path}"
    normalized_path = _normalize_repo_path(stripped)
    if normalized_path is None:
        return None
    return normalized_path


def _normalize_repo_path(raw_path: str) -> str | None:
    path = raw_path.strip().strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path or None


def _requested_strategy_key(strategy: str | None) -> str:
    if strategy is None:
        return "auto"
    return strategy

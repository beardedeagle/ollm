from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from subprocess import run
from typing import cast

from ollm.runtime.benchmark_history import (
    record_benchmark_history,
    summarize_benchmark_payload,
)
from ollm.runtime.benchmark_metadata import (
    normalize_git_remote_url,
    probe_comparison_key,
    resolve_history_codebase_label,
)
from ollm.runtime.benchmark_probe_types import (
    RuntimeProbeResult,
    WarmRuntimeProbeResult,
)
from tests.benchmark_support import build_request_probe_metrics, build_stage_resources


def test_record_benchmark_history_compares_with_previous_probe(tmp_path: Path) -> None:
    history_dir = tmp_path / ".omx" / "logs" / "benchmark-history"
    comparison_key = probe_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="cpu",
        backend="optimized-native",
        kv_cache_strategy="chunked",
        probe_mode="cold",
        prompt="Say hi.",
        max_new_tokens=4,
        iterations=1,
        warmup_iterations=0,
    )
    first_payload = RuntimeProbeResult(
        load_ms=10.0,
        load_resources=build_stage_resources(),
        request=build_request_probe_metrics(),
    ).to_dict()

    first = record_benchmark_history(
        repo_root=tmp_path,
        payload=first_payload,
        run_kind="probe-cold",
        history_dir=history_dir,
        comparison_key=comparison_key,
        codebase_label="github.com/beardedeagle/ollm",
    )

    assert first["comparison_to_previous"] is None

    second_payload = deepcopy(first_payload)
    second_payload["load_ms"] = 12.0
    second_request = cast(dict[str, object], second_payload["request"])
    second_request["total_ms"] = 36.0
    resources = cast(dict[str, object], second_request["resources"])
    resources["accelerator_peak_mb"] = 340.0

    second = record_benchmark_history(
        repo_root=tmp_path,
        payload=second_payload,
        run_kind="probe-cold",
        history_dir=history_dir,
        comparison_key=comparison_key,
        codebase_label="github.com/beardedeagle/ollm",
    )

    comparison = cast(dict[str, object], second["comparison_to_previous"])
    regressions = cast(list[str], comparison["potential_regressions"])
    assert "load_ms" in regressions
    assert "request_total_ms" in regressions
    assert "accelerator_peak_mb" in regressions
    assert second["codebase_label"] == "github.com/beardedeagle/ollm"


def test_record_benchmark_history_does_not_compare_across_codebases(
    tmp_path: Path,
) -> None:
    history_dir = tmp_path / ".omx" / "logs" / "benchmark-history"
    payload = RuntimeProbeResult(
        load_ms=10.0,
        load_resources=build_stage_resources(),
        request=build_request_probe_metrics(),
    ).to_dict()
    fork_key = probe_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="cpu",
        backend="optimized-native",
        kv_cache_strategy="chunked",
        probe_mode="cold",
        prompt="Say hi.",
        max_new_tokens=4,
        iterations=1,
        warmup_iterations=0,
    )
    upstream_key = probe_comparison_key(
        codebase_label="github.com/Mega4alik/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="cpu",
        backend="optimized-native",
        kv_cache_strategy="chunked",
        probe_mode="cold",
        prompt="Say hi.",
        max_new_tokens=4,
        iterations=1,
        warmup_iterations=0,
    )

    record_benchmark_history(
        repo_root=tmp_path,
        payload=payload,
        run_kind="probe-cold",
        history_dir=history_dir,
        comparison_key=fork_key,
        codebase_label="github.com/beardedeagle/ollm",
    )
    upstream = record_benchmark_history(
        repo_root=tmp_path,
        payload=payload,
        run_kind="probe-cold",
        history_dir=history_dir,
        comparison_key=upstream_key,
        codebase_label="github.com/Mega4alik/ollm",
    )

    assert upstream["comparison_to_previous"] is None


def test_resolve_history_codebase_label_uses_normalized_origin_url(
    tmp_path: Path,
) -> None:
    run(("git", "init"), cwd=tmp_path, check=True)
    run(
        ("git", "remote", "add", "origin", "git@github.com:Mega4alik/ollm.git"),
        cwd=tmp_path,
        check=True,
    )

    label = resolve_history_codebase_label(tmp_path, override_label=None)

    assert label == "github.com/Mega4alik/ollm"


def test_normalize_git_remote_url_handles_https() -> None:
    assert (
        normalize_git_remote_url("https://github.com/Mega4alik/ollm.git")
        == "github.com/Mega4alik/ollm"
    )


def test_summarize_benchmark_payload_for_warm_probe_uses_means() -> None:
    first = build_request_probe_metrics()
    second = build_request_probe_metrics()
    warm_payload = WarmRuntimeProbeResult(
        runtime_load_ms=9.0,
        runtime_load_resources=build_stage_resources(),
        warmup_iterations=0,
        measured_iterations=(
            first,
            replace(second, total_ms=50.0),
        ),
    ).to_dict()

    summary = summarize_benchmark_payload(warm_payload, run_kind="probe-warm")

    assert summary["load_ms"] == 9.0
    assert summary["request_total_ms"] == 40.0

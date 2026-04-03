import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from subprocess import run
from typing import cast

from ollm.runtime.benchmark.history import (
    _append_jsonl_entry,
    find_previous_record,
    record_benchmark_history,
    summarize_benchmark_payload,
)
from ollm.runtime.benchmark.metadata import (
    normalize_git_remote_url,
    probe_comparison_key,
    resolve_history_codebase_label,
)
from ollm.runtime.benchmark.probe_types import (
    PromptScalingCase,
    PromptScalingProbeResult,
    ReopenSessionGrowthProbeResult,
    ReopenSessionGrowthTurn,
    RuntimeProbeResult,
    SessionGrowthProbeResult,
    SessionGrowthTurn,
    WarmRuntimeProbeResult,
)
from tests.benchmark_support import build_request_probe_metrics, build_stage_resources


def test_record_benchmark_history_compares_with_previous_probe(tmp_path: Path) -> None:
    history_dir = tmp_path / ".ollm" / "benchmark-history"
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
        prompt_token_targets=(32, 128, 512),
        output_token_targets=(16, 64, 128),
        session_turns=4,
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
    history_dir = tmp_path / ".ollm" / "benchmark-history"
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
        prompt_token_targets=(32, 128, 512),
        output_token_targets=(16, 64, 128),
        session_turns=4,
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
        prompt_token_targets=(32, 128, 512),
        output_token_targets=(16, 64, 128),
        session_turns=4,
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


def test_summarize_benchmark_payload_for_session_growth_uses_final_turn() -> None:
    first = replace(build_request_probe_metrics(), total_ms=18.0)
    cache_state = build_request_probe_metrics().cache_state
    assert cache_state is not None
    second = replace(
        build_request_probe_metrics(),
        total_ms=30.0,
        cache_state=replace(
            cache_state,
            compaction_count=2,
            persisted_artifact_count=1,
            cold_store_format="ollm-kv-journal",
        ),
    )
    payload = SessionGrowthProbeResult(
        runtime_load_ms=9.0,
        runtime_load_resources=build_stage_resources(),
        turns=(
            SessionGrowthTurn(turn_index=1, request=first),
            SessionGrowthTurn(turn_index=2, request=second),
        ),
    ).to_dict()

    summary = summarize_benchmark_payload(payload, run_kind="probe-session-growth")

    assert summary["load_ms"] == 9.0
    assert summary["request_total_ms"] == 30.0
    assert summary["mean_request_total_ms"] == 24.0
    assert summary["compaction_count"] == 2
    assert summary["persisted_artifact_count"] == 1
    assert summary["cold_store_format"] == "ollm-kv-journal"
    assert summary["cold_tier_representation"] is None
    assert summary["session_turns"] == 2


def test_summarize_benchmark_payload_for_reopen_session_growth_uses_final_turn() -> (
    None
):
    first = replace(build_request_probe_metrics(), total_ms=18.0)
    cache_state = build_request_probe_metrics().cache_state
    assert cache_state is not None
    second = replace(
        build_request_probe_metrics(),
        total_ms=30.0,
        cache_state=replace(
            cache_state,
            compaction_count=2,
            persisted_artifact_count=1,
            cold_store_format="ollm-kv-journal-quantized",
            cold_tier_representation="int8-symmetric-per-tensor",
        ),
    )
    payload = ReopenSessionGrowthProbeResult(
        turns=(
            ReopenSessionGrowthTurn(
                turn_index=1,
                runtime_load_ms=10.0,
                runtime_load_resources=build_stage_resources(),
                request=first,
            ),
            ReopenSessionGrowthTurn(
                turn_index=2,
                runtime_load_ms=14.0,
                runtime_load_resources=build_stage_resources(),
                request=second,
            ),
        ),
    ).to_dict()

    summary = summarize_benchmark_payload(
        payload, run_kind="probe-reopen-session-growth"
    )

    assert summary["load_ms"] == 12.0
    assert summary["final_runtime_load_ms"] == 14.0
    assert summary["request_total_ms"] == 30.0
    assert summary["mean_request_total_ms"] == 24.0
    assert summary["compaction_count"] == 2
    assert summary["persisted_artifact_count"] == 1
    assert summary["cold_store_format"] == "ollm-kv-journal-quantized"
    assert summary["cold_tier_representation"] == "int8-symmetric-per-tensor"
    assert summary["session_turns"] == 2


def test_summarize_benchmark_payload_for_prompt_scaling_uses_last_case() -> None:
    first = build_request_probe_metrics()
    second = replace(build_request_probe_metrics(), total_ms=42.0)
    payload = PromptScalingProbeResult(
        runtime_load_ms=11.0,
        runtime_load_resources=build_stage_resources(),
        cases=(
            PromptScalingCase(requested_prompt_tokens=32, request=first),
            PromptScalingCase(requested_prompt_tokens=128, request=second),
        ),
    ).to_dict()

    summary = summarize_benchmark_payload(payload, run_kind="probe-prompt-scaling")

    assert summary["load_ms"] == 11.0
    assert summary["request_total_ms"] == 42.0
    assert summary["max_request_total_ms"] == 42.0
    assert summary["case_count"] == 2


def test_append_jsonl_entry_appends_without_rewriting_previous_lines(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.jsonl"

    first: dict[str, object] = {"generated_at": "1", "record_path": "a.json"}
    second: dict[str, object] = {"generated_at": "2", "record_path": "b.json"}
    _append_jsonl_entry(index_path, first)
    _append_jsonl_entry(index_path, second)

    lines = index_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [first, second]


def test_record_benchmark_history_writes_latest_sidecar(tmp_path: Path) -> None:
    history_dir = tmp_path / ".ollm" / "benchmark-history"
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
        prompt_token_targets=(32, 128, 512),
        output_token_targets=(16, 64, 128),
        session_turns=4,
    )
    payload = RuntimeProbeResult(
        load_ms=10.0,
        load_resources=build_stage_resources(),
        request=build_request_probe_metrics(),
    ).to_dict()

    result = record_benchmark_history(
        repo_root=tmp_path,
        payload=payload,
        run_kind="probe-cold",
        history_dir=history_dir,
        comparison_key=comparison_key,
        codebase_label="github.com/beardedeagle/ollm",
    )

    latest_dir = history_dir / "latest"
    latest_paths = tuple(latest_dir.glob("*.json"))

    assert len(latest_paths) == 1
    latest_entry = json.loads(latest_paths[0].read_text(encoding="utf-8"))
    assert latest_entry["record_path"] == result["record_path"]
    assert latest_entry["comparison_key"] == comparison_key


def test_find_previous_record_uses_latest_sidecar_before_index_scan(
    tmp_path: Path,
) -> None:
    history_dir = tmp_path / ".ollm" / "benchmark-history"
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
        prompt_token_targets=(32, 128, 512),
        output_token_targets=(16, 64, 128),
        session_turns=4,
    )
    payload = RuntimeProbeResult(
        load_ms=10.0,
        load_resources=build_stage_resources(),
        request=build_request_probe_metrics(),
    ).to_dict()

    result = record_benchmark_history(
        repo_root=tmp_path,
        payload=payload,
        run_kind="probe-cold",
        history_dir=history_dir,
        comparison_key=comparison_key,
        codebase_label="github.com/beardedeagle/ollm",
    )
    index_path = history_dir / "index.jsonl"
    index_path.unlink()

    previous = find_previous_record(index_path, comparison_key=comparison_key)

    assert previous is not None
    assert previous["record_path"] == result["record_path"]


def test_find_previous_record_falls_back_to_index_when_sidecar_is_invalid(
    tmp_path: Path,
) -> None:
    history_dir = tmp_path / ".ollm" / "benchmark-history"
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
        prompt_token_targets=(32, 128, 512),
        output_token_targets=(16, 64, 128),
        session_turns=4,
    )
    payload = RuntimeProbeResult(
        load_ms=10.0,
        load_resources=build_stage_resources(),
        request=build_request_probe_metrics(),
    ).to_dict()

    result = record_benchmark_history(
        repo_root=tmp_path,
        payload=payload,
        run_kind="probe-cold",
        history_dir=history_dir,
        comparison_key=comparison_key,
        codebase_label="github.com/beardedeagle/ollm",
    )
    latest_path = next((history_dir / "latest").glob("*.json"))
    latest_path.write_text("{invalid json\n", encoding="utf-8")

    previous = find_previous_record(history_dir / "index.jsonl", comparison_key=comparison_key)

    assert previous is not None
    assert previous["record_path"] == result["record_path"]

from dataclasses import replace

from ollm.runtime.benchmark_history import summarize_benchmark_payload
from ollm.runtime.benchmark_metadata import probe_comparison_key
from ollm.runtime.benchmark_probe_types import (
    ReopenSessionGrowthProbeResult,
    ReopenSessionGrowthTurn,
    SessionGrowthProbeResult,
    SessionGrowthTurn,
)
from tests.benchmark_support import build_request_probe_metrics, build_stage_resources


def test_probe_comparison_key_distinguishes_sliding_window_token_limits() -> None:
    first = probe_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="cpu",
        backend="optimized-native",
        kv_cache_strategy="sliding-window-ring-buffer",
        kv_cache_window_tokens=32,
        probe_mode="session-growth",
        prompt="Say hi.",
        max_new_tokens=16,
        iterations=1,
        warmup_iterations=0,
        prompt_token_targets=(32,),
        output_token_targets=(16,),
        session_turns=4,
    )
    second = probe_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="cpu",
        backend="optimized-native",
        kv_cache_strategy="sliding-window-ring-buffer",
        kv_cache_window_tokens=64,
        probe_mode="session-growth",
        prompt="Say hi.",
        max_new_tokens=16,
        iterations=1,
        warmup_iterations=0,
        prompt_token_targets=(32,),
        output_token_targets=(16,),
        session_turns=4,
    )

    assert first != second


def test_summarize_benchmark_payload_for_session_growth_keeps_window_and_eviction_fields() -> (
    None
):
    cache_state = build_request_probe_metrics().cache_state
    assert cache_state is not None
    payload = SessionGrowthProbeResult(
        runtime_load_ms=9.0,
        runtime_load_resources=build_stage_resources(),
        turns=(
            SessionGrowthTurn(
                turn_index=1,
                request=replace(
                    build_request_probe_metrics(),
                    cache_state=replace(
                        cache_state,
                        strategy_id="sliding-window-ring-buffer",
                        persistence_format="sliding-window-ring-buffer",
                        window_policy="sliding-window",
                        window_max_tokens=32,
                        eviction_policy="drop-oldest",
                        eviction_count=2,
                        evicted_tokens=16,
                        cold_store_format="ollm-kv-sliding-window",
                    ),
                ),
            ),
        ),
    ).to_dict()

    summary = summarize_benchmark_payload(payload, run_kind="probe-session-growth")

    assert summary["window_max_tokens"] == 32
    assert summary["eviction_policy"] == "drop-oldest"
    assert summary["eviction_count"] == 2
    assert summary["evicted_tokens"] == 16


def test_summarize_benchmark_payload_for_reopen_session_growth_keeps_window_and_eviction_fields() -> (
    None
):
    cache_state = build_request_probe_metrics().cache_state
    assert cache_state is not None
    payload = ReopenSessionGrowthProbeResult(
        turns=(
            ReopenSessionGrowthTurn(
                turn_index=1,
                runtime_load_ms=10.0,
                runtime_load_resources=build_stage_resources(),
                request=replace(
                    build_request_probe_metrics(),
                    cache_state=replace(
                        cache_state,
                        strategy_id="sliding-window-ring-buffer",
                        persistence_format="sliding-window-ring-buffer",
                        window_policy="sliding-window",
                        window_max_tokens=24,
                        eviction_policy="drop-oldest",
                        eviction_count=3,
                        evicted_tokens=18,
                        cold_store_format="ollm-kv-sliding-window",
                    ),
                ),
            ),
        ),
    ).to_dict()

    summary = summarize_benchmark_payload(
        payload,
        run_kind="probe-reopen-session-growth",
    )

    assert summary["window_max_tokens"] == 24
    assert summary["eviction_policy"] == "drop-oldest"
    assert summary["eviction_count"] == 3
    assert summary["evicted_tokens"] == 18

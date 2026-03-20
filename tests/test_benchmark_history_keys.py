from ollm.runtime.benchmark_metadata import probe_comparison_key, report_comparison_key


def test_probe_comparison_key_includes_probe_specific_targets() -> None:
    first = probe_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="cpu",
        backend="optimized-native",
        kv_cache_strategy="log-structured-journal",
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
        kv_cache_strategy="log-structured-journal",
        probe_mode="session-growth",
        prompt="Say hi.",
        max_new_tokens=16,
        iterations=1,
        warmup_iterations=0,
        prompt_token_targets=(32,),
        output_token_targets=(16,),
        session_turns=6,
    )

    assert first != second


def test_report_comparison_key_includes_session_max_new_tokens() -> None:
    first = report_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        benchmark_model_reference="gemma3-12B",
        device="cpu",
        kv_cache_strategy="chunked",
        profile_id="full",
        prompt_token_targets=(32, 128, 512),
        output_token_targets=(16, 64, 128),
        session_turns=4,
        session_max_new_tokens=4,
    )
    second = report_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        benchmark_model_reference="gemma3-12B",
        device="cpu",
        kv_cache_strategy="chunked",
        profile_id="full",
        prompt_token_targets=(32, 128, 512),
        output_token_targets=(16, 64, 128),
        session_turns=4,
        session_max_new_tokens=8,
    )

    assert first != second


def test_probe_comparison_key_includes_window_tokens() -> None:
    first = probe_comparison_key(
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
    second = probe_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="cpu",
        backend="optimized-native",
        kv_cache_strategy="sliding-window-ring-buffer",
        kv_cache_window_tokens=96,
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

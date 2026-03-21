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


def test_probe_comparison_key_includes_cpu_offload_policy() -> None:
    first = probe_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="mps",
        backend="optimized-native",
        kv_cache_strategy="resident",
        offload_cpu_layers=2,
        offload_cpu_policy="prefix",
        probe_mode="warm",
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
        device="mps",
        backend="optimized-native",
        kv_cache_strategy="resident",
        offload_cpu_layers=2,
        offload_cpu_policy="middle-band",
        probe_mode="warm",
        prompt="Say hi.",
        max_new_tokens=16,
        iterations=1,
        warmup_iterations=0,
        prompt_token_targets=(32,),
        output_token_targets=(16,),
        session_turns=4,
    )

    assert first != second


def test_probe_comparison_key_includes_strategy_selector_profile() -> None:
    first = probe_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="cpu",
        backend="optimized-native",
        kv_cache_strategy="chunked",
        strategy_selector_profile="balanced",
        probe_mode="warm",
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
        kv_cache_strategy="chunked",
        strategy_selector_profile="capacity",
        probe_mode="warm",
        prompt="Say hi.",
        max_new_tokens=16,
        iterations=1,
        warmup_iterations=0,
        prompt_token_targets=(32,),
        output_token_targets=(16,),
        session_turns=4,
    )

    assert first != second


def test_probe_comparison_key_distinguishes_auto_from_pinned_strategy() -> None:
    auto_selected = probe_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="cpu",
        backend="optimized-native",
        kv_cache_strategy=None,
        strategy_selector_profile="balanced",
        probe_mode="warm",
        prompt="Say hi.",
        max_new_tokens=16,
        iterations=1,
        warmup_iterations=0,
        prompt_token_targets=(32,),
        output_token_targets=(16,),
        session_turns=4,
    )
    pinned = probe_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="cpu",
        backend="optimized-native",
        kv_cache_strategy="paged",
        strategy_selector_profile="balanced",
        probe_mode="warm",
        prompt="Say hi.",
        max_new_tokens=16,
        iterations=1,
        warmup_iterations=0,
        prompt_token_targets=(32,),
        output_token_targets=(16,),
        session_turns=4,
    )

    assert auto_selected != pinned


def test_probe_comparison_key_includes_selector_result_for_auto_runs() -> None:
    first = probe_comparison_key(
        codebase_label="github.com/beardedeagle/ollm",
        model_reference="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device="cpu",
        backend="optimized-native",
        kv_cache_strategy=None,
        strategy_selector_profile="balanced",
        strategy_selector_rule_id="balanced-paged-default",
        strategy_selector_applied_kv_cache_strategy="paged",
        probe_mode="warm",
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
        kv_cache_strategy=None,
        strategy_selector_profile="balanced",
        strategy_selector_rule_id="balanced-small-model-resident",
        strategy_selector_applied_kv_cache_strategy="resident",
        probe_mode="warm",
        prompt="Say hi.",
        max_new_tokens=16,
        iterations=1,
        warmup_iterations=0,
        prompt_token_targets=(32,),
        output_token_targets=(16,),
        session_turns=4,
    )

    assert first != second

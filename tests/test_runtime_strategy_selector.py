from pathlib import Path

import pytest

from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.catalog import ModelModality
from ollm.runtime.plan import RuntimePlan, SpecializationState
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, NativeFamily, ResolvedModel
from ollm.runtime.strategy_selector import select_runtime_strategy


def _build_resolved_model(
    *,
    raw_reference: str = "llama3-1B-chat",
    normalized_name: str = "llama3-1B-chat",
    native_family: NativeFamily | None = NativeFamily.LLAMA,
    modalities: tuple[ModelModality, ...] = (ModelModality.TEXT,),
    requires_processor: bool = False,
) -> ResolvedModel:
    return ResolvedModel(
        reference=ModelReference.parse(raw_reference),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name=normalized_name,
        model_path=Path("test-models/model"),
        repo_id=None,
        revision=None,
        catalog_entry=None,
        capabilities=CapabilityProfile(
            support_level=SupportLevel.OPTIMIZED,
            modalities=modalities,
            requires_processor=requires_processor,
            supports_disk_cache=True,
            supports_local_materialization=True,
            supports_specialization=True,
            details={},
        ),
        native_family=native_family,
        resolution_message="ok",
        architecture="LlamaForCausalLM",
        model_type="llama",
        generic_model_kind=None,
    )


def _build_plan(
    *,
    supports_disk_cache: bool = True,
    execution_device_type: str = "cuda",
) -> RuntimePlan:
    resolved_model = _build_resolved_model()
    return RuntimePlan(
        resolved_model=resolved_model,
        backend_id="optimized-native",
        model_path=resolved_model.model_path,
        support_level=SupportLevel.OPTIMIZED,
        generic_model_kind=None,
        supports_disk_cache=supports_disk_cache,
        supports_cpu_offload=True,
        supports_gpu_offload=False,
        specialization_enabled=True,
        specialization_applied=False,
        specialization_provider_id="llama-native",
        specialization_state=SpecializationState.PLANNED,
        reason="optimized",
        details={"execution_device_type": execution_device_type},
    )


def test_selector_keeps_explicit_strategy_override_pinned() -> None:
    selection = select_runtime_strategy(
        resolved_model=_build_resolved_model(),
        runtime_plan=_build_plan(),
        requested_strategy_override="paged",
        strategy_selector_profile="latency",
        requested_window_tokens=None,
    )

    assert selection.rule_id == "explicit-override"
    assert selection.selected_kv_cache_strategy == "paged"
    assert selection.applied_kv_cache_strategy == "paged"
    assert selection.fallback_chain == ("paged",)


def test_selector_rejects_explicit_disk_strategy_without_disk_support() -> None:
    with pytest.raises(ValueError, match="requires disk-cache support"):
        select_runtime_strategy(
            resolved_model=_build_resolved_model(),
            runtime_plan=_build_plan(
                supports_disk_cache=False, execution_device_type="cpu"
            ),
            requested_strategy_override="paged",
            strategy_selector_profile="balanced",
            requested_window_tokens=None,
        )


def test_selector_uses_sliding_window_for_bounded_window_profile() -> None:
    selection = select_runtime_strategy(
        resolved_model=_build_resolved_model(),
        runtime_plan=_build_plan(),
        requested_strategy_override=None,
        strategy_selector_profile="bounded-window",
        requested_window_tokens=None,
    )

    assert selection.rule_id == "bounded-window-profile"
    assert selection.selected_kv_cache_strategy == "sliding-window-ring-buffer"
    assert selection.fallback_chain == ("sliding-window-ring-buffer",)


def test_selector_falls_back_to_resident_when_disk_cache_is_unavailable() -> None:
    selection = select_runtime_strategy(
        resolved_model=_build_resolved_model(raw_reference="gpt-oss-20B"),
        runtime_plan=_build_plan(
            supports_disk_cache=False, execution_device_type="cuda"
        ),
        requested_strategy_override=None,
        strategy_selector_profile="balanced",
        requested_window_tokens=None,
    )

    assert selection.rule_id == "no-disk-cache-support"
    assert selection.selected_kv_cache_strategy == "resident"


def test_selector_capacity_profile_prefers_quantized_cold_tier_for_text_models() -> (
    None
):
    selection = select_runtime_strategy(
        resolved_model=_build_resolved_model(normalized_name="qwen3-next-80B"),
        runtime_plan=_build_plan(),
        requested_strategy_override=None,
        strategy_selector_profile="capacity",
        requested_window_tokens=None,
    )

    assert selection.rule_id == "capacity-text-full-history"
    assert selection.selected_kv_cache_strategy == "quantized-cold-tier"
    assert selection.fallback_chain == (
        "quantized-cold-tier",
        "paged",
        "chunked",
        "resident",
    )


def test_selector_balanced_profile_prefers_resident_for_small_high_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ollm.runtime.strategy_selector._host_ram_tier", lambda: "large"
    )
    monkeypatch.setattr(
        "ollm.runtime.strategy_selector._accelerator_memory_tier",
        lambda accelerator_kind: "large",
    )

    selection = select_runtime_strategy(
        resolved_model=_build_resolved_model(normalized_name="llama3-1B-chat"),
        runtime_plan=_build_plan(),
        requested_strategy_override=None,
        strategy_selector_profile="balanced",
        requested_window_tokens=None,
    )

    assert selection.rule_id == "balanced-small-model-resident"
    assert selection.selected_kv_cache_strategy == "resident"


def test_selector_balanced_profile_uses_paged_for_multimodal() -> None:
    selection = select_runtime_strategy(
        resolved_model=_build_resolved_model(
            raw_reference="gemma3-12B",
            normalized_name="gemma3-12B",
            native_family=NativeFamily.GEMMA3,
            modalities=(ModelModality.TEXT, ModelModality.IMAGE),
            requires_processor=True,
        ),
        runtime_plan=_build_plan(),
        requested_strategy_override=None,
        strategy_selector_profile="balanced",
        requested_window_tokens=None,
    )

    assert selection.rule_id == "balanced-paged-default"
    assert selection.selected_kv_cache_strategy == "paged"

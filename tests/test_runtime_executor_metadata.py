from dataclasses import replace

from ollm.app.types import ContentPart, Message, MessageRole
from ollm.kv_cache_state import KVCacheStateSnapshot
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.plan import RuntimePlan, SpecializationState
from tests.test_runtime_executor import build_request, build_runtime


class FakeCache:
    def __init__(self) -> None:
        self.snapshot_calls = 0

    def cache_state_snapshot(self) -> KVCacheStateSnapshot:
        self.snapshot_calls += 1
        return KVCacheStateSnapshot(
            strategy_id="tiered-write-back",
            policy_id="test-tiered",
            persistence_format="log-structured-journal",
            residency_mode="tiered-write-back",
            window_policy="full-history",
            cold_tier_encoding="full-precision",
            cold_tier_representation=None,
            persisted_layer_count=2,
            persisted_tokens=64,
            persisted_artifact_count=5,
            resident_layer_count=2,
            resident_tokens=64,
            resident_bytes=2048,
            hot_layer_count=1,
            hot_tokens=8,
            hot_bytes=1024,
            compaction_count=1,
            spill_count=3,
            spilled_tokens=56,
            cold_store_format="ollm-kv-journal",
        )


def test_runtime_executor_includes_execution_device_details_in_metadata() -> None:
    capabilities = CapabilityProfile(support_level=SupportLevel.OPTIMIZED)
    runtime = build_runtime(capabilities)
    runtime.plan = RuntimePlan(
        resolved_model=runtime.plan.resolved_model,
        backend_id="optimized-native",
        model_path=runtime.plan.model_path,
        support_level=SupportLevel.OPTIMIZED,
        generic_model_kind=None,
        supports_disk_cache=True,
        supports_cpu_offload=True,
        supports_gpu_offload=False,
        specialization_enabled=True,
        specialization_applied=True,
        specialization_provider_id="llama-native",
        specialization_state=SpecializationState.APPLIED,
        reason="optimized",
        specialization_pass_ids=(),
        applied_specialization_pass_ids=(),
        details={
            "execution_device_type": "mps",
            "specialization_device_profile": "accelerator-resident",
        },
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )

    response = RuntimeExecutor().execute(runtime, request)

    assert response.metadata["execution_device_type"] == "mps"
    assert response.metadata["specialization_device_profile"] == "accelerator-resident"


def test_runtime_executor_includes_kv_cache_state_metadata() -> None:
    runtime = build_runtime(CapabilityProfile(support_level=SupportLevel.OPTIMIZED))
    runtime.config.use_cache = True
    runtime.config.kv_cache_strategy = "tiered-write-back"
    runtime.plan = replace(runtime.plan, supports_disk_cache=True)
    runtime.backend.create_cache = (
        lambda cache_dir, cache_strategy=None, cache_lifecycle=None: FakeCache()
    )
    request = build_request(
        runtime.config,
        Message(role=MessageRole.USER, content=[ContentPart.text("hello")]),
    )

    response = RuntimeExecutor().execute(runtime, request)

    assert response.metadata["kv_cache_strategy"] == "tiered-write-back"
    assert response.metadata["kv_cache_policy_id"] == "test-tiered"
    assert response.metadata["kv_cache_persisted_tokens"] == "64"
    assert response.metadata["kv_cache_persisted_artifacts"] == "5"
    assert response.metadata["kv_cache_cold_store_format"] == "ollm-kv-journal"
    assert response.metadata["kv_cache_persistence_format"] == "log-structured-journal"
    assert response.metadata["kv_cache_residency_mode"] == "tiered-write-back"
    assert response.metadata["kv_cache_window_policy"] == "full-history"
    assert response.metadata["kv_cache_cold_tier_encoding"] == "full-precision"
    assert "kv_cache_cold_tier_representation" not in response.metadata
    assert response.metadata["kv_cache_resident_layers"] == "2"
    assert response.metadata["kv_cache_resident_tokens"] == "64"
    assert response.metadata["kv_cache_resident_bytes"] == "2048"
    assert response.metadata["kv_cache_hot_tokens"] == "8"
    assert response.metadata["kv_cache_compaction_count"] == "1"
    assert response.metadata["kv_cache_spill_count"] == "3"
    assert response.metadata["kv_cache_lifecycle"] == "runtime-scoped"
    assert response.metadata["kv_cache_adaptation_mode"] == "observe-only"
    assert response.metadata["kv_cache_adaptation_recommendation_available"] == "true"
    assert (
        response.metadata["kv_cache_adaptation_recommended_strategy"]
        == "tiered-write-back"
    )

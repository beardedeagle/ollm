from pathlib import Path

import torch

from ollm.runtime.backends.native_optimized import NativeOptimizedBackend
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, NativeFamily, ResolvedModel
from ollm.runtime.specialization import SpecializationRegistry
from ollm.runtime.specialization.base import (
    OptimizedModelArtifacts,
    SpecializationMatch,
    SpecializationProvider,
    SpecializationTraits,
)


class StubProvider(SpecializationProvider):
    provider_id = "stub-llama"
    native_family = NativeFamily.LLAMA

    def __init__(self):
        self.load_count = 0

    def match(self, resolved_model: ResolvedModel, config: RuntimeConfig) -> SpecializationMatch | None:
        del resolved_model, config
        return SpecializationMatch(
            provider_id=self.provider_id,
            native_family=self.native_family,
            reason="stub match",
            traits=SpecializationTraits(
                supports_disk_cache=True,
                supports_cpu_offload=True,
                supports_gpu_offload=False,
            ),
        )

    def load(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        stats,
    ) -> OptimizedModelArtifacts:
        del resolved_model, config, stats
        self.load_count += 1
        return OptimizedModelArtifacts(
            model=object(),
            tokenizer=object(),
            processor=None,
            device=torch.device("cpu"),
            stats=None,
            create_cache=lambda cache_dir: str(cache_dir),
            apply_cpu_offload=lambda layers_num: None,
            apply_gpu_offload=None,
        )


def test_native_optimized_backend_loads_through_specialization_registry(tmp_path: Path) -> None:
    provider = StubProvider()
    registry = SpecializationRegistry((provider,))
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("llama3-1B-chat"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="llama3-1B-chat",
        model_path=tmp_path / "llama3-1B-chat",
        repo_id="repo",
        revision=None,
        provider_name=None,
        catalog_entry=None,
        capabilities=CapabilityProfile(support_level=SupportLevel.OPTIMIZED),
        native_family=NativeFamily.LLAMA,
        resolution_message="built-in",
        architecture="LlamaForCausalLM",
        model_type="llama",
        generic_model_kind=GenericModelKind.CAUSAL_LM,
    )
    plan = RuntimePlan(
        resolved_model=resolved_model,
        backend_id="optimized-native",
        model_path=resolved_model.model_path,
        support_level=SupportLevel.OPTIMIZED,
        generic_model_kind=GenericModelKind.CAUSAL_LM,
        supports_disk_cache=True,
        supports_cpu_offload=True,
        supports_gpu_offload=False,
        specialization_enabled=True,
        specialization_provider_id="stub-llama",
        reason="stub",
    )

    runtime = NativeOptimizedBackend(specialization_registry=registry).load(plan, RuntimeConfig(device="cpu"))

    assert runtime.backend_id == "optimized-native"
    assert provider.load_count == 1

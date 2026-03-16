import torch

from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, NativeFamily, ResolvedModel
from ollm.runtime.specialization import SpecializationRegistry
from ollm.runtime.specialization.base import (
    OptimizedModelArtifacts,
    SpecializationMatch,
    SpecializationProvider,
    SpecializationTraits,
)


class RegistryProvider(SpecializationProvider):
    provider_id = "registry-llama"
    native_family = NativeFamily.LLAMA

    def match(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> SpecializationMatch | None:
        del config
        if resolved_model.native_family is not NativeFamily.LLAMA:
            return None
        return SpecializationMatch(
            provider_id=self.provider_id,
            native_family=self.native_family,
            reason="registry match",
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
        return OptimizedModelArtifacts(
            model=object(),
            tokenizer=object(),
            processor=None,
            device=torch.device("cpu"),
            stats=None,
            supports_disk_cache=True,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            print_suppression_modules=(),
            create_cache=lambda cache_dir: str(cache_dir),
            apply_cpu_offload=lambda layers_num: None,
            apply_gpu_offload=None,
        )


def test_specialization_registry_selects_provider_by_native_family() -> None:
    registry = SpecializationRegistry((RegistryProvider(),))
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("llama3-1B-chat"),
        source_kind=ModelSourceKind.LOCAL_PATH,
        normalized_name="llama",
        model_path=None,
        repo_id=None,
        revision=None,
        catalog_entry=None,
        capabilities=CapabilityProfile(
            support_level=SupportLevel.GENERIC,
            supports_specialization=True,
        ),
        native_family=NativeFamily.LLAMA,
        resolution_message="local llama",
        architecture="LlamaForCausalLM",
        model_type="llama",
        generic_model_kind=GenericModelKind.CAUSAL_LM,
    )

    match = registry.select(resolved_model, RuntimeConfig(device="cpu"))

    assert match is not None
    assert match.provider_id == "registry-llama"
    assert match.traits.supports_cpu_offload is True

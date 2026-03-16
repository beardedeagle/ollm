import logging
from pathlib import Path
import sys
import types

import pytest
import torch

from ollm.runtime.backends.native_optimized import NativeOptimizedBackend
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan, SpecializationState
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, NativeFamily, ResolvedModel
from ollm.runtime.specialization import SpecializationRegistry
from ollm.runtime.specialization.base import (
    OptimizedModelArtifacts,
    SpecializationMatch,
    SpecializationLoadError,
    SpecializationProvider,
    SpecializationTraits,
)
from ollm.runtime.specialization.passes.base import SpecializationPassId


class StubProvider(SpecializationProvider):
    provider_id = "stub-llama"
    native_family = NativeFamily.LLAMA

    def __init__(self):
        self.load_count = 0
        self._module = types.ModuleType("stub_native_module")

    def match(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> SpecializationMatch | None:
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
            supports_disk_cache=True,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            print_suppression_modules=(self._module,),
            create_cache=lambda cache_dir: str(cache_dir),
            apply_cpu_offload=lambda layers_num: self._module.print(
                f"cpu-offload:{layers_num}"
            ),
            apply_gpu_offload=None,
        )


class RuntimeErrorProvider(SpecializationProvider):
    provider_id = "runtime-error-llama"
    native_family = NativeFamily.LLAMA

    def match(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> SpecializationMatch | None:
        del resolved_model, config
        return SpecializationMatch(
            provider_id=self.provider_id,
            native_family=self.native_family,
            reason="runtime error match",
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
        raise RuntimeError("simulated optimized load failure")


class NoisyProvider(SpecializationProvider):
    provider_id = "noisy-llama"
    native_family = NativeFamily.LLAMA

    def __init__(self):
        self._module = sys.modules[__name__]

    def match(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> SpecializationMatch | None:
        del resolved_model, config
        return SpecializationMatch(
            provider_id=self.provider_id,
            native_family=self.native_family,
            reason="noisy match",
            traits=SpecializationTraits(
                supports_disk_cache=True,
                supports_cpu_offload=False,
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
        print("stdout-noise")
        sys.stderr.write("stderr-noise")
        logging.getLogger("transformers").warning("logger-noise")
        return OptimizedModelArtifacts(
            model=object(),
            tokenizer=object(),
            processor=None,
            device=torch.device("cpu"),
            stats=None,
            supports_disk_cache=True,
            supports_cpu_offload=False,
            supports_gpu_offload=False,
            print_suppression_modules=(self._module,),
            create_cache=lambda cache_dir: None,
            apply_cpu_offload=None,
            apply_gpu_offload=None,
        )


def test_native_optimized_backend_loads_through_specialization_registry(
    tmp_path: Path,
) -> None:
    provider = StubProvider()
    registry = SpecializationRegistry((provider,))
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("llama3-1B-chat"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="llama3-1B-chat",
        model_path=tmp_path / "llama3-1B-chat",
        repo_id="repo",
        revision=None,
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
        specialization_applied=False,
        specialization_provider_id="stub-llama",
        specialization_state=SpecializationState.PLANNED,
        reason="stub",
        specialization_pass_ids=(
            SpecializationPassId.DISK_CACHE,
            SpecializationPassId.CPU_OFFLOAD,
        ),
    )

    runtime = NativeOptimizedBackend(specialization_registry=registry).load(
        plan, RuntimeConfig(device="cpu")
    )

    assert runtime.backend_id == "optimized-native"
    assert provider.load_count == 1
    assert runtime.applied_specialization is not None
    assert runtime.applied_specialization.applied_pass_ids == (
        SpecializationPassId.DISK_CACHE,
        SpecializationPassId.CPU_OFFLOAD,
    )


def test_native_optimized_backend_suppresses_module_prints_during_offload(
    tmp_path: Path, capfd
) -> None:
    provider = StubProvider()
    registry = SpecializationRegistry((provider,))
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("llama3-1B-chat"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="llama3-1B-chat",
        model_path=tmp_path / "llama3-1B-chat",
        repo_id="repo",
        revision=None,
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
        specialization_applied=False,
        specialization_provider_id="stub-llama",
        specialization_state=SpecializationState.PLANNED,
        reason="stub",
        specialization_pass_ids=(SpecializationPassId.CPU_OFFLOAD,),
    )

    runtime = NativeOptimizedBackend(specialization_registry=registry).load(
        plan,
        RuntimeConfig(device="cpu"),
    )
    runtime.apply_offload(RuntimeConfig(device="cpu", offload_cpu_layers=1))

    captured = capfd.readouterr()
    assert captured.out == ""


def test_native_optimized_backend_wraps_runtime_error_load_failures(
    tmp_path: Path,
) -> None:
    registry = SpecializationRegistry((RuntimeErrorProvider(),))
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("llama3-1B-chat"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="llama3-1B-chat",
        model_path=tmp_path / "llama3-1B-chat",
        repo_id="repo",
        revision=None,
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
        specialization_applied=False,
        specialization_provider_id="runtime-error-llama",
        specialization_state=SpecializationState.PLANNED,
        reason="stub",
        specialization_pass_ids=(SpecializationPassId.DISK_CACHE,),
    )

    with pytest.raises(
        SpecializationLoadError, match="simulated optimized load failure"
    ):
        NativeOptimizedBackend(specialization_registry=registry).load(
            plan, RuntimeConfig(device="cpu")
        )


def test_native_optimized_backend_suppresses_external_load_noise_by_default(
    tmp_path: Path, monkeypatch, capfd
) -> None:
    provider = NoisyProvider()
    registry = SpecializationRegistry((provider,))
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("llama3-1B-chat"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="llama3-1B-chat",
        model_path=tmp_path / "llama3-1B-chat",
        repo_id="repo",
        revision=None,
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
        supports_cpu_offload=False,
        supports_gpu_offload=False,
        specialization_enabled=True,
        specialization_applied=False,
        specialization_provider_id="noisy-llama",
        specialization_state=SpecializationState.PLANNED,
        reason="stub",
        specialization_pass_ids=(SpecializationPassId.DISK_CACHE,),
    )
    monkeypatch.setattr(
        "ollm.runtime.backends.native_optimized._modules_for_provider_id",
        lambda provider_id: (provider._module,),
    )

    NativeOptimizedBackend(specialization_registry=registry).load(
        plan, RuntimeConfig(device="cpu")
    )

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""

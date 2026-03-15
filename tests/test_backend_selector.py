from pathlib import Path

from ollm.runtime.backend_selector import BackendSelector
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.catalog import ModelCatalogEntry, ModelModality
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, NativeFamily, ResolvedModel
from ollm.runtime.specialization.passes.base import SpecializationPassId


def build_catalog_resolved_model() -> ResolvedModel:
    catalog_entry = ModelCatalogEntry(
        model_id="llama3-1B-chat",
        summary="test",
        repo_id="repo",
        modalities=(ModelModality.TEXT,),
    )
    return ResolvedModel(
        reference=ModelReference.parse("llama3-1B-chat"),
        source_kind=ModelSourceKind.BUILTIN,
        normalized_name="llama3-1B-chat",
        model_path=Path("/tmp/llama3-1B-chat"),
        repo_id="repo",
        revision=None,
        provider_name=None,
        catalog_entry=catalog_entry,
        capabilities=CapabilityProfile(support_level=SupportLevel.OPTIMIZED),
        native_family=NativeFamily.LLAMA,
        resolution_message="built-in",
        architecture="LlamaForCausalLM",
        model_type="llama",
        generic_model_kind=GenericModelKind.CAUSAL_LM,
    )


def test_backend_selector_prefers_optimized_native_for_built_in_aliases() -> None:
    plan = BackendSelector().select(build_catalog_resolved_model(), RuntimeConfig())
    assert plan.backend_id == "optimized-native"
    assert plan.support_level is SupportLevel.OPTIMIZED
    assert plan.specialization_provider_id == "llama-native"
    assert plan.specialization_pass_ids == (
        SpecializationPassId.DISK_CACHE,
        SpecializationPassId.CPU_OFFLOAD,
        SpecializationPassId.MLP_CHUNKING,
    )
    assert plan.supports_cpu_offload is True
    assert plan.supports_gpu_offload is False


def test_backend_selector_routes_catalog_models_with_adapters_to_generic_backend() -> (
    None
):
    plan = BackendSelector().select(
        build_catalog_resolved_model(),
        RuntimeConfig(adapter_dir=Path("/tmp/adapter")),
    )
    assert plan.backend_id == "transformers-generic"
    assert plan.support_level is SupportLevel.GENERIC
    assert plan.specialization_provider_id is None


def test_backend_selector_honors_explicit_generic_backend_override() -> None:
    plan = BackendSelector().select(
        build_catalog_resolved_model(),
        RuntimeConfig(backend="transformers-generic"),
    )
    assert plan.backend_id == "transformers-generic"
    assert plan.support_level is SupportLevel.GENERIC
    assert plan.specialization_enabled is False
    assert plan.details["backend_override"] == "transformers-generic"


def test_backend_selector_skips_specialization_when_disabled() -> None:
    plan = BackendSelector().select(
        build_catalog_resolved_model(),
        RuntimeConfig(use_specialization=False),
    )
    assert plan.backend_id == "transformers-generic"
    assert plan.support_level is SupportLevel.GENERIC
    assert plan.specialization_enabled is False


def test_backend_selector_prefers_optimized_native_for_local_native_family() -> None:
    resolved_model = build_catalog_resolved_model()
    resolved_model = ResolvedModel(
        reference=ModelReference.parse("/tmp/llama"),
        source_kind=ModelSourceKind.LOCAL_PATH,
        normalized_name="llama",
        model_path=Path("/tmp/llama"),
        repo_id=None,
        revision=None,
        provider_name=None,
        catalog_entry=None,
        capabilities=CapabilityProfile(
            support_level=SupportLevel.GENERIC,
            supports_specialization=True,
        ),
        native_family=NativeFamily.LLAMA,
        resolution_message="local",
        architecture="LlamaForCausalLM",
        model_type="llama",
        generic_model_kind=GenericModelKind.CAUSAL_LM,
    )
    plan = BackendSelector().select(resolved_model, RuntimeConfig())
    assert plan.backend_id == "optimized-native"
    assert plan.specialization_provider_id == "llama-native"
    assert plan.specialization_pass_ids == (
        SpecializationPassId.DISK_CACHE,
        SpecializationPassId.CPU_OFFLOAD,
        SpecializationPassId.MLP_CHUNKING,
    )


def test_backend_selector_falls_back_to_generic_when_no_specialization_matches() -> (
    None
):
    resolved_model = build_catalog_resolved_model()
    resolved_model = ResolvedModel(
        reference=resolved_model.reference,
        source_kind=resolved_model.source_kind,
        normalized_name=resolved_model.normalized_name,
        model_path=resolved_model.model_path,
        repo_id=resolved_model.repo_id,
        revision=resolved_model.revision,
        provider_name=resolved_model.provider_name,
        catalog_entry=resolved_model.catalog_entry,
        capabilities=resolved_model.capabilities,
        native_family=None,
        resolution_message=resolved_model.resolution_message,
        architecture=resolved_model.architecture,
        model_type=resolved_model.model_type,
        generic_model_kind=resolved_model.generic_model_kind,
    )
    plan = BackendSelector().select(resolved_model, RuntimeConfig())
    assert plan.backend_id == "transformers-generic"
    assert plan.support_level is SupportLevel.GENERIC


def test_backend_selector_rejects_provider_backend_override_for_local_models() -> None:
    plan = BackendSelector().select(
        build_catalog_resolved_model(),
        RuntimeConfig(backend="openai-compatible"),
    )
    assert plan.backend_id is None
    assert "only supports openai-compatible: or lmstudio:" in plan.reason

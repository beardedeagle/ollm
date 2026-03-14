from pathlib import Path

from ollm.runtime.backend_selector import BackendSelector
from ollm.runtime.capabilities import CapabilityProfile, SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.catalog import ModelCatalogEntry, ModelModality
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.reference import ModelReference
from ollm.runtime.resolver import ModelSourceKind, ResolvedModel


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
        native_family=None,
        resolution_message="built-in",
        architecture="LlamaForCausalLM",
        model_type="llama",
        generic_model_kind=GenericModelKind.CAUSAL_LM,
    )


def test_backend_selector_prefers_optimized_native_for_built_in_aliases() -> None:
    plan = BackendSelector().select(build_catalog_resolved_model(), RuntimeConfig())
    assert plan.backend_id == "optimized-native"
    assert plan.support_level is SupportLevel.OPTIMIZED


def test_backend_selector_routes_catalog_models_with_adapters_to_generic_backend() -> None:
    plan = BackendSelector().select(
        build_catalog_resolved_model(),
        RuntimeConfig(adapter_dir=Path("/tmp/adapter")),
    )
    assert plan.backend_id == "transformers-generic"
    assert plan.support_level is SupportLevel.GENERIC

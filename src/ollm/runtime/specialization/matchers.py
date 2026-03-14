from dataclasses import dataclass

from ollm.runtime.catalog import ModelModality
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.resolver import NativeFamily, ResolvedModel
from ollm.runtime.specialization.passes.base import SpecializationPassId


@dataclass(frozen=True, slots=True)
class SpecializationPassMatcher:
    matcher_id: str
    pass_id: SpecializationPassId
    provider_ids: tuple[str, ...] = ()
    native_families: tuple[NativeFamily, ...] = ()
    architectures: tuple[str, ...] = ()
    required_modalities: tuple[ModelModality, ...] = ()
    require_multimodal_config: bool | None = None
    require_gds_export: bool | None = None
    require_disk_cache_support: bool | None = None

    def matches(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        provider_id: str | None,
    ) -> bool:
        if self.provider_ids and provider_id not in self.provider_ids:
            return False
        if self.native_families:
            if resolved_model.native_family is None or resolved_model.native_family not in self.native_families:
                return False
        if self.architectures:
            if resolved_model.architecture is None or resolved_model.architecture not in self.architectures:
                return False
        if self.required_modalities:
            modalities = set(resolved_model.capabilities.modalities)
            if not set(self.required_modalities).issubset(modalities):
                return False
        if self.require_multimodal_config is not None and config.multimodal is not self.require_multimodal_config:
            return False
        if self.require_gds_export is not None:
            has_gds_export = _has_gds_export(resolved_model)
            if has_gds_export is not self.require_gds_export:
                return False
        if self.require_disk_cache_support is not None:
            supports_disk_cache = _supports_disk_cache(resolved_model)
            if supports_disk_cache is not self.require_disk_cache_support:
                return False
        return True


def build_default_pass_matchers() -> tuple[SpecializationPassMatcher, ...]:
    return (
        SpecializationPassMatcher(
            matcher_id="llama-disk-cache",
            pass_id=SpecializationPassId.DISK_CACHE,
            provider_ids=("llama-native",),
            native_families=(NativeFamily.LLAMA,),
            require_disk_cache_support=True,
        ),
        SpecializationPassMatcher(
            matcher_id="llama-cpu-offload",
            pass_id=SpecializationPassId.CPU_OFFLOAD,
            provider_ids=("llama-native",),
            native_families=(NativeFamily.LLAMA,),
        ),
        SpecializationPassMatcher(
            matcher_id="llama-mlp-chunking",
            pass_id=SpecializationPassId.MLP_CHUNKING,
            provider_ids=("llama-native",),
            native_families=(NativeFamily.LLAMA,),
        ),
        SpecializationPassMatcher(
            matcher_id="gemma3-disk-cache",
            pass_id=SpecializationPassId.DISK_CACHE,
            provider_ids=("gemma3-native",),
            native_families=(NativeFamily.GEMMA3,),
            require_disk_cache_support=True,
        ),
        SpecializationPassMatcher(
            matcher_id="gemma3-cpu-offload",
            pass_id=SpecializationPassId.CPU_OFFLOAD,
            provider_ids=("gemma3-native",),
            native_families=(NativeFamily.GEMMA3,),
        ),
        SpecializationPassMatcher(
            matcher_id="gemma3-mlp-chunking",
            pass_id=SpecializationPassId.MLP_CHUNKING,
            provider_ids=("gemma3-native",),
            native_families=(NativeFamily.GEMMA3,),
        ),
        SpecializationPassMatcher(
            matcher_id="gemma3-multimodal-shell",
            pass_id=SpecializationPassId.MULTIMODAL_SHELL,
            provider_ids=("gemma3-native",),
            native_families=(NativeFamily.GEMMA3,),
            required_modalities=(ModelModality.IMAGE,),
            require_multimodal_config=True,
        ),
        SpecializationPassMatcher(
            matcher_id="qwen3-next-disk-cache",
            pass_id=SpecializationPassId.DISK_CACHE,
            provider_ids=("qwen3-next-native",),
            native_families=(NativeFamily.QWEN3_NEXT,),
            require_disk_cache_support=True,
        ),
        SpecializationPassMatcher(
            matcher_id="qwen3-next-cpu-offload",
            pass_id=SpecializationPassId.CPU_OFFLOAD,
            provider_ids=("qwen3-next-native",),
            native_families=(NativeFamily.QWEN3_NEXT,),
        ),
        SpecializationPassMatcher(
            matcher_id="qwen3-next-gpu-offload",
            pass_id=SpecializationPassId.GPU_OFFLOAD,
            provider_ids=("qwen3-next-native",),
            native_families=(NativeFamily.QWEN3_NEXT,),
        ),
        SpecializationPassMatcher(
            matcher_id="qwen3-next-moe-routing",
            pass_id=SpecializationPassId.MOE_ROUTING,
            provider_ids=("qwen3-next-native",),
            native_families=(NativeFamily.QWEN3_NEXT,),
            architectures=("Qwen3NextForCausalLM", "Qwen3MoeForCausalLM"),
        ),
        SpecializationPassMatcher(
            matcher_id="gpt-oss-cpu-offload",
            pass_id=SpecializationPassId.CPU_OFFLOAD,
            provider_ids=("gpt-oss-native",),
            native_families=(NativeFamily.GPT_OSS,),
        ),
        SpecializationPassMatcher(
            matcher_id="gpt-oss-moe-routing",
            pass_id=SpecializationPassId.MOE_ROUTING,
            provider_ids=("gpt-oss-native",),
            native_families=(NativeFamily.GPT_OSS,),
        ),
        SpecializationPassMatcher(
            matcher_id="gpt-oss-attention-replacement",
            pass_id=SpecializationPassId.ATTENTION_REPLACEMENT,
            provider_ids=("gpt-oss-native",),
            native_families=(NativeFamily.GPT_OSS,),
        ),
        SpecializationPassMatcher(
            matcher_id="gpt-oss-gds-export",
            pass_id=SpecializationPassId.GDS_EXPORT_WEIGHTS,
            provider_ids=("gpt-oss-native",),
            native_families=(NativeFamily.GPT_OSS,),
            require_gds_export=True,
        ),
        SpecializationPassMatcher(
            matcher_id="voxtral-disk-cache",
            pass_id=SpecializationPassId.DISK_CACHE,
            provider_ids=("voxtral-native",),
            native_families=(NativeFamily.VOXTRAL,),
            require_disk_cache_support=True,
        ),
        SpecializationPassMatcher(
            matcher_id="voxtral-cpu-offload",
            pass_id=SpecializationPassId.CPU_OFFLOAD,
            provider_ids=("voxtral-native",),
            native_families=(NativeFamily.VOXTRAL,),
        ),
        SpecializationPassMatcher(
            matcher_id="voxtral-multimodal-shell",
            pass_id=SpecializationPassId.MULTIMODAL_SHELL,
            provider_ids=("voxtral-native",),
            native_families=(NativeFamily.VOXTRAL,),
            required_modalities=(ModelModality.AUDIO,),
            require_multimodal_config=True,
        ),
    )


def _has_gds_export(resolved_model: ResolvedModel) -> bool:
    if resolved_model.model_path is None:
        return False
    return (resolved_model.model_path.expanduser().resolve() / "gds_export").is_dir()


def _supports_disk_cache(resolved_model: ResolvedModel) -> bool:
    if resolved_model.catalog_entry is None:
        return True
    return resolved_model.catalog_entry.supports_disk_cache

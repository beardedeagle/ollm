from functools import lru_cache
from importlib import import_module
from typing import cast

import torch
from transformers import AutoProcessor, AutoTokenizer

from ollm.gds_loader import (
    DenseWeightsLoader,
    GDSWeights,
    MoEWeightsLoader,
    SingleDenseWeightsLoader,
)
from ollm.kvcache import KVCache
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.resolver import NativeFamily, ResolvedModel
from ollm.runtime.safety import (
    validate_safe_gds_export_artifacts,
    validate_safe_model_artifacts,
)
from ollm.runtime.specialization.base import (
    OptimizedModelArtifacts,
    SpecializationMatch,
    SpecializationProvider,
)
from ollm.runtime.specialization.passes.base import SpecializationPassId
from ollm.runtime.specialization.provider_support import (
    _CpuOffloadModel,
    _GpuCpuOffloadModel,
    build_match,
    finalize_model,
    get_attention_implementation,
    is_sharded_model_dir,
    load_specialized_model,
    matches_architecture,
    resolved_model_path,
    unsupported_disk_cache_factory,
)
from ollm.runtime.specialization.registry import SpecializationRegistry
from ollm.utils import Stats


class LlamaSpecializationProvider(SpecializationProvider):
    provider_id = "llama-native"
    native_family = NativeFamily.LLAMA

    def match(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> SpecializationMatch | None:
        del config
        if not matches_architecture(resolved_model, ("LlamaForCausalLM",)):
            return None
        supports_disk_cache = (
            resolved_model.catalog_entry is None
            or resolved_model.catalog_entry.supports_disk_cache
        )
        return build_match(
            resolved_model=resolved_model,
            native_family=self.native_family,
            provider_id=self.provider_id,
            supports_disk_cache=supports_disk_cache,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
        )

    def load(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        stats: Stats | None,
    ) -> OptimizedModelArtifacts:
        model_path = resolved_model_path(resolved_model)
        validate_safe_model_artifacts(model_path)
        module = import_module("ollm.llama")
        device = torch.device(config.device)
        if is_sharded_model_dir(model_path):
            setattr(
                module,
                "loader",
                DenseWeightsLoader(str(model_path), device=str(device)),
            )
        else:
            setattr(
                module,
                "loader",
                SingleDenseWeightsLoader(str(model_path), device=str(device)),
            )
        setattr(module, "stats", stats)
        model = cast(
            _CpuOffloadModel,
            load_specialized_model(
                module.MyLlamaForCausalLM.from_pretrained,
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=get_attention_implementation(),
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=False
        )
        return OptimizedModelArtifacts(
            model=finalize_model(model, device),
            tokenizer=tokenizer,
            processor=None,
            device=device,
            stats=stats,
            supports_disk_cache=True,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            print_suppression_modules=(module,),
            create_cache=lambda cache_dir: KVCache(
                cache_dir=str(cache_dir), device=device, stats=stats
            ),
            apply_cpu_offload=lambda layers_num: model.offload_layers_to_cpu(
                layers_num=layers_num
            ),
            apply_gpu_offload=None,
            provided_pass_ids=(SpecializationPassId.MLP_CHUNKING,),
        )


class Gemma3SpecializationProvider(SpecializationProvider):
    provider_id = "gemma3-native"
    native_family = NativeFamily.GEMMA3

    def match(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> SpecializationMatch | None:
        del config
        if not matches_architecture(
            resolved_model,
            ("Gemma3ForConditionalGeneration", "Gemma3ForCausalLM"),
        ):
            return None
        supports_disk_cache = (
            resolved_model.catalog_entry is None
            or resolved_model.catalog_entry.supports_disk_cache
        )
        return build_match(
            resolved_model=resolved_model,
            native_family=self.native_family,
            provider_id=self.provider_id,
            supports_disk_cache=supports_disk_cache,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
        )

    def load(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        stats: Stats | None,
    ) -> OptimizedModelArtifacts:
        model_path = resolved_model_path(resolved_model)
        validate_safe_model_artifacts(model_path)
        module = import_module("ollm.gemma3")
        device = torch.device(config.device)
        setattr(
            module, "loader", DenseWeightsLoader(str(model_path), device=str(device))
        )
        setattr(module, "stats", stats)
        model_class = (
            module.MyGemma3ForConditionalGeneration
            if config.multimodal
            else module.MyGemma3ForCausalLM
        )
        model = cast(
            _CpuOffloadModel,
            load_specialized_model(
                model_class.from_pretrained,
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=get_attention_implementation(),
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=False
        )
        processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=False
        )
        return OptimizedModelArtifacts(
            model=finalize_model(model, device),
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            stats=stats,
            supports_disk_cache=True,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            print_suppression_modules=(module,),
            create_cache=lambda cache_dir: KVCache(
                cache_dir=str(cache_dir), device=device, stats=stats
            ),
            apply_cpu_offload=lambda layers_num: model.offload_layers_to_cpu(
                layers_num=layers_num
            ),
            apply_gpu_offload=None,
            provided_pass_ids=(SpecializationPassId.MLP_CHUNKING,),
        )


class Qwen3NextSpecializationProvider(SpecializationProvider):
    provider_id = "qwen3-next-native"
    native_family = NativeFamily.QWEN3_NEXT

    def match(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> SpecializationMatch | None:
        del config
        if not matches_architecture(
            resolved_model,
            ("Qwen3NextForCausalLM", "Qwen3MoeForCausalLM"),
        ):
            return None
        supports_disk_cache = (
            resolved_model.catalog_entry is None
            or resolved_model.catalog_entry.supports_disk_cache
        )
        return build_match(
            resolved_model=resolved_model,
            native_family=self.native_family,
            provider_id=self.provider_id,
            supports_disk_cache=supports_disk_cache,
            supports_cpu_offload=True,
            supports_gpu_offload=True,
        )

    def load(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        stats: Stats | None,
    ) -> OptimizedModelArtifacts:
        model_path = resolved_model_path(resolved_model)
        validate_safe_model_artifacts(model_path)
        module = import_module("ollm.qwen3_next")
        device = torch.device(config.device)
        setattr(module, "loader", MoEWeightsLoader(str(model_path), device=str(device)))
        setattr(module, "stats", stats)
        model = cast(
            _GpuCpuOffloadModel,
            load_specialized_model(
                module.MyQwen3NextForCausalLM.from_pretrained,
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=get_attention_implementation(),
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=False
        )
        return OptimizedModelArtifacts(
            model=finalize_model(model, device),
            tokenizer=tokenizer,
            processor=None,
            device=device,
            stats=stats,
            supports_disk_cache=True,
            supports_cpu_offload=True,
            supports_gpu_offload=True,
            print_suppression_modules=(module,),
            create_cache=(
                lambda cache_dir: module.Qwen3NextDiskCache(
                    model.config,
                    cache_dir=str(cache_dir),
                    device=device,
                    stats=stats,
                )
            ),
            apply_cpu_offload=lambda layers_num: model.offload_layers_to_cpu(
                layers_num=layers_num
            ),
            apply_gpu_offload=(
                lambda gpu_layers_num, cpu_layers_num: model.offload_layers_to_gpu_cpu(
                    gpu_layers_num=gpu_layers_num,
                    cpu_layers_num=cpu_layers_num,
                )
            ),
            provided_pass_ids=(SpecializationPassId.MOE_ROUTING,),
        )


class GptOssSpecializationProvider(SpecializationProvider):
    provider_id = "gpt-oss-native"
    native_family = NativeFamily.GPT_OSS

    def match(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> SpecializationMatch | None:
        del config
        if not matches_architecture(
            resolved_model,
            ("GptOssForCausalLM", "OpenAIGptOssForCausalLM"),
        ):
            return None
        model_path = resolved_model.model_path
        if (
            model_path is None
            or not (model_path.expanduser().resolve() / "gds_export").exists()
        ):
            return None
        try:
            validate_safe_gds_export_artifacts(
                model_path.expanduser().resolve() / "gds_export"
            )
        except ValueError:
            return None
        return build_match(
            resolved_model=resolved_model,
            native_family=self.native_family,
            provider_id=self.provider_id,
            supports_disk_cache=False,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
        )

    def load(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        stats: Stats | None,
    ) -> OptimizedModelArtifacts:
        model_path = resolved_model_path(resolved_model)
        validate_safe_model_artifacts(model_path)
        export_path = model_path / "gds_export"
        if not export_path.exists() or not export_path.is_dir():
            raise ValueError(
                f"gpt-oss optimized specialization requires a gds_export directory in {model_path}"
            )
        validate_safe_gds_export_artifacts(export_path)
        module = import_module("ollm.gpt_oss")
        device = torch.device(config.device)
        setattr(module, "loader", GDSWeights(str(export_path), device=str(device)))
        setattr(module, "stats", stats)
        model = cast(
            _CpuOffloadModel,
            load_specialized_model(
                module.MyGptOssForCausalLM.from_pretrained,
                model_path,
                torch_dtype=torch.bfloat16,
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=False
        )
        return OptimizedModelArtifacts(
            model=finalize_model(model, device),
            tokenizer=tokenizer,
            processor=None,
            device=device,
            stats=stats,
            supports_disk_cache=False,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            print_suppression_modules=(module,),
            create_cache=unsupported_disk_cache_factory(resolved_model.reference.raw),
            apply_cpu_offload=lambda layers_num: model.offload_layers_to_cpu(
                layers_num=layers_num
            ),
            apply_gpu_offload=None,
            provided_pass_ids=(
                SpecializationPassId.MOE_ROUTING,
                SpecializationPassId.ATTENTION_REPLACEMENT,
                SpecializationPassId.GDS_EXPORT_WEIGHTS,
            ),
        )


class VoxtralSpecializationProvider(SpecializationProvider):
    provider_id = "voxtral-native"
    native_family = NativeFamily.VOXTRAL

    def match(
        self, resolved_model: ResolvedModel, config: RuntimeConfig
    ) -> SpecializationMatch | None:
        del config
        if not matches_architecture(
            resolved_model,
            ("VoxtralForConditionalGeneration", "VoxtralForSpeechSeq2Seq"),
        ):
            return None
        supports_disk_cache = (
            resolved_model.catalog_entry is None
            or resolved_model.catalog_entry.supports_disk_cache
        )
        return build_match(
            resolved_model=resolved_model,
            native_family=self.native_family,
            provider_id=self.provider_id,
            supports_disk_cache=supports_disk_cache,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
        )

    def load(
        self,
        resolved_model: ResolvedModel,
        config: RuntimeConfig,
        stats: Stats | None,
    ) -> OptimizedModelArtifacts:
        model_path = resolved_model_path(resolved_model)
        validate_safe_model_artifacts(model_path)
        module = import_module("ollm.voxtral")
        device = torch.device(config.device)
        setattr(
            module, "loader", DenseWeightsLoader(str(model_path), device=str(device))
        )
        setattr(module, "stats", stats)
        model = cast(
            _CpuOffloadModel,
            load_specialized_model(
                module.MyVoxtralForConditionalGeneration.from_pretrained,
                model_path,
                torch_dtype="auto",
                attn_implementation=get_attention_implementation(),
            ),
        )
        processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=False
        )
        tokenizer = processor.tokenizer
        return OptimizedModelArtifacts(
            model=finalize_model(model, device),
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            stats=stats,
            supports_disk_cache=True,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            print_suppression_modules=(module,),
            create_cache=lambda cache_dir: KVCache(
                cache_dir=str(cache_dir), device=device, stats=stats
            ),
            apply_cpu_offload=lambda layers_num: model.offload_layers_to_cpu(
                layers_num=layers_num
            ),
            apply_gpu_offload=None,
            provided_pass_ids=(),
        )


@lru_cache(maxsize=1)
def build_default_specialization_registry() -> SpecializationRegistry:
    return SpecializationRegistry(
        (
            LlamaSpecializationProvider(),
            Gemma3SpecializationProvider(),
            Qwen3NextSpecializationProvider(),
            GptOssSpecializationProvider(),
            VoxtralSpecializationProvider(),
        )
    )

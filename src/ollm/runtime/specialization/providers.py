import logging
from functools import lru_cache
from importlib import import_module
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer

from ollm.gds_loader import DenseWeightsLoader, GDSWeights, MoEWeightsLoader, SingleDenseWeightsLoader
from ollm.kvcache import KVCache
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.resolver import NativeFamily, ResolvedModel
from ollm.runtime.safety import validate_safe_gds_export_artifacts, validate_safe_model_artifacts
from ollm.runtime.specialization.base import (
    OptimizedModelArtifacts,
    SpecializationMatch,
    SpecializationProvider,
    SpecializationTraits,
)
from ollm.runtime.specialization.passes.base import SpecializationPassId
from ollm.runtime.specialization.registry import SpecializationRegistry
from ollm.utils import Stats

LOGGER = logging.getLogger(__name__)


def _get_attention_implementation() -> str | None:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        LOGGER.warning("flash_attention_2 is not imported. The context length will be limited.")
        return None


def _resolved_model_path(resolved_model: ResolvedModel) -> Path:
    if resolved_model.model_path is None:
        raise ValueError(f"Resolved model path is missing for {resolved_model.reference.raw}")
    model_path = resolved_model.model_path.expanduser().resolve()
    if not model_path.exists() or not model_path.is_dir():
        raise ValueError(f"Resolved model path does not exist: {model_path}")
    return model_path


def _build_match(
    *,
    resolved_model: ResolvedModel,
    native_family: NativeFamily,
    provider_id: str,
    supports_disk_cache: bool,
    supports_cpu_offload: bool,
    supports_gpu_offload: bool,
) -> SpecializationMatch | None:
    if resolved_model.native_family is not native_family:
        return None
    return SpecializationMatch(
        provider_id=provider_id,
        native_family=native_family,
        reason=(
            f"Selected specialization provider '{provider_id}' for native family "
            f"'{native_family.value}' and model reference '{resolved_model.reference.raw}'."
        ),
        traits=SpecializationTraits(
            supports_disk_cache=supports_disk_cache,
            supports_cpu_offload=supports_cpu_offload,
            supports_gpu_offload=supports_gpu_offload,
            details={"native_family": native_family.value},
        ),
    )


def _matches_architecture(resolved_model: ResolvedModel, architectures: tuple[str, ...]) -> bool:
    if resolved_model.architecture is None:
        return False
    return resolved_model.architecture in architectures


def _finalize_model(model: object, device: torch.device) -> object:
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()
    move_method = getattr(model, "to", None)
    if callable(move_method):
        move_method(device)
    return model


def _unsupported_disk_cache_factory(model_reference: str):
    def create_cache(cache_dir: Path) -> None:
        del cache_dir
        LOGGER.info(
            "%s DiskCache is not supported at the moment. Using default DynamicCache instead.",
            model_reference,
        )
        return None

    return create_cache


def _is_sharded_model_dir(model_path: Path) -> bool:
    return any("index.json" in file_path.name for file_path in model_path.iterdir() if file_path.is_file())


class LlamaSpecializationProvider(SpecializationProvider):
    provider_id = "llama-native"
    native_family = NativeFamily.LLAMA

    def match(self, resolved_model: ResolvedModel, config: RuntimeConfig) -> SpecializationMatch | None:
        del config
        if not _matches_architecture(resolved_model, ("LlamaForCausalLM",)):
            return None
        supports_disk_cache = resolved_model.catalog_entry is None or resolved_model.catalog_entry.supports_disk_cache
        return _build_match(
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
        model_path = _resolved_model_path(resolved_model)
        validate_safe_model_artifacts(model_path)
        module = import_module("ollm.llama")
        device = torch.device(config.device)
        if _is_sharded_model_dir(model_path):
            module.loader = DenseWeightsLoader(str(model_path), device=device)
        else:
            module.loader = SingleDenseWeightsLoader(str(model_path), device=device)
        module.stats = stats
        model = module.MyLlamaForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation=_get_attention_implementation(),
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
        return OptimizedModelArtifacts(
            model=_finalize_model(model, device),
            tokenizer=tokenizer,
            processor=None,
            device=device,
            stats=stats,
            supports_disk_cache=True,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            print_suppression_modules=(module,),
            create_cache=lambda cache_dir: KVCache(cache_dir=str(cache_dir), device=device, stats=stats),
            apply_cpu_offload=lambda layers_num: model.offload_layers_to_cpu(layers_num=layers_num),
            apply_gpu_offload=None,
            provided_pass_ids=(SpecializationPassId.MLP_CHUNKING,),
        )


class Gemma3SpecializationProvider(SpecializationProvider):
    provider_id = "gemma3-native"
    native_family = NativeFamily.GEMMA3

    def match(self, resolved_model: ResolvedModel, config: RuntimeConfig) -> SpecializationMatch | None:
        del config
        if not _matches_architecture(
            resolved_model,
            ("Gemma3ForConditionalGeneration", "Gemma3ForCausalLM"),
        ):
            return None
        supports_disk_cache = resolved_model.catalog_entry is None or resolved_model.catalog_entry.supports_disk_cache
        return _build_match(
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
        model_path = _resolved_model_path(resolved_model)
        validate_safe_model_artifacts(model_path)
        module = import_module("ollm.gemma3")
        device = torch.device(config.device)
        module.loader = DenseWeightsLoader(str(model_path), device=device)
        module.stats = stats
        model_class = module.MyGemma3ForConditionalGeneration if config.multimodal else module.MyGemma3ForCausalLM
        model = model_class.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation=_get_attention_implementation(),
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=False)
        return OptimizedModelArtifacts(
            model=_finalize_model(model, device),
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            stats=stats,
            supports_disk_cache=True,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            print_suppression_modules=(module,),
            create_cache=lambda cache_dir: KVCache(cache_dir=str(cache_dir), device=device, stats=stats),
            apply_cpu_offload=lambda layers_num: model.offload_layers_to_cpu(layers_num=layers_num),
            apply_gpu_offload=None,
            provided_pass_ids=(SpecializationPassId.MLP_CHUNKING,),
        )


class Qwen3NextSpecializationProvider(SpecializationProvider):
    provider_id = "qwen3-next-native"
    native_family = NativeFamily.QWEN3_NEXT

    def match(self, resolved_model: ResolvedModel, config: RuntimeConfig) -> SpecializationMatch | None:
        del config
        if not _matches_architecture(
            resolved_model,
            ("Qwen3NextForCausalLM", "Qwen3MoeForCausalLM"),
        ):
            return None
        supports_disk_cache = resolved_model.catalog_entry is None or resolved_model.catalog_entry.supports_disk_cache
        return _build_match(
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
        model_path = _resolved_model_path(resolved_model)
        validate_safe_model_artifacts(model_path)
        module = import_module("ollm.qwen3_next")
        device = torch.device(config.device)
        module.loader = MoEWeightsLoader(str(model_path), device=device)
        module.stats = stats
        model = module.MyQwen3NextForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation=_get_attention_implementation(),
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
        return OptimizedModelArtifacts(
            model=_finalize_model(model, device),
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
            apply_cpu_offload=lambda layers_num: model.offload_layers_to_cpu(layers_num=layers_num),
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

    def match(self, resolved_model: ResolvedModel, config: RuntimeConfig) -> SpecializationMatch | None:
        del config
        if not _matches_architecture(
            resolved_model,
            ("GptOssForCausalLM", "OpenAIGptOssForCausalLM"),
        ):
            return None
        model_path = resolved_model.model_path
        if model_path is None or not (model_path.expanduser().resolve() / "gds_export").exists():
            return None
        try:
            validate_safe_gds_export_artifacts(model_path.expanduser().resolve() / "gds_export")
        except ValueError:
            return None
        return _build_match(
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
        model_path = _resolved_model_path(resolved_model)
        validate_safe_model_artifacts(model_path)
        export_path = model_path / "gds_export"
        if not export_path.exists() or not export_path.is_dir():
            raise ValueError(f"gpt-oss optimized specialization requires a gds_export directory in {model_path}")
        validate_safe_gds_export_artifacts(export_path)
        module = import_module("ollm.gpt_oss")
        device = torch.device(config.device)
        module.loader = GDSWeights(export_path, device=device)
        module.stats = stats
        model = module.MyGptOssForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=False,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
        return OptimizedModelArtifacts(
            model=_finalize_model(model, device),
            tokenizer=tokenizer,
            processor=None,
            device=device,
            stats=stats,
            supports_disk_cache=False,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            print_suppression_modules=(module,),
            create_cache=_unsupported_disk_cache_factory(resolved_model.reference.raw),
            apply_cpu_offload=lambda layers_num: model.offload_layers_to_cpu(layers_num=layers_num),
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

    def match(self, resolved_model: ResolvedModel, config: RuntimeConfig) -> SpecializationMatch | None:
        del config
        if not _matches_architecture(
            resolved_model,
            ("VoxtralForConditionalGeneration", "VoxtralForSpeechSeq2Seq"),
        ):
            return None
        supports_disk_cache = resolved_model.catalog_entry is None or resolved_model.catalog_entry.supports_disk_cache
        return _build_match(
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
        model_path = _resolved_model_path(resolved_model)
        validate_safe_model_artifacts(model_path)
        module = import_module("ollm.voxtral")
        device = torch.device(config.device)
        module.loader = DenseWeightsLoader(str(model_path), device=device)
        module.stats = stats
        model = module.MyVoxtralForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation=_get_attention_implementation(),
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=False)
        tokenizer = processor.tokenizer
        return OptimizedModelArtifacts(
            model=_finalize_model(model, device),
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            stats=stats,
            supports_disk_cache=True,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            print_suppression_modules=(module,),
            create_cache=lambda cache_dir: KVCache(cache_dir=str(cache_dir), device=device, stats=stats),
            apply_cpu_offload=lambda layers_num: model.offload_layers_to_cpu(layers_num=layers_num),
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

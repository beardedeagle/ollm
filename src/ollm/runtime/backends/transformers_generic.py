from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
)

from ollm.inference import get_attn_implementation
from ollm.runtime.backends.base import BackendRuntime, ExecutionBackend
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.safety import (
    validate_safe_adapter_artifacts,
    validate_safe_model_artifacts,
)


class TransformersGenericBackend(ExecutionBackend):
    backend_id = "transformers-generic"

    def __init__(
        self,
        *,
        causal_loader: Callable[..., Any] | None = None,
        image_text_loader: Callable[..., Any] | None = None,
        seq2seq_loader: Callable[..., Any] | None = None,
        tokenizer_loader: Callable[..., Any] | None = None,
        processor_loader: Callable[..., Any] | None = None,
    ):
        self._causal_loader = (
            AutoModelForCausalLM.from_pretrained
            if causal_loader is None
            else causal_loader
        )
        self._image_text_loader = (
            AutoModelForImageTextToText.from_pretrained
            if image_text_loader is None
            else image_text_loader
        )
        self._seq2seq_loader = (
            AutoModelForSeq2SeqLM.from_pretrained
            if seq2seq_loader is None
            else seq2seq_loader
        )
        self._tokenizer_loader = (
            AutoTokenizer.from_pretrained
            if tokenizer_loader is None
            else tokenizer_loader
        )
        self._processor_loader = (
            AutoProcessor.from_pretrained
            if processor_loader is None
            else processor_loader
        )

    def load(self, plan: RuntimePlan, config: RuntimeConfig) -> BackendRuntime:
        if plan.model_path is None or plan.generic_model_kind is None:
            raise ValueError(
                "transformers-generic backend requires a materialized generic model path"
            )

        validate_safe_model_artifacts(plan.model_path)
        model = self._load_model(plan.model_path, plan.generic_model_kind)
        processor = self._load_processor(plan.model_path, plan.generic_model_kind)
        tokenizer = self._load_tokenizer(
            plan.model_path, plan.generic_model_kind, processor
        )
        model.eval()
        device = torch.device(config.device)
        model.to(device)
        if tokenizer is None and processor is not None:
            tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                f"No tokenizer could be loaded for generic model at {plan.model_path}"
            )
        _configure_padding(tokenizer, model)
        adapter_dir = config.resolved_adapter_dir()
        if adapter_dir is not None:
            model = _apply_peft_adapter(model, adapter_dir)
            model.eval()
        return BackendRuntime(
            backend_id=self.backend_id,
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            device=device,
            stats=None,
            print_suppression_modules=(),
            create_cache=lambda cache_dir: None,
            apply_offload=lambda runtime_config: _validate_generic_offload(
                runtime_config
            ),
        )

    def _load_model(self, model_path: Path, generic_model_kind: GenericModelKind):
        base_kwargs = {
            "pretrained_model_name_or_path": str(model_path),
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        attn_implementation = get_attn_implementation()
        if attn_implementation is not None:
            base_kwargs["attn_implementation"] = attn_implementation

        if generic_model_kind is GenericModelKind.CAUSAL_LM:
            return _load_with_fallbacks(self._causal_loader, base_kwargs)
        if generic_model_kind is GenericModelKind.IMAGE_TEXT_TO_TEXT:
            return _load_with_fallbacks(self._image_text_loader, base_kwargs)
        if generic_model_kind is GenericModelKind.SEQ2SEQ_LM:
            return _load_with_fallbacks(self._seq2seq_loader, base_kwargs)
        raise ValueError(f"Unsupported generic model kind: {generic_model_kind.value}")

    def _load_tokenizer(
        self, model_path: Path, generic_model_kind: GenericModelKind, processor
    ):
        if generic_model_kind is GenericModelKind.IMAGE_TEXT_TO_TEXT:
            if (
                processor is not None
                and getattr(processor, "tokenizer", None) is not None
            ):
                return processor.tokenizer
        return self._tokenizer_loader(str(model_path), trust_remote_code=False)

    def _load_processor(self, model_path: Path, generic_model_kind: GenericModelKind):
        if generic_model_kind is not GenericModelKind.IMAGE_TEXT_TO_TEXT:
            return None
        return self._processor_loader(str(model_path), trust_remote_code=False)


def _load_with_fallbacks(loader: Callable[..., Any], base_kwargs: dict[str, Any]):
    loader_kwargs = dict(base_kwargs)
    try:
        return loader(**loader_kwargs)
    except (TypeError, ValueError) as exc:
        if "attn_implementation" in loader_kwargs and "attn_implementation" in str(exc):
            loader_kwargs.pop("attn_implementation")
            return _load_with_fallbacks(loader, loader_kwargs)
        if "low_cpu_mem_usage" in loader_kwargs and "low_cpu_mem_usage" in str(exc):
            loader_kwargs.pop("low_cpu_mem_usage")
            return _load_with_fallbacks(loader, loader_kwargs)
        raise


def _configure_padding(tokenizer, model) -> None:
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    if (
        getattr(generation_config, "pad_token_id", None) is None
        and getattr(tokenizer, "pad_token_id", None) is not None
    ):
        generation_config.pad_token_id = tokenizer.pad_token_id


def _apply_peft_adapter(model, adapter_dir: Path):
    from peft import LoraConfig, get_peft_model

    validate_safe_adapter_artifacts(adapter_dir)

    peft_config = LoraConfig.from_pretrained(str(adapter_dir))
    adapted_model = get_peft_model(model, peft_config)
    adapted_model.load_adapter(
        str(adapter_dir), adapter_name="default", use_safetensors=True
    )
    return adapted_model


def _validate_generic_offload(config: RuntimeConfig) -> None:
    if config.offload_cpu_layers > 0 or config.offload_gpu_layers > 0:
        raise ValueError(
            "The transformers-generic backend does not support custom layer offload controls"
        )

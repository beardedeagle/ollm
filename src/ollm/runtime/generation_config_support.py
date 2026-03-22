"""Helpers for request-scoped generation-config normalization."""

from contextlib import contextmanager

from transformers import GenerationConfig as TransformersGenerationConfig

from ollm.app.types import PromptRequest
from ollm.runtime.loader import LoadedRuntime


def normalized_generation_config(
    runtime: LoadedRuntime,
    request: PromptRequest,
) -> TransformersGenerationConfig:
    base_generation_config = getattr(runtime.model, "generation_config", None)
    if isinstance(base_generation_config, TransformersGenerationConfig):
        normalized = TransformersGenerationConfig.from_dict(
            base_generation_config.to_dict()
        )
    else:
        model_config = getattr(runtime.model, "config", None)
        if model_config is not None:
            normalized = TransformersGenerationConfig.from_model_config(model_config)
        else:
            normalized = TransformersGenerationConfig()

    request_generation_config = request.generation_config
    normalized.max_new_tokens = request_generation_config.max_new_tokens
    normalized.max_length = None
    normalized.use_cache = True
    if request_generation_config.sampling_enabled():
        normalized.do_sample = True
        normalized.temperature = request_generation_config.temperature
        if request_generation_config.top_p is not None:
            normalized.top_p = request_generation_config.top_p
        if request_generation_config.top_k is not None:
            normalized.top_k = request_generation_config.top_k
    else:
        normalized.do_sample = False
        clear_sampling_fields(normalized)
    return normalized


def clear_sampling_fields(generation_config: TransformersGenerationConfig) -> None:
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 50


@contextmanager
def temporary_generation_config(model: object, generation_config):
    original = getattr(model, "generation_config", None)
    setattr(model, "generation_config", generation_config)
    try:
        yield
    finally:
        if original is None:
            delattr(model, "generation_config")
        else:
            setattr(model, "generation_config", original)

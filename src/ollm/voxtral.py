# voxtral-small-24B

import time
from typing import Protocol, Unpack, cast

import torch
from transformers.models.voxtral.modeling_voxtral import VoxtralForConditionalGeneration

from . import llama
from .utils import _assign_tensor_to_module, _set_meta_placeholder, _walk_to_parent


class _LoaderProtocol(Protocol):
    manifest: dict[str, dict[str, str]]

    def preload_layer_safetensors(self, base: str) -> None: ...
    def load_dict_to_cuda(self, base: str) -> dict[str, torch.Tensor]: ...
    def offload_dict_to_gpu_cpu(self, base: str, gpu: bool = False) -> None: ...


class _LayerLoaderContext(Protocol):
    layer_idx: int


class _OffloadProtocol(Protocol):
    num_hidden_layers: int


loader: _LoaderProtocol | None = None
stats = None


def _require_loader() -> _LoaderProtocol:
    if loader is None:
        raise RuntimeError("voxtral loader has not been initialized")
    return loader


def _record_stats(name: str, started_at: float) -> None:
    if stats is not None:
        stats.set(name, started_at)


class loaderLayer:
    def _load_layer_weights(self: _LayerLoaderContext) -> None:
        started_at = time.perf_counter()
        base = f"language_model.model.layers.{self.layer_idx}."
        current_loader = _require_loader()
        current_loader.preload_layer_safetensors(base)
        for attr_path, tensor in current_loader.load_dict_to_cuda(base).items():
            parent, leaf = _walk_to_parent(self, attr_path)
            _assign_tensor_to_module(parent, leaf, tensor)
        _record_stats("layer_load", started_at)

    def _unload_layer_weights(self: _LayerLoaderContext) -> None:
        base = f"language_model.model.layers.{self.layer_idx}."
        current_loader = _require_loader()
        for attr_path in current_loader.manifest[base]:
            parent, leaf = _walk_to_parent(self, attr_path)
            _set_meta_placeholder(parent, leaf)


class MyLlamaDecoderLayer(llama.LlamaDecoderLayer, loaderLayer):
    def __init__(self, config: object, layer_idx: int):
        self.layer_idx = layer_idx
        super().__init__(cast(llama.LlamaConfig, config), layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: llama.Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[llama.TransformersKwargs],
    ) -> torch.Tensor:
        self._load_layer_weights()
        result = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        self._unload_layer_weights()
        return result


setattr(llama, "MyLlamaDecoderLayer", MyLlamaDecoderLayer)


class oForGeneration:
    def offload_layers_to_cpu(self: _OffloadProtocol, layers_num: int = 2) -> None:
        print(f"offloading layers to CPU {layers_num}/{self.num_hidden_layers}...")
        current_loader = _require_loader()
        for layer_idx in range(min(layers_num, self.num_hidden_layers)):
            base = f"language_model.model.layers.{layer_idx}."
            current_loader.preload_layer_safetensors(base)
            current_loader.offload_dict_to_gpu_cpu(base, gpu=False)
        print(
            f"./finished offloading layers to CPU {layers_num}/{self.num_hidden_layers}"
        )


class MyVoxtralForConditionalGeneration(
    VoxtralForConditionalGeneration, oForGeneration
):
    def __init__(self, config: object):
        super().__init__(config)
        self.num_hidden_layers = config.text_config.num_hidden_layers  # type: ignore[attr-defined]
        self.language_model = llama.MyLlamaForCausalLM(config.text_config)  # type: ignore[attr-defined]
        llama.stats = stats

import time
from typing import Protocol, Unpack, cast

import torch
import transformers.models.gemma3.modeling_gemma3 as gemma3_modeling
from torch import nn
from transformers.models.gemma3.configuration_gemma3 import (
    Gemma3Config,
    Gemma3TextConfig,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Cache,
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3MLP,
    Gemma3Model,
    Gemma3TextModel,
    TransformersKwargs,
)

from ollm.device_staging import attach_parent_lm_head
from ollm.utils import _assign_tensor_to_module, _set_meta_placeholder, _walk_to_parent


class _LoaderProtocol(Protocol):
    manifest: dict[str, dict[str, str]]

    def prefetch_layer_weights(self, base: str) -> None: ...
    def preload_layer_safetensors(self, base: str) -> None: ...
    def load_dict_to_cuda(self, base: str) -> dict[str, torch.Tensor]: ...
    def offload_dict_to_gpu_cpu(self, base: str, gpu: bool = False) -> None: ...


class _LayerLoaderContext(Protocol):
    layer_idx: int

    def get_base(self, base: str) -> str: ...
    def _layer_weight_base(self) -> str: ...
    def _next_layer_weight_base(
        self, current_loader: _LoaderProtocol
    ) -> str | None: ...


class _OffloadProtocol(Protocol):
    num_hidden_layers: int

    def offload_layers_to_cpu_indices(self, layer_indices: tuple[int, ...]) -> None: ...

    def get_base(self, base: str) -> str: ...


loader: _LoaderProtocol | None = None
stats = None


def _coerce_hidden_layer_count(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("Gemma3 hidden layer count must be an integer")
    return value


def _require_loader() -> _LoaderProtocol:
    if loader is None:
        raise RuntimeError("gemma3 loader has not been initialized")
    return loader


def _record_stats(name: str, started_at: float) -> None:
    if stats is not None:
        stats.set(name, started_at)


def _unwrap_base_layer(parent: object) -> object:
    if hasattr(parent, "base_layer"):
        return getattr(parent, "base_layer")
    return parent


def _coerce_hidden_layer_count(value: object) -> int:
    if not isinstance(value, int):
        raise RuntimeError(
            "Expected transformer config to expose an integer hidden layer count"
        )
    return value


class loaderLayer:
    def get_base(self, base: str) -> str:
        current_loader = _require_loader()
        if base in current_loader.manifest:
            return base
        return f"language_model.{base}"

    def _layer_weight_base(self: _LayerLoaderContext) -> str:
        return self.get_base(f"model.layers.{self.layer_idx}.")

    def _next_layer_weight_base(
        self: _LayerLoaderContext, current_loader: _LoaderProtocol
    ) -> str | None:
        next_base = self.get_base(f"model.layers.{self.layer_idx + 1}.")
        if next_base not in current_loader.manifest:
            return None
        return next_base

    def _load_layer_weights(self: _LayerLoaderContext) -> None:
        started_at = time.perf_counter()
        current_loader = _require_loader()
        base = self._layer_weight_base()
        current_loader.preload_layer_safetensors(base)
        for attr_path, tensor in current_loader.load_dict_to_cuda(base).items():
            parent, leaf = _walk_to_parent(self, attr_path)
            _assign_tensor_to_module(_unwrap_base_layer(parent), leaf, tensor)
        next_base = self._next_layer_weight_base(current_loader)
        if next_base is not None:
            current_loader.prefetch_layer_weights(next_base)
        _record_stats("layer_load", started_at)

    def _unload_layer_weights(self: _LayerLoaderContext) -> None:
        base = self._layer_weight_base()
        current_loader = _require_loader()
        for attr_path in current_loader.manifest[base]:
            parent, leaf = _walk_to_parent(self, attr_path)
            _set_meta_placeholder(_unwrap_base_layer(parent), leaf)


class MyGemma3MLP(Gemma3MLP, loaderLayer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = 16384
        chunks: list[torch.Tensor] = []
        squeezed = x.squeeze(0)
        for index in range(0, squeezed.shape[0], chunk_size):
            chunk = squeezed[index : index + chunk_size]
            gate_chunk = self.act_fn(self.gate_proj(chunk))
            up_chunk = self.up_proj(chunk)
            chunks.append(self.down_proj(gate_chunk * up_chunk))
        return torch.cat(chunks, dim=0).unsqueeze(0)


class MyGemma3DecoderLayer(Gemma3DecoderLayer, loaderLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        self._load_layer_weights()
        if position_embeddings is None:
            raise RuntimeError("Gemma3 decoder layers require position embeddings")
        result = super().forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        self._unload_layer_weights()
        return result


class MyGemma3TextModel(Gemma3TextModel):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList()
        hidden_layer_count = _coerce_hidden_layer_count(config.num_hidden_layers)
        for layer_idx in range(hidden_layer_count):
            decoder_layer = MyGemma3DecoderLayer(config, layer_idx)
            decoder_layer._unload_layer_weights()
            self.layers.append(decoder_layer)


class MyGemma3Model(Gemma3Model):
    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        text_config = cast(Gemma3TextConfig, config.text_config)
        self.language_model = MyGemma3TextModel(text_config)


setattr(gemma3_modeling, "Gemma3MLP", MyGemma3MLP)
setattr(gemma3_modeling, "Gemma3TextModel", MyGemma3TextModel)
setattr(gemma3_modeling, "Gemma3Model", MyGemma3Model)


class oForGeneration(loaderLayer):
    def offload_layers_to_cpu_indices(
        self: _OffloadProtocol, layer_indices: tuple[int, ...]
    ) -> None:
        print(
            f"offloading layers to CPU {len(layer_indices)}/{self.num_hidden_layers}..."
        )
        current_loader = _require_loader()
        for layer_idx in layer_indices:
            base = self.get_base(f"model.layers.{layer_idx}.")
            current_loader.preload_layer_safetensors(base)
            current_loader.offload_dict_to_gpu_cpu(base, gpu=False)
        print(
            "./finished offloading layers to CPU "
            f"{len(layer_indices)}/{self.num_hidden_layers}"
        )

    def offload_layers_to_cpu(self: _OffloadProtocol, layers_num: int = 2) -> None:
        self.offload_layers_to_cpu_indices(
            tuple(range(min(layers_num, self.num_hidden_layers)))
        )


class MyGemma3ForCausalLM(Gemma3ForCausalLM, oForGeneration):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        attach_parent_lm_head(self.model, self.lm_head)
        self.num_hidden_layers = _coerce_hidden_layer_count(config.num_hidden_layers)


class MyGemma3ForConditionalGeneration(Gemma3ForConditionalGeneration, oForGeneration):
    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.num_hidden_layers = _coerce_hidden_layer_count(
            cast(Gemma3TextConfig, config.text_config).num_hidden_layers
        )

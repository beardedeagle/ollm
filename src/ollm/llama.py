# llama3-1B/3B/8B-chat

import time
from typing import Protocol, Unpack, cast

import torch
import transformers.models.llama.modeling_llama as llama_modeling
from torch import nn
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    Cache,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    TransformersKwargs,
    create_causal_mask,
)

from ollm.device_staging import (
    restore_static_modules_after_forward,
    stage_static_modules_on_host,
)
from ollm.utils import _assign_tensor_to_module, _set_meta_placeholder, _walk_to_parent


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


def _coerce_hidden_layer_count(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("Llama hidden layer count must be an integer")
    return value


def _require_loader() -> _LoaderProtocol:
    if loader is None:
        raise RuntimeError("llama loader has not been initialized")
    return loader


def _record_stats(name: str, started_at: float) -> None:
    if stats is not None:
        stats.set(name, started_at)


def _unwrap_base_layer(parent: object) -> object:
    if hasattr(parent, "base_layer"):
        return getattr(parent, "base_layer")
    return parent


class loaderLayer:
    def _load_layer_weights(self: _LayerLoaderContext) -> None:
        started_at = time.perf_counter()
        base = f"model.layers.{self.layer_idx}."
        current_loader = _require_loader()
        current_loader.preload_layer_safetensors(base)
        for attr_path, tensor in current_loader.load_dict_to_cuda(base).items():
            parent, leaf = _walk_to_parent(self, attr_path)
            _assign_tensor_to_module(_unwrap_base_layer(parent), leaf, tensor)
        _record_stats("layer_load", started_at)

    def _unload_layer_weights(self: _LayerLoaderContext) -> None:
        base = f"model.layers.{self.layer_idx}."
        current_loader = _require_loader()
        for attr_path in current_loader.manifest[base]:
            parent, leaf = _walk_to_parent(self, attr_path)
            _set_meta_placeholder(_unwrap_base_layer(parent), leaf)


class MyLlamaMLP(LlamaMLP):
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


class MyLlamaDecoderLayer(LlamaDecoderLayer, loaderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        self.layer_idx = layer_idx
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
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


class MyLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList()
        hidden_layer_count = _coerce_hidden_layer_count(config.num_hidden_layers)
        for layer_idx in range(hidden_layer_count):
            decoder_layer = MyLlamaDecoderLayer(config, layer_idx)
            decoder_layer._unload_layer_weights()
            self.layers.append(decoder_layer)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = cast(
                torch.LongTensor,
                torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                ),
            )

        if position_ids is None:
            position_ids = cast(torch.LongTensor, cache_position.unsqueeze(0))

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        stage_static_modules_on_host(
            self.embed_tokens,
            self.parent_lm_head,
            hidden_states.device,
        )
        decoder_layers = list(self.layers.children())[
            : _coerce_hidden_layer_count(self.config.num_hidden_layers)
        ]
        for decoder_layer in decoder_layers:
            assert isinstance(decoder_layer, MyLlamaDecoderLayer)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        restore_static_modules_after_forward(
            self.embed_tokens,
            self.parent_lm_head,
            hidden_states.device,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values
        )


setattr(llama_modeling, "LlamaMLP", MyLlamaMLP)
setattr(llama_modeling, "LlamaDecoderLayer", MyLlamaDecoderLayer)
setattr(llama_modeling, "LlamaModel", MyLlamaModel)


class oForGeneration:
    def offload_layers_to_cpu(self: _OffloadProtocol, layers_num: int = 2) -> None:
        print(f"offloading layers to CPU {layers_num}/{self.num_hidden_layers}...")
        current_loader = _require_loader()
        for layer_idx in range(min(layers_num, self.num_hidden_layers)):
            base = f"model.layers.{layer_idx}."
            current_loader.preload_layer_safetensors(base)
            current_loader.offload_dict_to_gpu_cpu(base, gpu=False)
        print(
            f"./finished offloading layers to CPU {layers_num}/{self.num_hidden_layers}"
        )


class MyLlamaForCausalLM(LlamaForCausalLM, oForGeneration):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model.parent_lm_head = self.lm_head
        self.num_hidden_layers = _coerce_hidden_layer_count(config.num_hidden_layers)

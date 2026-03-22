import importlib
from typing import Self

import torch
import transformers.models.llama.modeling_llama as llama_modeling
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

import ollm.llama as optimized_llama
from ollm.device_staging import (
    attach_parent_lm_head,
    restore_static_modules_after_forward,
    stage_static_modules_on_host,
)


class RecordingModule:
    def __init__(self) -> None:
        self.cpu_calls = 0
        self.to_calls: list[torch.device] = []

    def cpu(self) -> Self:
        self.cpu_calls += 1
        return self

    def to(self, device: torch.device) -> Self:
        self.to_calls.append(device)
        return self


def test_llama_static_modules_stay_on_accelerators() -> None:
    embed_tokens = RecordingModule()
    lm_head = RecordingModule()
    execution_device = torch.device("mps")

    stage_static_modules_on_host(embed_tokens, lm_head, execution_device)
    restore_static_modules_after_forward(embed_tokens, lm_head, execution_device)

    assert embed_tokens.cpu_calls == 0
    assert lm_head.cpu_calls == 0
    assert embed_tokens.to_calls == []
    assert lm_head.to_calls == []


def test_llama_static_modules_use_host_path_on_cpu() -> None:
    embed_tokens = RecordingModule()
    lm_head = RecordingModule()
    execution_device = torch.device("cpu")

    stage_static_modules_on_host(embed_tokens, lm_head, execution_device)
    restore_static_modules_after_forward(embed_tokens, lm_head, execution_device)

    assert embed_tokens.cpu_calls == 1
    assert lm_head.cpu_calls == 1
    assert embed_tokens.to_calls == [execution_device]
    assert lm_head.to_calls == [execution_device]


class _FakeLoader:
    def __init__(self, hidden_layer_count: int) -> None:
        self.manifest = {
            f"model.layers.{layer_idx}.": {} for layer_idx in range(hidden_layer_count)
        }

    def preload_layer_safetensors(self, base: str) -> None:
        del base

    def load_dict_to_cuda(self, base: str) -> dict[str, torch.Tensor]:
        del base
        return {}

    def offload_dict_to_gpu_cpu(self, base: str, gpu: bool = False) -> None:
        del base, gpu


def _build_tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


def test_importing_optimized_llama_does_not_mutate_transformers_llama_classes() -> None:
    original_llama_mlp = llama_modeling.LlamaMLP
    original_llama_decoder_layer = llama_modeling.LlamaDecoderLayer
    original_llama_model = llama_modeling.LlamaModel

    importlib.reload(optimized_llama)

    assert llama_modeling.LlamaMLP is original_llama_mlp
    assert llama_modeling.LlamaDecoderLayer is original_llama_decoder_layer
    assert llama_modeling.LlamaModel is original_llama_model


def test_optimized_llama_initialization_restores_transformers_llama_classes(
    monkeypatch,
) -> None:
    original_llama_mlp = llama_modeling.LlamaMLP
    original_llama_decoder_layer = llama_modeling.LlamaDecoderLayer
    original_llama_model = llama_modeling.LlamaModel
    config = _build_tiny_llama_config()
    hidden_layer_count = config.num_hidden_layers
    if not isinstance(hidden_layer_count, int):
        raise AssertionError(
            "Expected tiny Llama config to expose an integer layer count"
        )
    monkeypatch.setattr(
        optimized_llama,
        "loader",
        _FakeLoader(hidden_layer_count),
    )

    model = optimized_llama.MyLlamaForCausalLM(config)

    assert model.model.parent_lm_head is model.lm_head
    assert "parent_lm_head" not in model.model._modules
    assert "model.parent_lm_head.weight" not in model.state_dict()
    assert llama_modeling.LlamaMLP is original_llama_mlp
    assert llama_modeling.LlamaDecoderLayer is original_llama_decoder_layer
    assert llama_modeling.LlamaModel is original_llama_model


def test_attach_parent_lm_head_does_not_register_duplicate_module() -> None:
    parent = torch.nn.Module()
    lm_head = torch.nn.Linear(4, 4, bias=False)

    attach_parent_lm_head(parent, lm_head)

    assert parent.parent_lm_head is lm_head
    assert "parent_lm_head" not in parent._modules


def test_temporary_llama_patch_uses_current_module_overrides(monkeypatch) -> None:
    original_llama_decoder_layer = llama_modeling.LlamaDecoderLayer

    class TemporaryDecoderLayer:
        pass

    monkeypatch.setattr(
        optimized_llama,
        "MyLlamaDecoderLayer",
        TemporaryDecoderLayer,
    )

    with optimized_llama._temporary_llama_modeling_patch():
        assert llama_modeling.LlamaDecoderLayer is TemporaryDecoderLayer

    assert llama_modeling.LlamaDecoderLayer is original_llama_decoder_layer


def test_generic_llama_auto_model_stays_on_transformers_classes_after_optimized_import() -> (
    None
):
    importlib.reload(optimized_llama)
    generic_model = AutoModelForCausalLM.from_config(_build_tiny_llama_config())

    assert type(generic_model.model) is llama_modeling.LlamaModel
    assert not isinstance(generic_model.model, optimized_llama.MyLlamaModel)

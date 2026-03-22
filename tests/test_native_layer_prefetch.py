import torch

import ollm.gemma3 as optimized_gemma3
import ollm.llama as optimized_llama
import ollm.voxtral as optimized_voxtral


class _RecordingLoader:
    def __init__(self, manifest: dict[str, dict[str, str]]) -> None:
        self.manifest = manifest
        self.prefetch_calls: list[str] = []
        self.load_calls: list[str] = []

    def preload_layer_safetensors(self, base: str) -> None:
        self.prefetch_layer_weights(base)

    def prefetch_layer_weights(self, base: str) -> None:
        self.prefetch_calls.append(base)

    def load_dict_to_cuda(self, base: str) -> dict[str, torch.Tensor]:
        self.load_calls.append(base)
        return {}

    def offload_dict_to_gpu_cpu(self, base: str, gpu: bool = False) -> None:
        del base, gpu


class _FakeLlamaLayer(optimized_llama.loaderLayer):
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx


class _FakeGemmaLayer(optimized_gemma3.loaderLayer):
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx


class _FakeVoxtralLayer(optimized_voxtral.loaderLayer):
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx


def test_llama_layer_load_prefetches_next_layer(monkeypatch) -> None:
    loader = _RecordingLoader(
        {
            "model.layers.0.": {},
            "model.layers.1.": {},
        }
    )
    monkeypatch.setattr(optimized_llama, "loader", loader)
    monkeypatch.setattr(optimized_llama, "stats", None)

    _FakeLlamaLayer(0)._load_layer_weights()

    assert loader.prefetch_calls == [
        "model.layers.0.",
        "model.layers.1.",
    ]
    assert loader.load_calls == ["model.layers.0."]


def test_gemma_layer_load_prefetches_next_language_model_layer(monkeypatch) -> None:
    loader = _RecordingLoader(
        {
            "language_model.model.layers.0.": {},
            "language_model.model.layers.1.": {},
        }
    )
    monkeypatch.setattr(optimized_gemma3, "loader", loader)
    monkeypatch.setattr(optimized_gemma3, "stats", None)

    _FakeGemmaLayer(0)._load_layer_weights()

    assert loader.prefetch_calls == [
        "language_model.model.layers.0.",
        "language_model.model.layers.1.",
    ]
    assert loader.load_calls == ["language_model.model.layers.0."]


def test_voxtral_layer_load_prefetches_next_language_model_layer(monkeypatch) -> None:
    loader = _RecordingLoader(
        {
            "language_model.model.layers.0.": {},
            "language_model.model.layers.1.": {},
        }
    )
    monkeypatch.setattr(optimized_voxtral, "loader", loader)
    monkeypatch.setattr(optimized_voxtral, "stats", None)

    _FakeVoxtralLayer(0)._load_layer_weights()

    assert loader.prefetch_calls == [
        "language_model.model.layers.0.",
        "language_model.model.layers.1.",
    ]
    assert loader.load_calls == ["language_model.model.layers.0."]

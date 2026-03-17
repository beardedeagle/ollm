import torch

from ollm.llama import (
    _restore_static_modules_after_forward,
    _stage_static_modules_on_host,
)


class RecordingModule:
    def __init__(self) -> None:
        self.cpu_calls = 0
        self.to_calls: list[torch.device] = []

    def cpu(self) -> "RecordingModule":
        self.cpu_calls += 1
        return self

    def to(self, device: torch.device) -> "RecordingModule":
        self.to_calls.append(device)
        return self


def test_llama_static_modules_stay_on_accelerators() -> None:
    embed_tokens = RecordingModule()
    lm_head = RecordingModule()
    execution_device = torch.device("mps")

    _stage_static_modules_on_host(embed_tokens, lm_head, execution_device)
    _restore_static_modules_after_forward(embed_tokens, lm_head, execution_device)

    assert embed_tokens.cpu_calls == 0
    assert lm_head.cpu_calls == 0
    assert embed_tokens.to_calls == []
    assert lm_head.to_calls == []


def test_llama_static_modules_use_host_path_on_cpu() -> None:
    embed_tokens = RecordingModule()
    lm_head = RecordingModule()
    execution_device = torch.device("cpu")

    _stage_static_modules_on_host(embed_tokens, lm_head, execution_device)
    _restore_static_modules_after_forward(embed_tokens, lm_head, execution_device)

    assert embed_tokens.cpu_calls == 1
    assert lm_head.cpu_calls == 1
    assert embed_tokens.to_calls == [execution_device]
    assert lm_head.to_calls == [execution_device]

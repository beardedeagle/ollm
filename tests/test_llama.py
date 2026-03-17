from typing import Self

import torch

from ollm.device_staging import (
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

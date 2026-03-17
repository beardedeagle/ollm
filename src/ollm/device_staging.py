"""Helpers for staging static runtime modules on the host only when required."""

from typing import Protocol

import torch


class DeviceStagingTarget(Protocol):
    def cpu(self) -> object: ...
    def to(self, device: torch.device) -> object: ...


def stage_static_modules_on_host(
    embed_tokens: DeviceStagingTarget,
    lm_head: DeviceStagingTarget,
    device: torch.device,
) -> None:
    if device.type != "cpu":
        return
    embed_tokens.cpu()
    lm_head.cpu()


def restore_static_modules_after_forward(
    embed_tokens: DeviceStagingTarget,
    lm_head: DeviceStagingTarget,
    device: torch.device,
) -> None:
    if device.type != "cpu":
        return
    embed_tokens.to(device)
    lm_head.to(device)

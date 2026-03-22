"""Helpers for staging static runtime modules on the host only when required."""

from typing import Protocol

import torch
from torch import nn


class DeviceStagingTarget(Protocol):
    def cpu(self) -> object: ...
    def to(self, device: torch.device) -> object: ...


def attach_parent_lm_head(model: nn.Module, lm_head: nn.Module) -> None:
    """Attach an output-head reference without registering a duplicate submodule."""

    model.__dict__["parent_lm_head"] = lm_head


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

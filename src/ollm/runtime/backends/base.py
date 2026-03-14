from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan


@dataclass(slots=True)
class BackendRuntime:
    backend_id: str
    model: object
    tokenizer: object | None
    processor: object | None
    device: torch.device
    stats: object | None
    create_cache: Callable[[Path], object | None]
    apply_offload: Callable[[RuntimeConfig], None]


class ExecutionBackend(ABC):
    backend_id: str

    @abstractmethod
    def load(self, plan: RuntimePlan, config: RuntimeConfig) -> BackendRuntime:
        raise NotImplementedError

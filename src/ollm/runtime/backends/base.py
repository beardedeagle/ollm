from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.specialization.base import AppliedSpecialization


@dataclass(slots=True)
class BackendRuntime:
    backend_id: str
    model: object
    tokenizer: object | None
    processor: object | None
    device: torch.device
    stats: object | None
    print_suppression_modules: tuple[ModuleType, ...]
    create_cache: Callable[[Path], object | None]
    apply_offload: Callable[[RuntimeConfig], None]
    applied_specialization: AppliedSpecialization | None = None


class ExecutionBackend(ABC):
    backend_id: str

    @abstractmethod
    def load(self, plan: RuntimePlan, config: RuntimeConfig) -> BackendRuntime:
        raise NotImplementedError

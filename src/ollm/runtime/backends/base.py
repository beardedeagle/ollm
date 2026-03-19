from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from ollm.app.types import PromptRequest, PromptResponse
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.specialization.base import AppliedSpecialization
from ollm.runtime.streaming import StreamSink


@dataclass(slots=True)
class BackendRuntime:
    backend_id: str
    model: object | None
    tokenizer: object | None
    processor: object | None
    device: torch.device
    stats: object | None
    print_suppression_modules: tuple[ModuleType, ...]
    create_cache: Callable[[Path, str | None], object | None]
    apply_offload: Callable[[RuntimeConfig], None]
    validate_request: Callable[[PromptRequest], None] | None = None
    execute_prompt: Callable[[PromptRequest, StreamSink], PromptResponse] | None = None
    allows_multimodal_without_processor: bool = False
    applied_specialization: AppliedSpecialization | None = None


class ExecutionBackend(ABC):
    backend_id: str

    def refine_plan(self, plan: RuntimePlan, config: RuntimeConfig) -> RuntimePlan:
        del config
        return plan

    @abstractmethod
    def load(self, plan: RuntimePlan, config: RuntimeConfig) -> BackendRuntime:
        raise NotImplementedError

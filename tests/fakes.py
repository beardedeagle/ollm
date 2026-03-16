from dataclasses import dataclass
from pathlib import Path

from ollm.app.doctor import DoctorCheck, DoctorReport
from ollm.app.types import Message, PromptResponse
from ollm.runtime.capabilities import SupportLevel
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.config import RuntimeConfig
from ollm.runtime.plan import RuntimePlan, SpecializationState
from ollm.runtime.resolver import ModelResolver, ResolvedModel
from ollm.runtime.specialization.passes.base import SpecializationPassId


@dataclass(slots=True)
class FakeLoadedRuntime:
    config: RuntimeConfig


class FakeRuntimeLoader:
    def __init__(self):
        self.load_calls: list[str] = []
        self.loaded_configs: list[RuntimeConfig] = []
        self.download_calls: list[tuple[str, Path, bool]] = []
        self.plan_calls: list[RuntimeConfig] = []
        self._resolver = ModelResolver()

    def load(self, config: RuntimeConfig) -> FakeLoadedRuntime:
        self.load_calls.append(config.model_reference)
        self.loaded_configs.append(config)
        return FakeLoadedRuntime(config=config)

    def download(
        self, model_reference: str, models_dir: Path, force_download: bool = False
    ) -> Path:
        self.download_calls.append((model_reference, models_dir, force_download))
        target = models_dir / model_reference.replace("/", "--")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def resolve(self, model_reference: str, models_dir: Path) -> ResolvedModel:
        return self._resolver.resolve(model_reference, models_dir)

    def discover_local_models(self, models_dir: Path) -> tuple[ResolvedModel, ...]:
        return self._resolver.discover_local_models(models_dir)

    def plan(self, config: RuntimeConfig) -> RuntimePlan:
        self.plan_calls.append(config)
        resolved_model = self.resolve(
            config.model_reference, config.resolved_models_dir()
        )
        return RuntimePlan(
            resolved_model=resolved_model,
            backend_id=config.resolved_backend() or "optimized-native",
            model_path=resolved_model.model_path,
            support_level=SupportLevel.GENERIC
            if not config.use_specialization
            else SupportLevel.OPTIMIZED,
            generic_model_kind=resolved_model.generic_model_kind
            or GenericModelKind.CAUSAL_LM,
            supports_disk_cache=True,
            supports_cpu_offload=True,
            supports_gpu_offload=False,
            specialization_enabled=config.use_specialization,
            specialization_applied=False,
            specialization_provider_id="fake-provider"
            if config.use_specialization
            else None,
            specialization_state=SpecializationState.PLANNED
            if config.use_specialization
            else SpecializationState.NOT_PLANNED,
            reason="fake planned specialization"
            if config.use_specialization
            else "fake generic plan",
            specialization_pass_ids=(SpecializationPassId.DISK_CACHE,)
            if config.use_specialization
            else (),
            details={"backend_override": config.resolved_backend() or ""},
        )


class FakeRuntimeExecutor:
    def __init__(self):
        self.prompts: list[str] = []

    def execute(self, runtime, request, sink=None) -> PromptResponse:
        del runtime
        prompt_text = request.messages[-1].text_content()
        self.prompts.append(prompt_text)
        text = f"echo:{prompt_text}"
        if sink is not None:
            sink.on_status("fake")
            sink.on_text(text)
            sink.on_complete(text)
        return PromptResponse(
            text=text, assistant_message=Message.assistant_text(text), metadata={}
        )


class FakeDoctorService:
    def run(
        self,
        runtime_config,
        include_imports=True,
        include_runtime=True,
        include_paths=True,
        include_download=False,
    ):
        del (
            runtime_config,
            include_imports,
            include_runtime,
            include_paths,
            include_download,
        )
        return DoctorReport([DoctorCheck(name="doctor:fake", ok=True, message="ok")])

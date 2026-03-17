"""Transport-agnostic application service layer shared by CLI and server flows."""

from dataclasses import dataclass, field
from pathlib import Path

from ollm.app.doctor import DoctorReport, DoctorService
from ollm.app.session import ChatSession
from ollm.app.types import ContentPart, Message, PromptResponse
from ollm.client import RuntimeClient
from ollm.runtime.config import DEFAULT_SYSTEM_PROMPT, GenerationConfig, RuntimeConfig
from ollm.runtime.inspection import PlanJsonPayload
from ollm.runtime.loader import RuntimeLoader
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.resolver import ModelResolver, ResolvedModel
from ollm.runtime.streaming import StreamSink


@dataclass(slots=True)
class ApplicationService:
    """Shared control-plane service used by CLI and future server transports."""

    runtime_client: RuntimeClient = field(default_factory=RuntimeClient)
    doctor_service: DoctorService | None = None

    def __post_init__(self) -> None:
        if self.doctor_service is None:
            self.doctor_service = DoctorService(
                runtime_loader=self.runtime_client.runtime_loader
            )

    def resolve_model(self, model_reference: str, models_dir: Path) -> ResolvedModel:
        """Resolve a model reference without loading a runtime."""
        return self.runtime_client.resolve(model_reference, models_dir)

    def discover_local_models(self, models_dir: Path) -> tuple[ResolvedModel, ...]:
        """Discover local materialized models under a models directory."""
        return self.runtime_client.discover_local_models(models_dir)

    def plan(self, runtime_config: RuntimeConfig) -> RuntimePlan:
        """Build a runtime plan without loading a backend."""
        return self.runtime_client.plan(runtime_config)

    def describe_plan(self, runtime_config: RuntimeConfig) -> PlanJsonPayload:
        """Return a JSON-serializable inspection payload for a runtime plan."""
        return self.runtime_client.describe_plan(runtime_config)

    def prompt_parts(
        self,
        parts: list[ContentPart],
        *,
        runtime_config: RuntimeConfig,
        generation_config: GenerationConfig,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history: list[Message] | None = None,
        sink: StreamSink | None = None,
    ) -> PromptResponse:
        """Execute a prompt composed from explicit content parts."""
        return self.runtime_client.prompt_parts(
            parts,
            runtime_config=runtime_config,
            generation_config=generation_config,
            system_prompt=system_prompt,
            history=history,
            sink=sink,
        )

    def create_session(
        self,
        *,
        runtime_config: RuntimeConfig,
        generation_config: GenerationConfig,
        session_name: str,
        system_prompt: str,
        autosave_path: Path | None = None,
        messages: list[Message] | None = None,
    ) -> ChatSession:
        """Create a reusable chat session over the shared runtime stack."""
        return self.runtime_client.session(
            runtime_config=runtime_config,
            generation_config=generation_config,
            session_name=session_name,
            system_prompt=system_prompt,
            autosave_path=autosave_path,
            messages=messages,
        )

    def run_doctor(
        self,
        *,
        runtime_config: RuntimeConfig,
        include_imports: bool = True,
        include_runtime: bool = True,
        include_paths: bool = True,
        include_download: bool = False,
    ) -> DoctorReport:
        """Run the doctor checks through the shared runtime stack."""
        if self.doctor_service is None:
            raise ValueError("ApplicationService doctor_service must be initialized")
        return self.doctor_service.run(
            runtime_config=runtime_config,
            include_imports=include_imports,
            include_runtime=include_runtime,
            include_paths=include_paths,
            include_download=include_download,
        )

    def download_model(
        self,
        model_reference: str,
        models_dir: Path,
        *,
        force_download: bool = False,
    ) -> Path:
        """Materialize runtime artifacts for a model reference."""
        return self.runtime_client.runtime_loader.download(
            model_reference,
            models_dir.expanduser().resolve(),
            force_download=force_download,
        )


def build_default_application_service() -> ApplicationService:
    """Build the default shared application service for CLI and server surfaces."""
    resolver = ModelResolver()
    runtime_client = RuntimeClient(runtime_loader=RuntimeLoader(resolver=resolver))
    return ApplicationService(runtime_client=runtime_client)

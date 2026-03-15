"""High-level public runtime client built on the resolver, planner, loader, and executor stack."""

from dataclasses import dataclass, field, replace
from pathlib import Path

from ollm.app.session import ChatSession
from ollm.app.types import ContentKind, ContentPart, Message, MessageRole, PromptRequest, PromptResponse
from ollm.runtime.config import DEFAULT_SYSTEM_PROMPT, GenerationConfig, RuntimeConfig
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.inspection import plan_json_payload
from ollm.runtime.loader import DiscoveredRuntimeModel, LoadedRuntime, RuntimeLoader
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.resolver import ResolvedModel
from ollm.runtime.streaming import StreamSink


@dataclass(slots=True)
class RuntimeClient:
    """High-level API for resolving, inspecting, loading, and prompting model references."""

    runtime_loader: RuntimeLoader = field(default_factory=RuntimeLoader)
    runtime_executor: RuntimeExecutor = field(default_factory=RuntimeExecutor)

    def resolve(self, model_reference: str, models_dir: Path = Path("models")) -> ResolvedModel:
        """Resolve a model reference without loading a runtime."""
        return self.runtime_loader.resolve(model_reference, models_dir.expanduser().resolve())

    def discover_local_models(self, models_dir: Path = Path("models")) -> tuple[ResolvedModel, ...]:
        """Discover local materialized models under a models directory."""
        return self.runtime_loader.discover_local_models(models_dir.expanduser().resolve())

    def discover_provider_models(
        self,
        provider_names: tuple[str, ...],
        *,
        models_dir: Path = Path("models"),
        provider_endpoint: str | None = None,
        strict: bool = False,
    ) -> tuple[DiscoveredRuntimeModel, ...]:
        """Discover models exposed by one or more provider backends."""
        return self.runtime_loader.discover_provider_models(
            models_dir.expanduser().resolve(),
            provider_names,
            provider_endpoint,
            strict=strict,
        )

    def plan(self, runtime_config: RuntimeConfig) -> RuntimePlan:
        """Build a runtime plan without loading a backend."""
        return self.runtime_loader.plan(runtime_config)

    def describe_plan(self, runtime_config: RuntimeConfig) -> dict[str, object]:
        """Return a JSON-serializable inspection payload for a runtime plan."""
        return plan_json_payload(runtime_config, self.plan(runtime_config))

    def load(self, runtime_config: RuntimeConfig) -> LoadedRuntime:
        """Resolve and load a runtime backend for the given configuration."""
        return self.runtime_loader.load(runtime_config)

    def prompt(
        self,
        prompt: str,
        *,
        runtime_config: RuntimeConfig,
        generation_config: GenerationConfig | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        images: tuple[str, ...] = (),
        audio: tuple[str, ...] = (),
        sink: StreamSink | None = None,
    ) -> PromptResponse:
        """Execute a single prompt using plain text plus optional image or audio inputs."""
        parts = [ContentPart.text(prompt)]
        parts.extend(ContentPart.image(item) for item in images)
        parts.extend(ContentPart.audio(item) for item in audio)
        return self.prompt_parts(
            parts,
            runtime_config=runtime_config,
            generation_config=generation_config,
            system_prompt=system_prompt,
            sink=sink,
        )

    def prompt_parts(
        self,
        parts: list[ContentPart],
        *,
        runtime_config: RuntimeConfig,
        generation_config: GenerationConfig | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history: list[Message] | None = None,
        sink: StreamSink | None = None,
    ) -> PromptResponse:
        """Execute a prompt composed from explicit content parts."""
        if not parts:
            raise ValueError("A prompt requires at least one content part")
        effective_runtime_config = self._runtime_config_for_parts(runtime_config, parts)
        effective_generation_config = GenerationConfig() if generation_config is None else generation_config
        effective_runtime_config.validate()
        effective_generation_config.validate()
        runtime = self.runtime_loader.load(effective_runtime_config)
        request_messages = []
        if system_prompt:
            request_messages.append(Message.system_text(system_prompt))
        if history:
            request_messages.extend(history)
        request_messages.append(Message(role=MessageRole.USER, content=list(parts)))
        request = PromptRequest(
            runtime_config=effective_runtime_config,
            generation_config=effective_generation_config,
            messages=request_messages,
        )
        return self.runtime_executor.execute(runtime, request, sink=sink)

    def session(
        self,
        *,
        runtime_config: RuntimeConfig,
        generation_config: GenerationConfig | None = None,
        session_name: str = "default",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        messages: list[Message] | None = None,
        autosave_path: Path | None = None,
    ) -> ChatSession:
        """Create a reusable chat session over the shared runtime stack."""
        session = ChatSession(
            runtime_loader=self.runtime_loader,
            runtime_executor=self.runtime_executor,
            runtime_config=runtime_config,
            generation_config=GenerationConfig() if generation_config is None else generation_config,
            session_name=session_name,
            system_prompt=system_prompt,
            autosave_path=autosave_path,
        )
        session.runtime_config.validate()
        session.generation_config.validate()
        if messages:
            session.messages.extend(messages)
        return session

    def _runtime_config_for_parts(
        self,
        runtime_config: RuntimeConfig,
        parts: list[ContentPart],
    ) -> RuntimeConfig:
        """Enable multimodal planning automatically when non-text parts are present."""
        requires_multimodal = any(part.kind is not ContentKind.TEXT for part in parts)
        if not requires_multimodal or runtime_config.multimodal:
            return runtime_config
        return replace(runtime_config, multimodal=True)

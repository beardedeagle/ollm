"""High-level public runtime client built on the resolver, planner, loader, and executor stack."""

from dataclasses import dataclass, field, replace
from pathlib import Path

from ollm.app.session import ChatSession
from ollm.app.types import (
    ContentKind,
    ContentPart,
    Message,
    MessageRole,
    PromptRequest,
    PromptResponse,
)
from ollm.runtime.config import DEFAULT_SYSTEM_PROMPT, GenerationConfig, RuntimeConfig
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.inspection import PlanJsonPayload, plan_json_payload
from ollm.runtime.loader import LoadedRuntime, RuntimeLoader
from ollm.runtime.plan import RuntimePlan
from ollm.runtime.resolver import ResolvedModel
from ollm.runtime.streaming import StreamSink


@dataclass(slots=True)
class RuntimeClient:
    """High-level runtime API shared by the CLI and the Python library.

    Attributes:
        runtime_loader (RuntimeLoader): Resolver, planner, materialization, and
            backend-loading boundary.
        runtime_executor (RuntimeExecutor): Prompt execution boundary used once
            a runtime has been loaded.
    """

    runtime_loader: RuntimeLoader = field(default_factory=RuntimeLoader)
    runtime_executor: RuntimeExecutor = field(default_factory=RuntimeExecutor)

    def resolve(
        self, model_reference: str, models_dir: Path = Path("models")
    ) -> ResolvedModel:
        """Resolve a model reference without loading a runtime.

        Args:
            model_reference (str): User-facing model reference such as a built-in
                alias, Hugging Face repository, or local model path.
            models_dir (Path): Local models root used for implicit path
                resolution.

        Returns:
            ResolvedModel: Normalized model metadata for planning or inspection.
        """
        return self.runtime_loader.resolve(
            model_reference, models_dir.expanduser().resolve()
        )

    def discover_local_models(
        self, models_dir: Path = Path("models")
    ) -> tuple[ResolvedModel, ...]:
        """Discover local materialized models under a models directory.

        Args:
            models_dir (Path): Local models root to inspect.

        Returns:
            tuple[ResolvedModel, ...]: Materialized model directories discovered
            under the given root.
        """
        return self.runtime_loader.discover_local_models(
            models_dir.expanduser().resolve()
        )

    def plan(self, runtime_config: RuntimeConfig) -> RuntimePlan:
        """Build a runtime plan without loading a backend.

        Args:
            runtime_config (RuntimeConfig): Execution configuration to inspect.

        Returns:
            RuntimePlan: Planned backend, specialization, and capability result.

        Raises:
            ValueError: Raised when the runtime configuration is invalid or no
                executable plan can be produced.
        """
        return self.runtime_loader.plan(runtime_config)

    def describe_plan(self, runtime_config: RuntimeConfig) -> PlanJsonPayload:
        """Return a JSON-serializable inspection payload for a runtime plan.

        Args:
            runtime_config (RuntimeConfig): Execution configuration to inspect.

        Returns:
            PlanJsonPayload: Serialized inspection payload for CLI or HTTP use.

        Raises:
            ValueError: Raised when the runtime configuration is invalid or no
                executable plan can be produced.
        """
        return plan_json_payload(runtime_config, self.plan(runtime_config))

    def load(self, runtime_config: RuntimeConfig) -> LoadedRuntime:
        """Resolve and load a runtime backend for the given configuration.

        Args:
            runtime_config (RuntimeConfig): Execution configuration to load.

        Returns:
            LoadedRuntime: Loaded backend runtime bundle ready for execution.

        Raises:
            ValueError: Raised when the model cannot be resolved, materialized,
                planned, or loaded.
        """
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
        """Execute one prompt using text plus optional image or audio inputs.

        Args:
            prompt (str): Primary text prompt.
            runtime_config (RuntimeConfig): Runtime configuration to execute.
            generation_config (GenerationConfig | None): Optional generation
                overrides. Defaults to ``GenerationConfig()`` when omitted.
            system_prompt (str): System instruction prepended to the request
                when non-empty.
            images (tuple[str, ...]): Optional image input paths or URIs.
            audio (tuple[str, ...]): Optional audio input paths or URIs.
            sink (StreamSink | None): Optional streaming sink for incremental
                text callbacks.

        Returns:
            PromptResponse: Final prompt response and assistant message payload.

        Raises:
            ValueError: Raised when the runtime or generation configuration is
                invalid or when no executable backend exists.
        """
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
        """Execute a prompt composed from explicit content parts.

        Args:
            parts (list[ContentPart]): Prompt payload parts in final user-message
                order.
            runtime_config (RuntimeConfig): Runtime configuration to execute.
            generation_config (GenerationConfig | None): Optional generation
                overrides. Defaults to ``GenerationConfig()`` when omitted.
            system_prompt (str): System instruction prepended to the request
                when non-empty.
            history (list[Message] | None): Optional prior conversation messages
                to prepend before the new user message.
            sink (StreamSink | None): Optional streaming sink for incremental
                callbacks.

        Returns:
            PromptResponse: Final prompt response and assistant message payload.

        Raises:
            ValueError: Raised when ``parts`` is empty or when runtime/generation
                validation or backend loading fails.
        """
        if not parts:
            raise ValueError("A prompt requires at least one content part")
        effective_runtime_config = self._runtime_config_for_parts(runtime_config, parts)
        effective_generation_config = (
            GenerationConfig() if generation_config is None else generation_config
        )
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
            runtime_config=runtime.config,
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
        """Create a reusable chat session over the shared runtime stack.

        Args:
            runtime_config (RuntimeConfig): Runtime configuration for the
                session.
            generation_config (GenerationConfig | None): Optional generation
                overrides. Defaults to ``GenerationConfig()`` when omitted.
            session_name (str): Human-readable session label.
            system_prompt (str): Session-wide system instruction.
            messages (list[Message] | None): Optional initial transcript
                messages.
            autosave_path (Path | None): Optional transcript autosave path.

        Returns:
            ChatSession: Reusable session object bound to the shared runtime
            stack.

        Raises:
            ValueError: Raised when the runtime or generation configuration is
                invalid.
        """
        session = ChatSession(
            runtime_loader=self.runtime_loader,
            runtime_executor=self.runtime_executor,
            runtime_config=runtime_config,
            generation_config=GenerationConfig()
            if generation_config is None
            else generation_config,
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

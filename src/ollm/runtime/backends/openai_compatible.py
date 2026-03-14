from collections.abc import Callable
from dataclasses import replace

import torch

from ollm.app.types import Message, PromptRequest, PromptResponse
from ollm.runtime.backends.base import BackendRuntime, DiscoveredProviderModel, ExecutionBackend
from ollm.runtime.capabilities import SupportLevel, provider_capabilities
from ollm.runtime.config import RuntimeConfig, normalize_provider_endpoint
from ollm.runtime.plan import RuntimePlan, SpecializationState
from ollm.runtime.providers.openai_compatible_client import (
	DEFAULT_LMSTUDIO_ENDPOINT,
	OpenAICompatibleClient,
	OpenAICompatibleConnectionError,
	OpenAICompatibleRequestError,
)
from ollm.runtime.resolver import ModelSourceKind
from ollm.runtime.streaming import StreamSink


class OpenAICompatibleBackend(ExecutionBackend):
	backend_id = "openai-compatible"

	def __init__(
		self,
		client_factory: Callable[[str], OpenAICompatibleClient] | None = None,
	):
		self._client_factory = OpenAICompatibleClient if client_factory is None else client_factory

	def supports_provider_discovery(self, provider_name: str) -> bool:
		return provider_name in {"openai-compatible", "lmstudio"}

	def discover_provider_models(
		self,
		provider_name: str,
		provider_endpoint: str | None = None,
	) -> tuple[DiscoveredProviderModel, ...]:
		if provider_name not in {"openai-compatible", "lmstudio"}:
			return ()
		resolved_endpoint = _resolve_discovery_endpoint(provider_name, provider_endpoint)
		client = self._client_factory(resolved_endpoint)
		model_names = client.list_models()
		return tuple(
			DiscoveredProviderModel(
				model_reference=f"{provider_name}:{model_name}",
				provider_name=provider_name,
				provider_endpoint=resolved_endpoint,
			)
			for model_name in model_names
		)

	def refine_plan(self, plan: RuntimePlan, config: RuntimeConfig) -> RuntimePlan:
		if plan.resolved_model.source_kind is not ModelSourceKind.PROVIDER:
			return plan
		provider_name = plan.resolved_model.provider_name
		if provider_name not in {"openai-compatible", "lmstudio"}:
			return plan
		if config.resolved_adapter_dir() is not None:
			reason = (
				f"{provider_name} provider execution does not support PEFT adapters for "
				f"{plan.resolved_model.reference.raw}."
			)
			return self._provider_failure_plan(plan, reason, config.resolved_provider_endpoint())
		try:
			provider_endpoint = _resolve_provider_endpoint(provider_name, config)
		except ValueError as exc:
			return self._provider_failure_plan(plan, str(exc), config.resolved_provider_endpoint())
		client = self._client_factory(provider_endpoint)
		try:
			model_ids = client.list_models()
		except OpenAICompatibleConnectionError as exc:
			return self._provider_failure_plan(plan, str(exc), provider_endpoint)
		except OpenAICompatibleRequestError as exc:
			return self._provider_failure_plan(
				plan,
				str(exc),
				provider_endpoint,
				provider_status_code=str(exc.status_code),
			)
		model_name = plan.resolved_model.reference.identifier
		if model_name not in model_ids:
			return self._provider_failure_plan(
				plan,
				f"Provider endpoint {provider_endpoint} does not advertise model '{model_name}'.",
				provider_endpoint,
				available_models=",".join(model_ids),
			)
		return self._provider_success_plan(
			plan=plan,
			provider_name=provider_name,
			provider_endpoint=provider_endpoint,
			available_model_count=len(model_ids),
		)

	def load(self, plan: RuntimePlan, config: RuntimeConfig) -> BackendRuntime:
		provider_name = plan.resolved_model.provider_name
		if provider_name not in {"openai-compatible", "lmstudio"}:
			raise ValueError("openai-compatible backend requires an openai-compatible: or lmstudio: model reference")
		provider_endpoint = _resolve_provider_endpoint(provider_name, config)
		client = self._client_factory(provider_endpoint)
		return BackendRuntime(
			backend_id=self.backend_id,
			model=None,
			tokenizer=None,
			processor=None,
			device=torch.device("cpu"),
			stats=None,
			print_suppression_modules=(),
			create_cache=lambda cache_dir: None,
			apply_offload=_validate_provider_offload,
			execute_prompt=lambda request, sink: self._execute_prompt(client, plan, request, sink),
		)

	def _provider_success_plan(
		self,
		plan: RuntimePlan,
		provider_name: str,
		provider_endpoint: str,
		available_model_count: int,
	) -> RuntimePlan:
		reason = (
			f"{provider_name} model '{plan.resolved_model.reference.identifier}' is executable via "
			f"{provider_endpoint}."
		)
		capabilities = provider_capabilities(
			provider_name,
			details={
				"provider_endpoint": provider_endpoint,
				"provider_backend": self.backend_id,
				"available_model_count": str(available_model_count),
				"api_style": "openai-compatible",
			},
		)
		resolved_model = replace(
			plan.resolved_model,
			capabilities=capabilities,
			resolution_message=reason,
		)
		details = dict(plan.details)
		details.update(capabilities.details)
		return replace(
			plan,
			resolved_model=resolved_model,
			backend_id=self.backend_id,
			support_level=SupportLevel.PROVIDER_BACKED,
			specialization_enabled=False,
			specialization_applied=False,
			specialization_provider_id=None,
			specialization_state=SpecializationState.NOT_PLANNED,
			specialization_pass_ids=(),
			applied_specialization_pass_ids=(),
			fallback_reason=None,
			reason=reason,
			details=details,
		)

	def _provider_failure_plan(
		self,
		plan: RuntimePlan,
		reason: str,
		provider_endpoint: str | None,
		**extra_details: str,
	) -> RuntimePlan:
		details = dict(plan.details)
		details["provider_backend"] = self.backend_id
		if provider_endpoint is not None:
			details["provider_endpoint"] = provider_endpoint
		for key, value in extra_details.items():
			details[key] = value
		provider_name = plan.resolved_model.provider_name or "openai-compatible"
		capabilities = provider_capabilities(provider_name, details=details)
		resolved_model = replace(
			plan.resolved_model,
			capabilities=capabilities,
			resolution_message=reason,
		)
		return replace(
			plan,
			resolved_model=resolved_model,
			backend_id=None,
			support_level=SupportLevel.PROVIDER_BACKED,
			reason=reason,
			details=details,
		)

	def _execute_prompt(
		self,
		client: OpenAICompatibleClient,
		plan: RuntimePlan,
		request: PromptRequest,
		sink: StreamSink,
	) -> PromptResponse:
		if request.generation_config.top_k is not None:
			raise ValueError("The openai-compatible backend does not support --top-k")
		chat_result = client.chat_completions(
			provider_name=plan.resolved_model.provider_name or "openai-compatible",
			model_name=plan.resolved_model.reference.identifier,
			messages=_openai_messages(request.messages),
			options=_openai_options(request),
			stream=request.generation_config.stream,
			on_text=sink.on_text if request.generation_config.stream else None,
		)
		if request.generation_config.stream:
			sink.on_complete(chat_result.text)
		metadata = dict(chat_result.metadata)
		metadata["provider"] = plan.resolved_model.provider_name or "openai-compatible"
		return PromptResponse(
			text=chat_result.text,
			assistant_message=Message.assistant_text(chat_result.text),
			metadata=metadata,
		)


def _resolve_provider_endpoint(provider_name: str, config: RuntimeConfig) -> str:
	resolved_endpoint = config.resolved_provider_endpoint()
	if resolved_endpoint is not None:
		return resolved_endpoint
	if provider_name == "lmstudio":
		return DEFAULT_LMSTUDIO_ENDPOINT
	raise ValueError(
		f"Provider-backed model reference '{config.model_reference}' requires --provider-endpoint."
	)


def _resolve_discovery_endpoint(provider_name: str, provider_endpoint: str | None) -> str:
	if provider_endpoint is not None:
		normalized_endpoint = normalize_provider_endpoint(provider_endpoint)
		if normalized_endpoint is None:
			raise ValueError("Provider discovery requires a valid provider endpoint.")
		return normalized_endpoint
	if provider_name == "lmstudio":
		return DEFAULT_LMSTUDIO_ENDPOINT
	raise ValueError(
		"OpenAI-compatible provider discovery requires --provider-endpoint."
	)


def _validate_provider_offload(config: RuntimeConfig) -> None:
	if config.offload_cpu_layers > 0 or config.offload_gpu_layers > 0:
		raise ValueError("Provider-backed backends do not support custom layer offload controls")


def _openai_options(request: PromptRequest) -> dict[str, object]:
	generation_config = request.generation_config
	options: dict[str, object] = {
		"max_tokens": generation_config.max_new_tokens,
		"temperature": generation_config.temperature,
	}
	if generation_config.top_p is not None:
		options["top_p"] = generation_config.top_p
	if generation_config.seed is not None:
		options["seed"] = generation_config.seed
	return options


def _openai_messages(messages: list[Message]) -> list[dict[str, str]]:
	return [_openai_message(message) for message in messages]


def _openai_message(message: Message) -> dict[str, str]:
	if message.contains_non_text():
		raise ValueError("The openai-compatible backend currently supports text-only inputs")
	return {"role": message.role.value, "content": message.text_content()}

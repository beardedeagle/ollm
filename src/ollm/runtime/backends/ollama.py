from collections.abc import Callable
from dataclasses import replace

import torch

from ollm.app.types import ContentKind, Message, PromptRequest, PromptResponse
from ollm.runtime.backends.base import BackendRuntime, DiscoveredProviderModel, ExecutionBackend
from ollm.runtime.capabilities import SupportLevel, provider_capabilities
from ollm.runtime.config import RuntimeConfig, normalize_provider_endpoint
from ollm.runtime.media_inputs import encode_image_input_base64
from ollm.runtime.plan import RuntimePlan, SpecializationState
from ollm.runtime.providers.ollama_client import (
	OllamaClient,
	OllamaConnectionError,
	OllamaModelDetails,
	OllamaRequestError,
)
from ollm.runtime.resolver import ModelSourceKind
from ollm.runtime.streaming import StreamSink


class OllamaBackend(ExecutionBackend):
	backend_id = "ollama"

	def __init__(
		self,
		client: OllamaClient | None = None,
		client_factory: Callable[[str], OllamaClient] | None = None,
	):
		self._default_client = OllamaClient() if client is None else client
		self._client_factory = OllamaClient if client_factory is None else client_factory

	def supports_provider_discovery(self, provider_name: str) -> bool:
		return provider_name in {"ollama", "msty"}

	def discover_provider_models(
		self,
		provider_name: str,
		provider_endpoint: str | None = None,
	) -> tuple[DiscoveredProviderModel, ...]:
		if provider_name not in {"ollama", "msty"}:
			return ()
		resolved_endpoint = _resolve_discovery_endpoint(
			provider_name,
			provider_endpoint,
			self._default_client.base_url,
		)
		client = self._client_for_endpoint(resolved_endpoint)
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
		if provider_name not in {"ollama", "msty"}:
			return plan
		if config.resolved_adapter_dir() is not None:
			reason = (
				f"{provider_name} provider execution does not support PEFT adapters for "
				f"{plan.resolved_model.reference.raw}."
			)
			return self._provider_failure_plan(
				plan,
				provider_name,
				reason,
				{"provider_endpoint": _provider_endpoint_details(provider_name, config, self._default_client.base_url)},
			)
		try:
			client = self._client_from_config(provider_name, config)
		except ValueError as exc:
			return self._provider_failure_plan(
				plan,
				provider_name,
				str(exc),
				{
					"provider_endpoint": _provider_endpoint_details(
						provider_name,
						config,
						self._default_client.base_url,
					),
				},
			)
		try:
			model_details = client.show_model(plan.resolved_model.reference.identifier)
		except OllamaConnectionError as exc:
			return self._provider_failure_plan(
				plan,
				provider_name,
				str(exc),
				{"provider_endpoint": client.base_url},
			)
		except OllamaRequestError as exc:
			return self._provider_failure_plan(
				plan,
				provider_name,
				str(exc),
				{
					"provider_endpoint": client.base_url,
					"provider_status_code": str(exc.status_code),
				},
			)
		return self._provider_success_plan(plan, provider_name, client.base_url, model_details)

	def load(self, plan: RuntimePlan, config: RuntimeConfig) -> BackendRuntime:
		provider_name = plan.resolved_model.provider_name
		if provider_name not in {"ollama", "msty"}:
			raise ValueError("ollama backend requires an ollama: or msty: model reference")
		client = self._client_from_config(provider_name, config)
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
			allows_multimodal_without_processor=True,
		)

	def _provider_success_plan(
		self,
		plan: RuntimePlan,
		provider_name: str,
		provider_endpoint: str,
		model_details: OllamaModelDetails,
	) -> RuntimePlan:
		reason = (
			f"{provider_name} model '{model_details.name}' is executable via {provider_endpoint}."
		)
		capabilities = provider_capabilities(
			provider_name,
			modalities=model_details.modalities,
			details={
				"provider_backend": self.backend_id,
				"provider_endpoint": provider_endpoint,
				"api_style": "ollama",
				"capabilities": ",".join(model_details.capabilities),
				"family": "" if model_details.family is None else model_details.family,
				"parameter_size": (
					"" if model_details.parameter_size is None else model_details.parameter_size
				),
				"quantization_level": (
					""
					if model_details.quantization_level is None
					else model_details.quantization_level
				),
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
		provider_name: str,
		reason: str,
		extra_details: dict[str, str],
	) -> RuntimePlan:
		details = dict(plan.details)
		for key, value in extra_details.items():
			if value:
				details[key] = value
		details["provider_backend"] = self.backend_id
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
		client: OllamaClient,
		plan: RuntimePlan,
		request: PromptRequest,
		sink: StreamSink,
	) -> PromptResponse:
		provider_name = plan.resolved_model.provider_name or "ollama"
		chat_result = client.chat(
			model_name=plan.resolved_model.reference.identifier,
			messages=_ollama_messages(request.messages),
			options=_ollama_options(request.generation_config),
			stream=request.generation_config.stream,
			on_text=sink.on_text if request.generation_config.stream else None,
		)
		if request.generation_config.stream:
			sink.on_complete(chat_result.text)
		metadata = dict(chat_result.metadata)
		metadata["provider"] = provider_name
		metadata["provider_backend"] = self.backend_id
		return PromptResponse(
			text=chat_result.text,
			assistant_message=Message.assistant_text(chat_result.text),
			metadata=metadata,
		)

	def _client_from_config(self, provider_name: str, config: RuntimeConfig) -> OllamaClient:
		resolved_endpoint = _resolve_provider_endpoint(
			provider_name,
			config,
			self._default_client.base_url,
		)
		return self._client_for_endpoint(resolved_endpoint)

	def _client_for_endpoint(self, endpoint: str) -> OllamaClient:
		if endpoint == self._default_client.base_url:
			return self._default_client
		return self._client_factory(endpoint)


def _validate_provider_offload(config: RuntimeConfig) -> None:
	if config.offload_cpu_layers > 0 or config.offload_gpu_layers > 0:
		raise ValueError("Provider-backed backends do not support custom layer offload controls")


def _resolve_provider_endpoint(
	provider_name: str,
	config: RuntimeConfig,
	default_ollama_endpoint: str,
) -> str:
	if provider_name == "ollama":
		return default_ollama_endpoint
	resolved_endpoint = config.resolved_provider_endpoint()
	if resolved_endpoint is not None:
		return resolved_endpoint
	raise ValueError(
		f"Provider-backed model reference '{config.model_reference}' requires --provider-endpoint."
	)


def _resolve_discovery_endpoint(
	provider_name: str,
	provider_endpoint: str | None,
	default_ollama_endpoint: str,
) -> str:
	if provider_name == "ollama":
		return default_ollama_endpoint
	if provider_endpoint is not None:
		normalized_endpoint = normalize_provider_endpoint(provider_endpoint)
		if normalized_endpoint is None:
			raise ValueError("Provider discovery requires a valid provider endpoint.")
		return normalized_endpoint
	raise ValueError("--discover-provider msty requires --provider-endpoint")


def _provider_endpoint_details(
	provider_name: str,
	config: RuntimeConfig,
	default_ollama_endpoint: str,
) -> str:
	if provider_name == "ollama":
		return default_ollama_endpoint
	resolved_endpoint = config.resolved_provider_endpoint()
	if resolved_endpoint is not None:
		return resolved_endpoint
	return ""


def _ollama_options(generation_config) -> dict[str, object]:
	options: dict[str, object] = {
		"num_predict": generation_config.max_new_tokens,
		"temperature": generation_config.temperature,
	}
	if generation_config.top_p is not None:
		options["top_p"] = generation_config.top_p
	if generation_config.top_k is not None:
		options["top_k"] = generation_config.top_k
	if generation_config.seed is not None:
		options["seed"] = generation_config.seed
	return options


def _ollama_messages(messages: list[Message]) -> list[dict[str, object]]:
	return [_ollama_message(message) for message in messages]


def _ollama_message(message: Message) -> dict[str, object]:
	ollama_message: dict[str, object] = {
		"role": message.role.value,
		"content": message.text_content(),
	}
	images: list[str] = []
	for part in message.content:
		if part.kind is ContentKind.TEXT:
			continue
		if part.kind is ContentKind.AUDIO:
			raise ValueError("The Ollama backend does not support audio inputs")
		images.append(_encoded_image_value(part.value))
	if images:
		ollama_message["images"] = images
	return ollama_message


def _encoded_image_value(value: str) -> str:
	return encode_image_input_base64(value)

from dataclasses import replace

import torch

from ollm.app.types import ContentKind, Message, PromptRequest, PromptResponse
from ollm.runtime.backends.base import BackendRuntime, DiscoveredProviderModel, ExecutionBackend
from ollm.runtime.capabilities import SupportLevel, provider_capabilities
from ollm.runtime.config import RuntimeConfig
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

	def __init__(self, client: OllamaClient | None = None):
		self._client = OllamaClient() if client is None else client

	def supports_provider_discovery(self, provider_name: str) -> bool:
		return provider_name == "ollama"

	def discover_provider_models(
		self,
		provider_name: str,
		provider_endpoint: str | None = None,
	) -> tuple[DiscoveredProviderModel, ...]:
		del provider_endpoint
		if provider_name != "ollama":
			return ()
		model_names = self._client.list_models()
		return tuple(
			DiscoveredProviderModel(
				model_reference=f"ollama:{model_name}",
				provider_name="ollama",
				provider_endpoint=self._client.base_url,
			)
			for model_name in model_names
		)

	def refine_plan(self, plan: RuntimePlan, config: RuntimeConfig) -> RuntimePlan:
		if plan.resolved_model.source_kind is not ModelSourceKind.PROVIDER:
			return plan
		if plan.resolved_model.provider_name != "ollama":
			return plan
		if config.resolved_adapter_dir() is not None:
			reason = (
				f"Ollama provider execution does not support PEFT adapters for "
				f"{plan.resolved_model.reference.raw}."
			)
			return self._provider_failure_plan(plan, reason, {"provider_endpoint": self._client.base_url})
		try:
			model_details = self._client.show_model(plan.resolved_model.reference.identifier)
		except OllamaConnectionError as exc:
			return self._provider_failure_plan(
				plan,
				str(exc),
				{"provider_endpoint": self._client.base_url},
			)
		except OllamaRequestError as exc:
			return self._provider_failure_plan(
				plan,
				str(exc),
				{
					"provider_endpoint": self._client.base_url,
					"provider_status_code": str(exc.status_code),
				},
			)
		return self._provider_success_plan(plan, model_details)

	def load(self, plan: RuntimePlan, config: RuntimeConfig) -> BackendRuntime:
		del config
		if plan.resolved_model.provider_name != "ollama":
			raise ValueError("ollama backend requires an ollama: model reference")
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
			execute_prompt=lambda request, sink: self._execute_prompt(plan, request, sink),
			allows_multimodal_without_processor=True,
		)

	def _provider_success_plan(
		self,
		plan: RuntimePlan,
		model_details: OllamaModelDetails,
	) -> RuntimePlan:
		reason = (
			f"Ollama model '{model_details.name}' is executable via {self._client.base_url}."
		)
		capabilities = provider_capabilities(
			"ollama",
			modalities=model_details.modalities,
			details={
				"provider_endpoint": self._client.base_url,
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
		reason: str,
		extra_details: dict[str, str],
	) -> RuntimePlan:
		details = dict(plan.details)
		details.update(extra_details)
		capabilities = provider_capabilities("ollama", details=details)
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
		plan: RuntimePlan,
		request: PromptRequest,
		sink: StreamSink,
	) -> PromptResponse:
		chat_result = self._client.chat(
			model_name=plan.resolved_model.reference.identifier,
			messages=_ollama_messages(request.messages),
			options=_ollama_options(request.generation_config),
			stream=request.generation_config.stream,
			on_text=sink.on_text if request.generation_config.stream else None,
		)
		if request.generation_config.stream:
			sink.on_complete(chat_result.text)
		return PromptResponse(
			text=chat_result.text,
			assistant_message=Message.assistant_text(chat_result.text),
			metadata=chat_result.metadata,
		)


def _validate_provider_offload(config: RuntimeConfig) -> None:
	if config.offload_cpu_layers > 0 or config.offload_gpu_layers > 0:
		raise ValueError("Provider-backed backends do not support custom layer offload controls")


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

from dataclasses import dataclass

import torch

from ollm.app.types import ContentKind, Message, PromptRequest, PromptResponse
from ollm.runtime.capability_discovery import GenericModelKind
from ollm.runtime.catalog import ModelModality
from ollm.runtime.loader import LoadedRuntime
from ollm.runtime.output_control import suppress_module_prints
from ollm.runtime.streaming import BufferedTextStreamer, NullStreamSink, StreamSink


class PromptExecutionError(RuntimeError):
    """Raised when a prompt cannot be executed safely."""


@dataclass(slots=True)
class RuntimeExecutor:
    def execute(self, runtime: LoadedRuntime, request: PromptRequest, sink: StreamSink | None = None) -> PromptResponse:
        stream_sink = sink or NullStreamSink()
        self._validate_request(runtime, request)
        if request.generation_config.seed is not None:
            torch.manual_seed(request.generation_config.seed)

        inputs = self._build_inputs(runtime, request.messages)
        streamer = None
        if request.generation_config.stream:
            streamer = BufferedTextStreamer(runtime.tokenizer, stream_sink, skip_prompt=True, skip_special_tokens=False)

        generate_kwargs = self._build_generate_kwargs(runtime, request, streamer)
        stream_sink.on_status(
            f"Running {runtime.config.model_reference} on {runtime.config.device} via {runtime.plan.backend_id}"
        )

        with torch.inference_mode():
            with suppress_module_prints(runtime.backend.print_suppression_modules):
                outputs = runtime.model.generate(**inputs, **generate_kwargs)

        if hasattr(outputs, "detach"):
            outputs = outputs.detach()
        outputs = outputs.cpu()
        response_text = self._decode_response(runtime, inputs, outputs)
        if streamer is not None and not response_text.strip():
            response_text = streamer.text
        assistant_message = Message.assistant_text(response_text)
        metadata = {
            "backend_id": runtime.plan.backend_id or "unknown",
            "specialization_state": runtime.plan.specialization_state.value,
            "specialization_applied": str(runtime.plan.specialization_applied).lower(),
            "specialization_provider_id": runtime.plan.specialization_provider_id or "",
            "specialization_pass_ids": ",".join(
                pass_id.value for pass_id in runtime.plan.specialization_pass_ids
            ),
            "applied_specialization_pass_ids": ",".join(
                pass_id.value for pass_id in runtime.plan.applied_specialization_pass_ids
            ),
            "fallback_reason": runtime.plan.fallback_reason or "",
        }
        if runtime.backend.stats is not None:
            metadata["stats"] = runtime.backend.stats.print_and_clean()
        return PromptResponse(text=response_text, assistant_message=assistant_message, metadata=metadata)

    def _validate_request(self, runtime: LoadedRuntime, request: PromptRequest) -> None:
        if not request.messages:
            raise PromptExecutionError("At least one message is required")

        contains_image = any(
            part.kind is ContentKind.IMAGE
            for message in request.messages
            for part in message.content
        )
        contains_audio = any(
            part.kind is ContentKind.AUDIO
            for message in request.messages
            for part in message.content
        )

        if contains_image and not runtime.capabilities.supports_modality(ModelModality.IMAGE):
            raise PromptExecutionError(f"{runtime.config.model_reference} does not support image inputs")
        if contains_audio and not runtime.capabilities.supports_modality(ModelModality.AUDIO):
            raise PromptExecutionError(f"{runtime.config.model_reference} does not support audio inputs")
        if (contains_image or contains_audio) and runtime.processor is None:
            raise PromptExecutionError(
                "Multimodal inputs require a processor-backed runtime. "
                "Enable --multimodal with a compatible model reference."
            )

    def _build_inputs(self, runtime: LoadedRuntime, messages: list[Message]) -> dict[str, object]:
        transformers_messages = [message.as_transformers_message() for message in messages]
        if runtime.processor is not None:
            inputs = runtime.processor.apply_chat_template(
                transformers_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            contains_image = any(
                part.kind is ContentKind.IMAGE
                for message in messages
                for part in message.content
            )
            if contains_image:
                return inputs.to(runtime.device, dtype=torch.bfloat16)
            return inputs.to(runtime.device)

        if hasattr(runtime.tokenizer, "apply_chat_template"):
            try:
                input_ids = runtime.tokenizer.apply_chat_template(
                    transformers_messages,
                    reasoning_effort="minimal",
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=False,
                ).to(runtime.device)
                return {"input_ids": input_ids}
            except (TypeError, ValueError, AttributeError):
                try:
                    input_ids = runtime.tokenizer.apply_chat_template(
                        transformers_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=False,
                    ).to(runtime.device)
                    return {"input_ids": input_ids}
                except (TypeError, ValueError, AttributeError):
                    pass

        rendered_prompt = _render_plain_prompt(messages)
        tokenized = runtime.tokenizer(rendered_prompt, return_tensors="pt")
        return {key: value.to(runtime.device) for key, value in tokenized.items()}

    def _build_generate_kwargs(self, runtime: LoadedRuntime, request: PromptRequest, streamer) -> dict[str, object]:
        config = request.generation_config
        generate_kwargs: dict[str, object] = {
            "max_new_tokens": config.max_new_tokens,
            "use_cache": True,
        }

        if request.runtime_config.use_cache:
            cache = runtime.backend.create_cache(request.runtime_config.resolved_cache_dir())
            if cache is not None:
                generate_kwargs["past_key_values"] = cache

        if config.sampling_enabled():
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                generate_kwargs["top_p"] = config.top_p
            if config.top_k is not None:
                generate_kwargs["top_k"] = config.top_k
        else:
            generate_kwargs["do_sample"] = False

        if streamer is not None:
            generate_kwargs["streamer"] = streamer

        if runtime.processor is not None and any(
            part.kind is ContentKind.AUDIO for message in request.messages for part in message.content
        ):
            generate_kwargs["do_sample"] = False

        return generate_kwargs

    def _decode_response(self, runtime: LoadedRuntime, inputs: dict[str, object], outputs) -> str:
        if runtime.processor is not None:
            input_ids = inputs["input_ids"]
            decoded = runtime.processor.batch_decode(
                outputs[:, input_ids.shape[1]:],
                skip_special_tokens=False,
            )
            if not decoded:
                return ""
            return decoded[0]

        if runtime.plan.generic_model_kind is GenericModelKind.SEQ2SEQ_LM:
            return runtime.tokenizer.decode(outputs[0], skip_special_tokens=False)

        input_ids = inputs["input_ids"]
        return runtime.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)


def _render_plain_prompt(messages: list[Message]) -> str:
    rendered_messages: list[str] = []
    for message in messages:
        text = message.text_content().strip()
        if not text:
            continue
        rendered_messages.append(f"{message.role.value.upper()}: {text}")
    rendered_messages.append("ASSISTANT:")
    return "\n\n".join(rendered_messages)

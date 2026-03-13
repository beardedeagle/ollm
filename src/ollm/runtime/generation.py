from dataclasses import dataclass

import torch

from ollm.app.types import ContentKind, Message, MessageRole, PromptRequest, PromptResponse
from ollm.runtime.catalog import ModelModality
from ollm.runtime.loader import LoadedRuntime
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
        stream_sink.on_status(f"Running {runtime.config.model_id} on {runtime.config.device}")

        with torch.inference_mode():
            outputs = runtime.model.generate(**inputs, **generate_kwargs)

        if hasattr(outputs, "detach"):
            outputs = outputs.detach()
        outputs = outputs.cpu()
        response_text = self._decode_response(runtime, inputs, outputs)
        if streamer is not None and not response_text.strip():
            response_text = streamer.text
        assistant_message = Message.assistant_text(response_text)
        metadata = {}
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

        if contains_image and not runtime.entry.supports_modality(ModelModality.IMAGE):
            raise PromptExecutionError(f"{runtime.entry.model_id} does not support image inputs")
        if contains_audio and not runtime.entry.supports_modality(ModelModality.AUDIO):
            raise PromptExecutionError(f"{runtime.entry.model_id} does not support audio inputs")
        if (contains_image or contains_audio) and runtime.processor is None:
            raise PromptExecutionError(
                "Multimodal inputs require a processor-backed runtime. "
                "Enable --multimodal with a compatible model."
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

        input_ids = runtime.tokenizer.apply_chat_template(
            transformers_messages,
            reasoning_effort="minimal",
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        ).to(runtime.device)
        return {"input_ids": input_ids}

    def _build_generate_kwargs(self, runtime: LoadedRuntime, request: PromptRequest, streamer) -> dict[str, object]:
        config = request.generation_config
        generate_kwargs: dict[str, object] = {
            "max_new_tokens": config.max_new_tokens,
            "use_cache": True,
        }

        if request.runtime_config.use_cache:
            generate_kwargs["past_key_values"] = runtime.backend.DiskCache(
                cache_dir=str(request.runtime_config.resolved_cache_dir())
            )
        else:
            generate_kwargs["past_key_values"] = None

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

        input_ids = inputs["input_ids"]
        return runtime.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)

"""Manual native-model smoke runner with explicit model and cache paths."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import torch
from transformers import AutoTokenizer, TextStreamer

from ollm import Inference

_DEFAULT_AUDIO_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/"
    "resolve/main/dude_where_is_my_car.wav"
)


class _GenerativeModelProtocol(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...


class _ChatTokenizerProtocol(Protocol):
    def apply_chat_template(self, *args: object, **kwargs: object) -> torch.Tensor: ...

    def decode(self, *args: object, **kwargs: object) -> str: ...


class _AudioProcessorProtocol(Protocol):
    tokenizer: AutoTokenizer

    def apply_chat_template(self, *args: object, **kwargs: object) -> object: ...

    def batch_decode(self, *args: object, **kwargs: object) -> list[str]: ...


class _AudioInputsProtocol(Protocol):
    def to(self, device: torch.device) -> dict[str, object]: ...


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the manual model smoke."""
    parser = argparse.ArgumentParser(
        description="Run a manual chat or audio smoke against a local optimized model."
    )
    parser.add_argument(
        "--model-id",
        choices=("qwen3-next-80B", "gemma3-12B", "voxtral-small-24B"),
        required=True,
        help="Built-in optimized model alias to load.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Directory containing the local optimized model folders.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string.",
    )
    parser.add_argument(
        "--mode",
        choices=("chat", "audio"),
        default="chat",
        help="Smoke mode to run.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for chat mode.",
    )
    parser.add_argument(
        "--offload-cpu-layers",
        type=int,
        default=0,
        help="Optional number of layers to offload to CPU before generation.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant.",
        help="System prompt for chat mode.",
    )
    parser.add_argument(
        "--user-prompt",
        default="List planets starting from Mercury.",
        help="User prompt for chat mode.",
    )
    parser.add_argument(
        "--audio-url",
        default=_DEFAULT_AUDIO_URL,
        help="Audio URL for audio mode.",
    )
    parser.add_argument(
        "--audio-prompt",
        default="What can you tell me about this audio?",
        help="Text prompt paired with the audio input.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum generated tokens for the smoke run.",
    )
    return parser.parse_args()


def _load_inference(args: argparse.Namespace) -> Inference:
    inference = Inference(args.model_id, device=args.device, logging=False)
    inference.ini_model(
        models_dir=str(args.models_dir.expanduser().resolve()),
        force_download=False,
    )
    if args.offload_cpu_layers > 0:
        inference.offload_layers_to_cpu(layers_num=args.offload_cpu_layers)
    return inference


def run_chat_smoke(inference: Inference, args: argparse.Namespace) -> None:
    """Run the chat-mode smoke.

    Args:
        inference (Inference): Loaded optimized inference helper.
        args (argparse.Namespace): Parsed CLI arguments.
    """
    tokenizer = cast(_ChatTokenizerProtocol, inference.tokenizer)
    text_streamer = TextStreamer(
        cast(AutoTokenizer, inference.tokenizer),
        skip_prompt=True,
        skip_special_tokens=False,
    )
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        reasoning_effort="minimal",
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    ).to(inference.device)
    past_key_values = None
    if args.cache_dir is not None:
        past_key_values = inference.DiskCache(
            cache_dir=str(args.cache_dir.expanduser().resolve())
        )
    print(
        "\n\nGenerate started.",
        datetime.now().strftime("%H:%M:%S"),
        "input_ids.shape:",
        input_ids.shape,
    )
    outputs = (
        cast(_GenerativeModelProtocol, inference.model)
        .generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            past_key_values=past_key_values,
            use_cache=True,
            streamer=text_streamer,
        )
        .detach()
        .cpu()
    )
    answer = tokenizer.decode(
        outputs[0][input_ids.shape[-1] :],
        skip_special_tokens=False,
    )
    print(answer)


def run_audio_smoke(inference: Inference, args: argparse.Namespace) -> None:
    """Run the audio-mode smoke.

    Args:
        inference (Inference): Loaded optimized inference helper.
        args (argparse.Namespace): Parsed CLI arguments.
    """
    processor = cast(_AudioProcessorProtocol, inference.processor)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "url": args.audio_url},
                {"type": "text", "text": args.audio_prompt},
            ],
        }
    ]
    raw_inputs = cast(
        _AudioInputsProtocol,
        processor.apply_chat_template(messages, return_tensors="pt"),
    )
    typed_inputs = raw_inputs.to(inference.device)
    input_ids = cast(torch.Tensor, typed_inputs["input_ids"])
    text_streamer = TextStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
    )
    print("\n\nAudio Generate started.", datetime.now().strftime("%H:%M:%S"))
    outputs = (
        cast(_GenerativeModelProtocol, inference.model)
        .generate(
            **typed_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            past_key_values=None,
            use_cache=True,
            streamer=text_streamer,
        )
        .detach()
        .cpu()
    )
    answer = processor.batch_decode(
        outputs[:, input_ids.shape[1] :],
        skip_special_tokens=False,
    )
    print(answer)


def main() -> int:
    """Run the selected manual smoke mode."""
    args = parse_args()
    inference = _load_inference(args)
    if args.mode == "audio":
        if not hasattr(inference, "processor"):
            raise ValueError(f"{args.model_id} does not expose a processor")
        run_audio_smoke(inference, args)
        return 0
    run_chat_smoke(inference, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

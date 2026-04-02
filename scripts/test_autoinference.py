"""Manual AutoInference smoke runner with explicit path-driven configuration."""

import argparse
from pathlib import Path
from typing import Protocol, cast

import torch
from transformers import AutoTokenizer

from ollm import AutoInference, TextStreamer, file_get_contents

_DEFAULT_SAMPLE_FILE = "./temp/xsum_sample.txt"


class _GenerativeModelProtocol(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...


class _TokenBatchProtocol(Protocol):
    @property
    def input_ids(self) -> torch.Tensor: ...


class _TokenizerProtocol(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> _TokenBatchProtocol: ...

    def decode(self, *args: object, **kwargs: object) -> str: ...


def _build_prompt(sample_file: Path) -> str:
    sample_text = file_get_contents(str(sample_file.expanduser().resolve()))
    return f"Extract short summary from document: {sample_text}\nSummary:\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the AutoInference smoke script."""
    parser = argparse.ArgumentParser(
        description="Run a manual AutoInference smoke against a local model directory."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local Llama or Gemma model directory.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Optional PEFT adapter checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./kv_cache"),
        help="Optional cache directory. Pass --no-cache to disable it.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk-backed KV cache use.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=Path(_DEFAULT_SAMPLE_FILE),
        help="Sample text file used to build the summarization prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Maximum generation length for the smoke run.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the manual AutoInference smoke."""
    args = parse_args()
    inference = AutoInference(
        str(args.model_dir.expanduser().resolve()),
        adapter_dir=(
            None
            if args.adapter_dir is None
            else str(args.adapter_dir.expanduser().resolve())
        ),
        device=args.device,
        multimodality=False,
        logging=False,
    )
    past_key_values = None
    if not args.no_cache:
        past_key_values = inference.DiskCache(
            cache_dir=str(args.cache_dir.expanduser().resolve())
        )
    tokenizer = cast(_TokenizerProtocol, inference.tokenizer)
    model = cast(_GenerativeModelProtocol, inference.model)
    text_streamer = TextStreamer(
        cast(AutoTokenizer, inference.tokenizer),
        skip_prompt=True,
        skip_special_tokens=False,
    )
    prompt = _build_prompt(args.sample_file)
    input_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(inference.device)
    outputs = model.generate(
        input_ids=input_ids,
        past_key_values=past_key_values,
        max_new_tokens=args.max_new_tokens,
        streamer=text_streamer,
    ).cpu()
    answer = tokenizer.decode(
        outputs[0][input_ids.shape[-1] :],
        skip_special_tokens=False,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

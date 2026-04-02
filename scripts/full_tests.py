"""Manual multi-scenario runtime regression script with explicit path inputs."""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import torch
from transformers import AutoTokenizer

from ollm import Inference, TextStreamer, file_get_contents

_DEFAULT_TEST_IDS = (4,)


class _GenerativeModelProtocol(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...


class _ChatTokenizerProtocol(Protocol):
    def apply_chat_template(self, *args: object, **kwargs: object) -> torch.Tensor: ...

    def decode(self, *args: object, **kwargs: object) -> str: ...


@dataclass(frozen=True, slots=True)
class _ManualTestCase:
    test_id: int
    model_id: str
    sample_file: str
    prompt_mode: str
    user_prompt: str
    use_disk_cache: bool
    offload_layers_to_cpu: int
    offload_layers_to_gpu: int
    max_new_tokens: int


def _test_cases() -> dict[int, _ManualTestCase]:
    return {
        1: _ManualTestCase(
            test_id=1,
            model_id="llama3-8B-chat",
            sample_file="10k_sample.txt",
            prompt_mode="chat",
            user_prompt="Analyze chats above and write top 10 most popular questions (translate to english).",
            use_disk_cache=True,
            offload_layers_to_cpu=2,
            offload_layers_to_gpu=0,
            max_new_tokens=500,
        ),
        2: _ManualTestCase(
            test_id=2,
            model_id="gpt-oss-20B",
            sample_file="2k_sample.txt",
            prompt_mode="chat",
            user_prompt="Analyze chats above and write top 10 most popular questions (translate to english).",
            use_disk_cache=True,
            offload_layers_to_cpu=6,
            offload_layers_to_gpu=0,
            max_new_tokens=10,
        ),
        3: _ManualTestCase(
            test_id=3,
            model_id="llama3-8B-chat",
            sample_file="85k_sample.txt",
            prompt_mode="paper",
            user_prompt="Analyze papers above and find 3 common similarities.",
            use_disk_cache=True,
            offload_layers_to_cpu=2,
            offload_layers_to_gpu=0,
            max_new_tokens=10,
        ),
        4: _ManualTestCase(
            test_id=4,
            model_id="qwen3-next-80B",
            sample_file="45k_sample.txt",
            prompt_mode="paper",
            user_prompt="Analyze papers above and find 3 common similarities.",
            use_disk_cache=True,
            offload_layers_to_cpu=48,
            offload_layers_to_gpu=0,
            max_new_tokens=100,
        ),
        5: _ManualTestCase(
            test_id=5,
            model_id="qwen3-next-80B",
            sample_file="2k_sample.txt",
            prompt_mode="chat",
            user_prompt="Analyze chats above and write top 10 most popular questions (translate to english).",
            use_disk_cache=True,
            offload_layers_to_cpu=0,
            offload_layers_to_gpu=0,
            max_new_tokens=100,
        ),
        6: _ManualTestCase(
            test_id=6,
            model_id="gemma3-12B",
            sample_file="2k_sample.txt",
            prompt_mode="chat",
            user_prompt="Analyze chats above and write top 10 most popular questions (translate to english).",
            use_disk_cache=False,
            offload_layers_to_cpu=12,
            offload_layers_to_gpu=0,
            max_new_tokens=10,
        ),
    }


def _build_messages(sample_dir: Path, test_case: _ManualTestCase) -> tuple[str, str]:
    sample_text = file_get_contents(
        str((sample_dir / test_case.sample_file).expanduser().resolve())
    )
    if test_case.prompt_mode == "chat":
        return (
            f"[CHATS]:\n{sample_text}[/END CHATS]",
            test_case.user_prompt,
        )
    return sample_text, test_case.user_prompt


def run_test(
    *,
    models_dir: Path,
    cache_dir: Path,
    device: str,
    sample_dir: Path,
    test_case: _ManualTestCase,
) -> str:
    """Run one manual regression scenario.

    Args:
        models_dir (Path): Root models directory passed to `Inference.ini_model`.
        cache_dir (Path): Disk cache directory for cache-backed scenarios.
        device (str): Target torch device string.
        sample_dir (Path): Directory containing the sample prompt files.
        test_case (_ManualTestCase): Scenario definition to execute.

    Returns:
        str: Decoded assistant output.
    """
    inference = Inference(test_case.model_id, device=device, logging=True)
    inference.ini_model(
        models_dir=str(models_dir.expanduser().resolve()),
        force_download=False,
    )
    if test_case.offload_layers_to_gpu > 0:
        inference.offload_layers_to_gpu_cpu(
            gpu_layers_num=test_case.offload_layers_to_gpu,
            cpu_layers_num=test_case.offload_layers_to_cpu,
        )
    elif test_case.offload_layers_to_cpu > 0:
        inference.offload_layers_to_cpu(layers_num=test_case.offload_layers_to_cpu)
    past_key_values = None
    if test_case.use_disk_cache:
        past_key_values = inference.DiskCache(
            cache_dir=str(cache_dir.expanduser().resolve())
        )

    model = cast(_GenerativeModelProtocol, inference.model)
    tokenizer = cast(_ChatTokenizerProtocol, inference.tokenizer)
    system_message, user_message = _build_messages(sample_dir, test_case)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        reasoning_effort="minimal",
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    ).to(inference.device)
    text_streamer = TextStreamer(
        cast(AutoTokenizer, inference.tokenizer),
        skip_prompt=True,
        skip_special_tokens=False,
    )
    with torch.no_grad():
        print(
            f"\n\n#{test_case.test_id}.TestingStarted.{test_case.model_id}",
            datetime.now().strftime("%H:%M:%S"),
            "input_ids.shape:",
            input_ids.shape,
        )
        outputs = (
            model.generate(
                input_ids=input_ids,
                max_new_tokens=test_case.max_new_tokens,
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
    return answer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the manual regression suite."""
    parser = argparse.ArgumentParser(
        description="Run one or more manual long-context regression scenarios."
    )
    parser.add_argument(
        "--test-id",
        dest="test_ids",
        action="append",
        type=int,
        help="Specific test ID to run. May be provided more than once.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Root directory containing the local model folders.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./kv_cache"),
        help="Disk cache directory used by cache-backed scenarios.",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path("./samples"),
        help="Directory containing the sample prompt files.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the selected manual regression scenarios."""
    args = parse_args()
    test_cases = _test_cases()
    selected_ids = _DEFAULT_TEST_IDS if not args.test_ids else tuple(args.test_ids)
    for test_id in selected_ids:
        if test_id not in test_cases:
            raise ValueError(f"Unsupported test id: {test_id}")
        run_test(
            models_dir=args.models_dir,
            cache_dir=args.cache_dir,
            device=args.device,
            sample_dir=args.sample_dir,
            test_case=test_cases[test_id],
        )
        print(f"#{test_id}.TestSuccess")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

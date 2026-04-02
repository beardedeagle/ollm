"""Manual utility entrypoints for safe local experiments and explicit Hub uploads."""

import argparse
import time
from pathlib import Path

import safetensors.torch as safetensors_torch
import torch
from huggingface_hub import HfApi, create_repo

from ollm.async_io import path_mkdir, torch_load_file, torch_save_file


def benchmark_tensor_storage(
    *,
    output_dir: Path,
    expert_count: int,
    hidden_size: int,
    expert_ids: tuple[int, ...],
) -> None:
    """Compare `.pt` shards against one shared safetensors file.

    Args:
        output_dir (Path): Directory used for generated benchmark artifacts.
        expert_count (int): Number of fake experts to generate.
        hidden_size (int): Square tensor size for each expert weight.
        expert_ids (tuple[int, ...]): Expert IDs to load during the benchmark.
    """
    pt_dir = output_dir / "experts_pt"
    safetensors_path = output_dir / "experts_all.safetensors"
    path_mkdir(pt_dir, parents=True, exist_ok=True)

    experts = {
        f"layer1.expert{expert_id}": torch.randn(hidden_size, hidden_size)
        for expert_id in range(expert_count)
    }
    for name, tensor in experts.items():
        torch_save_file(tensor, pt_dir / f"{name}.pt")
    safetensors_torch.save_file(experts, str(safetensors_path))

    started_at = time.perf_counter()
    for expert_id in expert_ids:
        torch_load_file(pt_dir / f"layer1.expert{expert_id}.pt", map_location="cpu")
    print(f"PT load time: {time.perf_counter() - started_at:.6f} sec")

    started_at = time.perf_counter()
    tensors = safetensors_torch.load_file(str(safetensors_path))
    for expert_id in expert_ids:
        _ = tensors[f"layer1.expert{expert_id}"]
    print(f"Safetensors load time: {time.perf_counter() - started_at:.6f} sec")


def push_folder_to_hub(
    *,
    folder_path: Path,
    repo_id: str,
    private: bool,
    confirm_upload: bool,
) -> None:
    """Upload a local model folder to the Hugging Face Hub.

    Args:
        folder_path (Path): Local folder to upload.
        repo_id (str): Target Hub repository in `owner/name` form.
        private (bool): Whether the destination repository should be private.
        confirm_upload (bool): Explicit safety switch for the upload action.
    """
    if not confirm_upload:
        raise ValueError("Refusing to upload without --confirm-upload")
    resolved_folder = folder_path.expanduser().resolve()
    if not resolved_folder.is_dir():
        raise ValueError(f"Folder does not exist: {resolved_folder}")

    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    HfApi().upload_large_folder(
        folder_path=str(resolved_folder),
        repo_id=repo_id,
        repo_type="model",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the manual utility script."""
    parser = argparse.ArgumentParser(
        description="Safe manual utilities for local storage benchmarks and Hub uploads."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser(
        "benchmark-storage",
        help="Benchmark per-expert `.pt` files against one safetensors file.",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where benchmark artifacts should be written.",
    )
    benchmark_parser.add_argument(
        "--expert-count",
        type=int,
        default=512,
        help="Number of fake experts to generate.",
    )
    benchmark_parser.add_argument(
        "--hidden-size",
        type=int,
        default=1024,
        help="Square tensor dimension for generated fake experts.",
    )
    benchmark_parser.add_argument(
        "--expert-ids",
        default="1,3,9",
        help="Comma-separated expert IDs to load during the benchmark.",
    )

    upload_parser = subparsers.add_parser(
        "push-to-hub",
        help="Upload a local model folder to the Hugging Face Hub.",
    )
    upload_parser.add_argument(
        "--folder-path",
        type=Path,
        required=True,
        help="Local folder containing the model artifacts to upload.",
    )
    upload_parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination repository in `owner/name` form.",
    )
    upload_parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the destination repository as private.",
    )
    upload_parser.add_argument(
        "--confirm-upload",
        action="store_true",
        help="Required acknowledgement for the upload side effect.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the selected utility subcommand."""
    args = parse_args()
    if args.command == "benchmark-storage":
        benchmark_tensor_storage(
            output_dir=args.output_dir,
            expert_count=args.expert_count,
            hidden_size=args.hidden_size,
            expert_ids=tuple(
                int(raw_value)
                for raw_value in args.expert_ids.split(",")
                if raw_value.strip()
            ),
        )
        return 0

    push_folder_to_hub(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        private=args.private,
        confirm_upload=args.confirm_upload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export selected Llama-family weights into raw tensor files plus a manifest."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ollm.async_io import path_mkdir, path_write_text

_EXPORT_PREFIXES = ("model.layers", "transformer.h")


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def export_llama_weights(
    *,
    model_id: str,
    out_dir: Path,
    torch_dtype_name: str,
) -> int:
    """Export selected model weights into raw `.bin` tensor files.

    Args:
        model_id (str): Hugging Face model ID or local model path.
        out_dir (Path): Output directory for exported tensors and the manifest.
        torch_dtype_name (str): Dtype selector passed to `from_pretrained`.

    Returns:
        int: Number of exported tensors.
    """
    resolved_out_dir = out_dir.expanduser().resolve()
    path_mkdir(resolved_out_dir, parents=True, exist_ok=True)

    state_dict = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=None,
        torch_dtype=_resolve_torch_dtype(torch_dtype_name),
        low_cpu_mem_usage=True,
    ).state_dict()

    manifest: dict[str, dict[str, object]] = {}
    for name, tensor in state_dict.items():
        if not name.startswith(_EXPORT_PREFIXES):
            continue
        contiguous_tensor = tensor.to("cpu").contiguous()
        filename = f"{name.replace('.', '__')}.bin"
        output_path = resolved_out_dir / filename
        path_mkdir(output_path.parent, parents=True, exist_ok=True)
        contiguous_tensor.numpy().tofile(output_path)
        manifest[name] = {
            "path": filename,
            "dtype": str(contiguous_tensor.dtype).replace("torch.", ""),
            "shape": list(contiguous_tensor.shape),
        }

    path_write_text(
        resolved_out_dir / "manifest.json",
        json.dumps(manifest, indent=2),
    )
    return len(manifest)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Llama export utility."""
    parser = argparse.ArgumentParser(
        description="Export selected Llama-family weights into raw tensor files."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model ID or local path to export from.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination directory for exported weights.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="float16",
        help="Torch dtype used while loading the source model.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the export utility."""
    args = parse_args()
    exported_count = export_llama_weights(
        model_id=args.model_id,
        out_dir=args.out_dir,
        torch_dtype_name=args.torch_dtype,
    )
    print(f"Exported {exported_count} tensors to {args.out_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

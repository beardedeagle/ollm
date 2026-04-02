"""Export GPT-OSS layer weights and packed metadata into oLLM's GDS format."""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import safe_open
from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config

from ollm.async_io import path_mkdir, path_write_text, torch_save_file

_DEFAULT_SHARD_STEMS = (
    "model-00000-of-00002",
    "model-00001-of-00002",
    "model-00002-of-00002",
)
_EXPORT_PREFIXES = ("model.layers", "transformer.h")
FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def convert_moe_packed_tensors(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """Dequantize GPT-OSS MXFP4 expert tensors into dense weights.

    Args:
        blocks (torch.Tensor): Packed MXFP4 block values.
        scales (torch.Tensor): Packed exponent values.
        dtype (torch.dtype): Output dtype for the dense tensor.
        rows_per_chunk (int): Chunk size used while expanding packed rows.

    Returns:
        torch.Tensor: Dense dequantized tensor.
    """
    scales = scales.to(torch.int32) - 127
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(
            f"Packed tensor shape mismatch: {blocks.shape[:-1]} vs {scales.shape}"
        )

    lookup_table = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
    *prefix_shape, groups, packed_width = blocks.shape
    rows_total = math.prod(prefix_shape) * groups

    reshaped_blocks = blocks.reshape(rows_total, packed_width)
    reshaped_scales = scales.reshape(rows_total, 1)
    output = torch.empty(
        rows_total,
        packed_width * 2,
        dtype=dtype,
        device=blocks.device,
    )
    for row_start in range(0, rows_total, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows_total)
        packed_chunk = reshaped_blocks[row_start:row_end]
        exponent_chunk = reshaped_scales[row_start:row_end]
        lower_indices = (packed_chunk & 0x0F).to(torch.long)
        upper_indices = (packed_chunk >> 4).to(torch.long)
        output_chunk = output[row_start:row_end]
        output_chunk[:, 0::2] = lookup_table[lower_indices]
        output_chunk[:, 1::2] = lookup_table[upper_indices]
        torch.ldexp(output_chunk, exponent_chunk, out=output_chunk)

    return (
        output.reshape(*prefix_shape, groups, packed_width * 2)
        .view(*prefix_shape, groups * packed_width * 2)
        .transpose(1, 2)
        .contiguous()
    )


def load_tensor_specs(
    model_dir: Path,
    shard_stems: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Load original safetensor entries used to preserve packed GPT-OSS tensors.

    Args:
        model_dir (Path): Directory containing the original shard files.
        shard_stems (tuple[str, ...]): Shard basenames without `.safetensors`.

    Returns:
        dict[str, torch.Tensor]: Mapping from tensor name to tensor value.
    """
    tensor_specs: dict[str, torch.Tensor] = {}
    for shard_stem in shard_stems:
        shard_path = model_dir / f"{shard_stem}.safetensors"
        with safe_open(shard_path, framework="pt") as handle:
            for key in handle.keys():
                tensor_specs[key] = handle.get_tensor(key)
    return tensor_specs


def export_gpt_oss_weights(
    *,
    model_dir: Path,
    out_dir: Path,
    shard_stems: tuple[str, ...],
) -> int:
    """Export GPT-OSS weights into `.pt` files plus a manifest.

    Args:
        model_dir (Path): Source model directory containing the original shards.
        out_dir (Path): Destination directory for exported weights.
        shard_stems (tuple[str, ...]): Shard basenames without `.safetensors`.

    Returns:
        int: Number of exported tensors.
    """
    resolved_model_dir = model_dir.expanduser().resolve()
    resolved_out_dir = out_dir.expanduser().resolve()
    path_mkdir(resolved_out_dir, parents=True, exist_ok=True)

    tensor_specs = load_tensor_specs(resolved_model_dir, shard_stems)
    state_dict = AutoModelForCausalLM.from_pretrained(
        str(resolved_model_dir),
        quantization_config=Mxfp4Config(dequantize=False),
        device_map="cpu",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    ).state_dict()

    manifest: dict[str, dict[str, object]] = {}
    for name, tensor in state_dict.items():
        if not name.startswith(_EXPORT_PREFIXES):
            continue
        exported_tensor: torch.Tensor | dict[str, torch.Tensor]
        exported_tensor = tensor.to("cpu").contiguous()
        packed: str | None = None
        dtype = str(exported_tensor.dtype)
        shape = list(exported_tensor.shape)
        blocks = tensor_specs.get(f"{name}_blocks")
        scales = tensor_specs.get(f"{name}_scales")
        if blocks is not None and scales is not None:
            exported_tensor = {"_blocks": blocks, "_scales": scales}
            packed = "mxfp4"

        filename = f"{name.replace('.', '__')}.pt"
        output_path = resolved_out_dir / filename
        path_mkdir(output_path.parent, parents=True, exist_ok=True)
        torch_save_file(exported_tensor, output_path)
        manifest[name] = {
            "path": filename,
            "dtype": dtype,
            "shape": shape,
            "packed": packed,
        }

    path_write_text(
        resolved_out_dir / "manifest.json",
        json.dumps(manifest, indent=2),
    )
    return len(manifest)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the GPT-OSS export script."""
    parser = argparse.ArgumentParser(
        description="Export GPT-OSS GDS weights from a local model directory."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Source model directory containing the original safetensor shards.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination directory for exported `.pt` weights.",
    )
    parser.add_argument(
        "--shard-stems",
        default=",".join(_DEFAULT_SHARD_STEMS),
        help="Comma-separated shard basenames without the `.safetensors` suffix.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the GPT-OSS export script."""
    args = parse_args()
    shard_stems = tuple(
        shard_name.strip()
        for shard_name in args.shard_stems.split(",")
        if shard_name.strip()
    )
    exported_count = export_gpt_oss_weights(
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        shard_stems=shard_stems,
    )
    print(f"Exported {exported_count} tensors to {args.out_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export Qwen3-Next layers and experts into oLLM's local tensor format."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

from ollm.async_io import path_mkdir, path_read_text, path_write_text, torch_save_file


def _read_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    return json.loads(path_read_text(index_path, encoding="utf-8"))["weight_map"]


def generate_manifest(
    *,
    weight_map: dict[str, str],
    out_dir: Path,
    num_hidden_layers: int,
    num_experts: int,
) -> None:
    """Generate the Qwen3-Next manifest describing per-layer export contents.

    Args:
        weight_map (dict[str, str]): Source safetensors weight map.
        out_dir (Path): Output directory for the generated manifest.
        num_hidden_layers (int): Number of decoder layers to export.
        num_experts (int): Number of experts per layer.
    """
    manifest: dict[str, list[str]] = {}
    for layer_idx in range(num_hidden_layers):
        base = f"model.layers.{layer_idx}."
        manifest[base] = [
            manifest_name.replace(base, "")
            for manifest_name in weight_map
            if manifest_name.startswith(base) and ".mlp.experts." not in manifest_name
        ]

    for layer_idx in range(num_hidden_layers):
        for expert_idx in range(num_experts):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
            manifest[base] = [
                manifest_name.replace(base, "")
                for manifest_name in weight_map
                if manifest_name.startswith(base)
            ]

    path_write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2))


def export_nonlayer_weights(
    *, model_dir: Path, out_dir: Path, weight_map: dict[str, str]
) -> None:
    """Export non-layer weights such as embeddings and output heads.

    Args:
        model_dir (Path): Source model directory.
        out_dir (Path): Destination export directory.
        weight_map (dict[str, str]): Source safetensors weight map.
    """
    exported_tensors: dict[str, torch.Tensor] = {}
    for manifest_name, filename in weight_map.items():
        if "model.layers" in manifest_name:
            continue
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            exported_tensors[manifest_name] = handle.get_tensor(manifest_name)
    save_file(exported_tensors, out_dir / "model.safetensors")


def export_layer_weights(
    *,
    model_dir: Path,
    out_dir: Path,
    weight_map: dict[str, str],
    num_hidden_layers: int,
) -> None:
    """Export dense decoder-layer weights as one `.pt` file per layer.

    Args:
        model_dir (Path): Source model directory.
        out_dir (Path): Destination export directory.
        weight_map (dict[str, str]): Source safetensors weight map.
        num_hidden_layers (int): Number of decoder layers to export.
    """
    for layer_idx in range(num_hidden_layers):
        base = f"model.layers.{layer_idx}."
        exported_layer: dict[str, object] = {}
        for manifest_name, filename in weight_map.items():
            if not manifest_name.startswith(base):
                continue
            if ".mlp.experts." in manifest_name:
                continue
            attr_name = manifest_name.replace(base, "")
            with safe_open(
                model_dir / filename, framework="pt", device="cpu"
            ) as handle:
                exported_layer[attr_name] = handle.get_tensor(manifest_name)
        torch_save_file(exported_layer, out_dir / f"{base.replace('.', '__')}.pt")


def export_expert_weights(
    *,
    model_dir: Path,
    out_dir: Path,
    weight_map: dict[str, str],
    num_hidden_layers: int,
    num_experts: int,
) -> None:
    """Export per-expert tensors as one `.pt` file per expert group.

    Args:
        model_dir (Path): Source model directory.
        out_dir (Path): Destination export directory.
        weight_map (dict[str, str]): Source safetensors weight map.
        num_hidden_layers (int): Number of decoder layers to export.
        num_experts (int): Number of experts per layer.
    """
    for layer_idx in range(num_hidden_layers):
        for expert_idx in range(num_experts):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
            exported_expert: dict[str, object] = {}
            for manifest_name, filename in weight_map.items():
                if not manifest_name.startswith(base):
                    continue
                attr_name = manifest_name.replace(base, "")
                with safe_open(
                    model_dir / filename, framework="pt", device="cpu"
                ) as handle:
                    exported_expert[attr_name] = handle.get_tensor(manifest_name)
            torch_save_file(
                exported_expert,
                out_dir / f"{base.replace('.', '__')}.pt",
            )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Qwen3-Next export utility."""
    parser = argparse.ArgumentParser(
        description="Export Qwen3-Next layers, experts, and manifests."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Source model directory containing `model.safetensors.index.json`.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination export directory.",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        required=True,
        help="Number of decoder layers in the source model.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        required=True,
        help="Number of experts per decoder layer.",
    )
    parser.add_argument(
        "--generate-manifest",
        action="store_true",
        help="Write the per-layer manifest.",
    )
    parser.add_argument(
        "--export-nonlayer",
        action="store_true",
        help="Export non-layer weights into `model.safetensors`.",
    )
    parser.add_argument(
        "--export-layers",
        action="store_true",
        help="Export dense decoder-layer weights.",
    )
    parser.add_argument(
        "--export-experts",
        action="store_true",
        help="Export per-expert weight bundles.",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Run all export stages.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the Qwen3-Next export utility."""
    args = parse_args()
    resolved_model_dir = args.model_dir.expanduser().resolve()
    resolved_out_dir = args.out_dir.expanduser().resolve()
    path_mkdir(resolved_out_dir, parents=True, exist_ok=True)
    weight_map = _read_weight_map(resolved_model_dir)
    run_all = args.export_all

    if run_all or args.generate_manifest:
        generate_manifest(
            weight_map=weight_map,
            out_dir=resolved_out_dir,
            num_hidden_layers=args.num_hidden_layers,
            num_experts=args.num_experts,
        )
    if run_all or args.export_nonlayer:
        export_nonlayer_weights(
            model_dir=resolved_model_dir,
            out_dir=resolved_out_dir,
            weight_map=weight_map,
        )
    if run_all or args.export_layers:
        export_layer_weights(
            model_dir=resolved_model_dir,
            out_dir=resolved_out_dir,
            weight_map=weight_map,
            num_hidden_layers=args.num_hidden_layers,
        )
    if run_all or args.export_experts:
        export_expert_weights(
            model_dir=resolved_model_dir,
            out_dir=resolved_out_dir,
            weight_map=weight_map,
            num_hidden_layers=args.num_hidden_layers,
            num_experts=args.num_experts,
        )
    if not (
        run_all
        or args.generate_manifest
        or args.export_nonlayer
        or args.export_layers
        or args.export_experts
    ):
        raise ValueError("Select at least one export stage or pass --export-all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

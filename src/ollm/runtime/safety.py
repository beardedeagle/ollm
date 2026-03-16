import json
from pathlib import Path

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

SAFE_ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"


def validate_safe_model_artifacts(model_path: Path) -> None:
    present_files = {
        file_path.name for file_path in model_path.iterdir() if file_path.is_file()
    }
    if present_files.intersection(
        {"pytorch_model.bin", "pytorch_model.bin.index.json"}
    ) or _contains_unsafe_weight_files(model_path):
        raise ValueError(
            f"The generic backend refuses unsafe model artifacts in {model_path}. "
            "Use safetensors weights only."
        )
    if not present_files.intersection(
        {SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME}
    ) and not any(model_path.glob("*.safetensors")):
        raise ValueError(
            f"The generic backend requires safetensors weights in {model_path}. "
            "Materialize or convert the model to safetensors before loading."
        )


def validate_safe_adapter_artifacts(adapter_dir: Path) -> None:
    present_files = {
        file_path.name for file_path in adapter_dir.iterdir() if file_path.is_file()
    }
    if "adapter_model.bin" in present_files or _contains_unsafe_weight_files(
        adapter_dir
    ):
        raise ValueError(
            f"The runtime refuses unsafe adapter artifacts in {adapter_dir}. "
            "Use safetensors adapter weights only."
        )
    if SAFE_ADAPTER_WEIGHTS_NAME not in present_files:
        raise ValueError(
            f"The runtime requires {SAFE_ADAPTER_WEIGHTS_NAME} in {adapter_dir}."
        )


def validate_safe_gds_export_artifacts(export_dir: Path) -> None:
    export_root = export_dir.expanduser().resolve()
    manifest_path = export_root / "manifest.json"
    if not manifest_path.exists() or not manifest_path.is_file():
        raise ValueError(
            f"The optimized gpt-oss specialization requires {manifest_path}."
        )

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, dict):
        raise ValueError(
            f"The optimized gpt-oss specialization requires a JSON object manifest in {manifest_path}."
        )

    for parameter_name, metadata in manifest.items():
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Invalid gds_export metadata for {parameter_name!r} in {manifest_path}."
            )
        relative_path = metadata.get("path")
        dtype = metadata.get("dtype")
        packed = metadata.get("packed")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(
                f"Invalid gds_export path for {parameter_name!r} in {manifest_path}."
            )

        file_path = (export_root / relative_path).resolve()
        if export_root not in file_path.parents:
            raise ValueError(
                f"gds_export entry {relative_path!r} escapes {export_root}."
            )
        if not file_path.exists() or not file_path.is_file():
            raise ValueError(
                f"gds_export entry {relative_path!r} does not exist in {export_root}."
            )
        if file_path.suffix in {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}:
            raise ValueError(
                f"The optimized gpt-oss specialization refuses unsafe gds_export artifacts in {export_root}. "
                "Use raw tensor exports instead of pickle-backed files."
            )
        if isinstance(dtype, str) and dtype.startswith("torch"):
            raise ValueError(
                f"The optimized gpt-oss specialization refuses torch-serialized gds_export artifacts in {export_root}."
            )
        if packed == "mxfp4":
            raise ValueError(
                f"The optimized gpt-oss specialization refuses packed torch gds_export artifacts in {export_root}."
            )


def _contains_unsafe_weight_files(model_path: Path) -> bool:
    for pattern in ("*.bin", "*.pt", "*.pth", "*.ckpt", "*.pkl", "*.pickle"):
        if any(model_path.glob(pattern)):
            return True
    return False

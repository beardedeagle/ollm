from pathlib import Path

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

SAFE_ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"


def validate_safe_model_artifacts(model_path: Path) -> None:
    present_files = {file_path.name for file_path in model_path.iterdir() if file_path.is_file()}
    if present_files.intersection({"pytorch_model.bin", "pytorch_model.bin.index.json"}) or _contains_unsafe_weight_files(model_path):
        raise ValueError(
            f"The generic backend refuses unsafe model artifacts in {model_path}. "
            "Use safetensors weights only."
        )
    if not present_files.intersection({SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME}) and not any(model_path.glob("*.safetensors")):
        raise ValueError(
            f"The generic backend requires safetensors weights in {model_path}. "
            "Materialize or convert the model to safetensors before loading."
        )


def validate_safe_adapter_artifacts(adapter_dir: Path) -> None:
    present_files = {file_path.name for file_path in adapter_dir.iterdir() if file_path.is_file()}
    if "adapter_model.bin" in present_files or _contains_unsafe_weight_files(adapter_dir):
        raise ValueError(
            f"The runtime refuses unsafe adapter artifacts in {adapter_dir}. "
            "Use safetensors adapter weights only."
        )
    if SAFE_ADAPTER_WEIGHTS_NAME not in present_files:
        raise ValueError(f"The runtime requires {SAFE_ADAPTER_WEIGHTS_NAME} in {adapter_dir}.")


def _contains_unsafe_weight_files(model_path: Path) -> bool:
    for pattern in ("*.bin", "*.pt", "*.pth", "*.ckpt", "*.pkl", "*.pickle"):
        if any(model_path.glob(pattern)):
            return True
    return False

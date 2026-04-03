"""Host and device helpers for runtime benchmark reporting."""

import platform

import torch


def build_host_summary() -> dict[str, object]:
    """Return basic host information for the benchmark report."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def choose_default_device() -> str:
    """Choose the default benchmark device for the current host."""

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

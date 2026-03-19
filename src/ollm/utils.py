import statistics
import time
from pathlib import Path

import torch

from ollm.async_io import path_read_text, path_write_text

type StatsSummary = dict[str, int | float]


def file_put_contents(filename, st):
    path_write_text(Path(filename), st, encoding="utf-8")


def file_get_contents(name):
    return path_read_text(Path(name), encoding="utf-8")


def tensor_size_gb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / 1024**3


class Stats:
    def __init__(self) -> None:
        self._samples_seconds: dict[str, list[float]] = {}

    def set(self, name: str, started_at: float) -> None:
        elapsed_seconds = time.perf_counter() - started_at
        self.record_elapsed_seconds(name, elapsed_seconds)

    def record_elapsed_seconds(self, name: str, elapsed_seconds: float) -> None:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be zero or greater")
        self._samples_seconds.setdefault(name, []).append(elapsed_seconds)

    def clear(self) -> None:
        self._samples_seconds.clear()

    def snapshot_seconds(self) -> dict[str, tuple[float, ...]]:
        return {name: tuple(samples) for name, samples in self._samples_seconds.items()}

    def collect_and_clear_ms(self) -> dict[str, StatsSummary]:
        snapshot = self.snapshot_seconds()
        self.clear()
        return {
            name: _summarize_elapsed_samples_ms(samples)
            for name, samples in snapshot.items()
        }

    def print_and_clean(self) -> str:
        summaries = self.collect_and_clear_ms()
        if not summaries:
            return "Stats: none"
        parts = []
        for name in sorted(summaries):
            summary = summaries[name]
            parts.append(
                (
                    f"{name}: count={summary['count']} total_ms={summary['total_ms']} "
                    f"mean_ms={summary['mean_ms']} max_ms={summary['max_ms']}"
                )
            )
        return "Stats: " + ", ".join(parts)


def _summarize_elapsed_samples_ms(samples_seconds: tuple[float, ...]) -> StatsSummary:
    samples_ms = sorted(sample * 1000.0 for sample in samples_seconds)
    p95_index = max(0, int(round((len(samples_ms) - 1) * 0.95)))
    total_ms = round(sum(samples_ms), 6)
    return {
        "count": len(samples_ms),
        "total_ms": total_ms,
        "min_ms": round(samples_ms[0], 6),
        "median_ms": round(statistics.median(samples_ms), 6),
        "p95_ms": round(samples_ms[p95_index], 6),
        "max_ms": round(samples_ms[-1], 6),
        "mean_ms": round(statistics.fmean(samples_ms), 6),
    }


# === Helper utilities ===
def _walk_to_parent(obj, attr_path):
    """Return (parent_obj, leaf_name) for attr_path like 'self_attn.q_proj.weight'"""
    parts = attr_path.split(".")
    parent = obj
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _assign_tensor_to_module(target_parent, leaf, tensor):
    """
    Assign a tensor into target_parent.<leaf>.
    - If target_parent.<leaf> has a .load call, call it with tensor.
    - Else, if attribute endswith 'weight' or 'bias' and current attr is nn.Parameter, replace it.
    - Else, set attribute to nn.Parameter(tensor) (read-only).
    """
    existing = getattr(target_parent, leaf, None)

    # If target object has a load(tensor) method (user's custom modules), call it.
    if hasattr(existing, "load") and callable(getattr(existing, "load")):
        existing.load(tensor)  # user-supplied API
        return

    # If existing is a Parameter (typical), replace with new Parameter on CUDA
    if (
        isinstance(existing, torch.nn.Parameter)
        or getattr(existing, "__class__", None) is torch.nn.Parameter
    ):
        param = torch.nn.Parameter(tensor.detach(), requires_grad=False)
        setattr(target_parent, leaf, param)
        return

    # If attribute is a module (like a Linear) we attempt to set its weight/bias
    if isinstance(existing, torch.nn.Linear) or hasattr(existing, "weight"):
        # try to set weight and bias if given tensor is 2D weight
        if tensor.ndim == 2 and hasattr(existing, "weight"):
            existing.weight = torch.nn.Parameter(tensor.detach(), requires_grad=False)
            return
        # fallback: set attribute to Parameter
    # Default fallback: replace attribute with a Parameter
    setattr(
        target_parent, leaf, torch.nn.Parameter(tensor.detach(), requires_grad=False)
    )


def _set_meta_placeholder(target_parent, leaf):
    """Replace parameter/module attribute with a tiny meta-device Parameter to free VRAM."""
    placeholder = torch.nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)
    setattr(target_parent, leaf, placeholder)


def remove_layers_weights(model):
    # 2. Remove heavy decoder block weights (keep skeleton)
    for layer in model.model.layers:
        for name, module in layer.named_children():
            if hasattr(module, "weight"):
                module.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = torch.nn.Parameter(torch.empty(0), requires_grad=False)

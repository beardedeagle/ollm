import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from ollm.async_io import subprocess_popen_process
from ollm.runtime.benchmark_process_memory import current_process_memory_snapshot


@dataclass(frozen=True, slots=True)
class NumericSummary:
    min: float
    median: float
    p95: float
    max: float
    mean: float

    def to_dict(self) -> dict[str, float]:
        return {
            "min": self.min,
            "median": self.median,
            "p95": self.p95,
            "max": self.max,
            "mean": self.mean,
        }


@dataclass(frozen=True, slots=True)
class AcceleratorUtilizationSnapshot:
    gpu_utilization_percent: NumericSummary | None
    memory_utilization_percent: NumericSummary | None

    def to_dict(self) -> dict[str, dict[str, float] | None]:
        return {
            "gpu_utilization_percent": None
            if self.gpu_utilization_percent is None
            else self.gpu_utilization_percent.to_dict(),
            "memory_utilization_percent": None
            if self.memory_utilization_percent is None
            else self.memory_utilization_percent.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StageResourceSnapshot:
    current_rss_mb: float | None
    peak_rss_mb: float | None
    peak_rss_source: str | None
    accelerator_kind: str | None
    accelerator_current_mb: float | None
    accelerator_peak_mb: float | None
    accelerator_reserved_mb: float | None
    accelerator_peak_reserved_mb: float | None
    accelerator_peak_source: str | None
    process_cpu_utilization_percent: float | None
    accelerator_utilization: AcceleratorUtilizationSnapshot | None

    def to_dict(self) -> dict[str, object]:
        return {
            "current_rss_mb": self.current_rss_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "peak_rss_source": self.peak_rss_source,
            "accelerator_kind": self.accelerator_kind,
            "accelerator_current_mb": self.accelerator_current_mb,
            "accelerator_peak_mb": self.accelerator_peak_mb,
            "accelerator_reserved_mb": self.accelerator_reserved_mb,
            "accelerator_peak_reserved_mb": self.accelerator_peak_reserved_mb,
            "accelerator_peak_source": self.accelerator_peak_source,
            "process_cpu_utilization_percent": self.process_cpu_utilization_percent,
            "accelerator_utilization": None
            if self.accelerator_utilization is None
            else self.accelerator_utilization.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _AcceleratorMemorySnapshot:
    accelerator_kind: str | None
    current_mb: float | None
    peak_mb: float | None
    reserved_mb: float | None
    peak_reserved_mb: float | None
    peak_source: str | None


class _ResourceSampler:
    def __init__(self, device: str, *, interval_seconds: float = 0.01):
        self._device = device
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._max_current_rss_mb: float | None = None
        self._max_accelerator_current_mb: float | None = None
        self._max_accelerator_reserved_mb: float | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        return {
            "peak_rss_mb": self._max_current_rss_mb,
            "peak_accelerator_current_mb": self._max_accelerator_current_mb,
            "peak_accelerator_reserved_mb": self._max_accelerator_reserved_mb,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            process_snapshot = current_process_memory_snapshot()
            if process_snapshot.current_rss_mb is not None:
                self._max_current_rss_mb = _max_optional(
                    self._max_current_rss_mb, process_snapshot.current_rss_mb
                )
            accelerator_snapshot = capture_accelerator_memory(self._device)
            if accelerator_snapshot.current_mb is not None:
                self._max_accelerator_current_mb = _max_optional(
                    self._max_accelerator_current_mb, accelerator_snapshot.current_mb
                )
            if accelerator_snapshot.reserved_mb is not None:
                self._max_accelerator_reserved_mb = _max_optional(
                    self._max_accelerator_reserved_mb,
                    accelerator_snapshot.reserved_mb,
                )
            self._stop_event.wait(self._interval_seconds)


class _NvidiaUtilizationSampler:
    def __init__(self, device: str, *, interval_milliseconds: int = 100):
        self._device = torch.device(device)
        self._interval_milliseconds = interval_milliseconds
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._gpu_samples: list[float] = []
        self._memory_samples: list[float] = []

    def start(self) -> bool:
        if self._device.type != "cuda":
            return False
        if shutil.which("nvidia-smi") is None:
            return False
        device_index = 0 if self._device.index is None else self._device.index
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
            "-i",
            str(device_index),
            "-lms",
            str(self._interval_milliseconds),
        ]
        try:
            self._process = subprocess_popen_process(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            self._process = None
            return False
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()
        return True

    def stop(self) -> AcceleratorUtilizationSnapshot | None:
        if self._process is None:
            return None
        self._process.terminate()
        try:
            self._process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=1.0)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        self._process = None
        gpu_summary = summarize_optional_numeric_values(self._gpu_samples)
        memory_summary = summarize_optional_numeric_values(self._memory_samples)
        return AcceleratorUtilizationSnapshot(
            gpu_utilization_percent=gpu_summary,
            memory_utilization_percent=memory_summary,
        )

    def _read_output(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2:
                continue
            gpu_utilization = _parse_optional_percent(fields[0])
            memory_utilization = _parse_optional_percent(fields[1])
            if gpu_utilization is not None:
                self._gpu_samples.append(gpu_utilization)
            if memory_utilization is not None:
                self._memory_samples.append(memory_utilization)


def measure_stage(
    device: str,
    operation: Callable[[], object],
    *,
    sample_accelerator_utilization: bool = False,
) -> tuple[object, float, StageResourceSnapshot]:
    resource_sampler = _ResourceSampler(device)
    utilization_sampler = (
        _NvidiaUtilizationSampler(device) if sample_accelerator_utilization else None
    )
    reset_accelerator_metrics(device)
    resource_sampler.start()
    utilization_enabled = False
    if utilization_sampler is not None:
        utilization_enabled = utilization_sampler.start()
    cpu_started = time.process_time()
    started = time.perf_counter()
    try:
        result = operation()
        synchronize_device(device)
    finally:
        elapsed_seconds = time.perf_counter() - started
        cpu_elapsed = time.process_time() - cpu_started
        sampled_resources = resource_sampler.stop()
        utilization = (
            None
            if utilization_sampler is None or not utilization_enabled
            else utilization_sampler.stop()
        )
    process_snapshot = current_process_memory_snapshot()
    accelerator_snapshot = capture_accelerator_memory(device)
    peak_rss_mb = sampled_resources["peak_rss_mb"]
    peak_rss_source = "sampled"
    if peak_rss_mb is None:
        peak_rss_mb = process_snapshot.peak_rss_mb
        peak_rss_source = process_snapshot.peak_rss_source
    accelerator_peak_mb = accelerator_snapshot.peak_mb
    accelerator_peak_reserved_mb = accelerator_snapshot.peak_reserved_mb
    accelerator_peak_source = accelerator_snapshot.peak_source
    if (
        accelerator_peak_mb is None
        and sampled_resources["peak_accelerator_current_mb"] is not None
    ):
        accelerator_peak_mb = sampled_resources["peak_accelerator_current_mb"]
        accelerator_peak_source = "sampled"
    if (
        accelerator_peak_reserved_mb is None
        and sampled_resources["peak_accelerator_reserved_mb"] is not None
    ):
        accelerator_peak_reserved_mb = sampled_resources["peak_accelerator_reserved_mb"]
        if accelerator_peak_source is None:
            accelerator_peak_source = "sampled"
    process_cpu_utilization = None
    if elapsed_seconds > 0:
        process_cpu_utilization = round(
            (cpu_elapsed / elapsed_seconds) * 100.0,
            6,
        )
    snapshot = StageResourceSnapshot(
        current_rss_mb=process_snapshot.current_rss_mb,
        peak_rss_mb=peak_rss_mb,
        peak_rss_source=peak_rss_source,
        accelerator_kind=accelerator_snapshot.accelerator_kind,
        accelerator_current_mb=accelerator_snapshot.current_mb,
        accelerator_peak_mb=accelerator_peak_mb,
        accelerator_reserved_mb=accelerator_snapshot.reserved_mb,
        accelerator_peak_reserved_mb=accelerator_peak_reserved_mb,
        accelerator_peak_source=accelerator_peak_source,
        process_cpu_utilization_percent=process_cpu_utilization,
        accelerator_utilization=utilization,
    )
    return result, round(elapsed_seconds * 1000.0, 6), snapshot


def reset_accelerator_metrics(device: str) -> None:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(resolved_device)


def capture_accelerator_memory(device: str) -> _AcceleratorMemorySnapshot:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved_device)
        return _AcceleratorMemorySnapshot(
            accelerator_kind="cuda",
            current_mb=bytes_to_mb(float(torch.cuda.memory_allocated(resolved_device))),
            peak_mb=bytes_to_mb(
                float(torch.cuda.max_memory_allocated(resolved_device))
            ),
            reserved_mb=bytes_to_mb(float(torch.cuda.memory_reserved(resolved_device))),
            peak_reserved_mb=bytes_to_mb(
                float(torch.cuda.max_memory_reserved(resolved_device))
            ),
            peak_source="native",
        )
    if (
        resolved_device.type == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        current_allocated = _mps_memory_stat("current_allocated_memory")
        driver_allocated = _mps_memory_stat("driver_allocated_memory")
        return _AcceleratorMemorySnapshot(
            accelerator_kind="mps",
            current_mb=current_allocated,
            peak_mb=None,
            reserved_mb=driver_allocated,
            peak_reserved_mb=None,
            peak_source=None,
        )
    return _AcceleratorMemorySnapshot(
        accelerator_kind=None,
        current_mb=None,
        peak_mb=None,
        reserved_mb=None,
        peak_reserved_mb=None,
        peak_source=None,
    )


def synchronize_device(device: str) -> None:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved_device)
        return
    if (
        resolved_device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "synchronize")
    ):
        torch.mps.synchronize()


def bytes_to_mb(value: float) -> float:
    return round(value / (1024.0 * 1024.0), 6)


def cache_dir_size_mb(cache_dir: Path) -> float | None:
    if not cache_dir.exists():
        return None
    total_bytes = 0
    for path in cache_dir.rglob("*"):
        if path.is_file():
            total_bytes += path.stat().st_size
    return bytes_to_mb(float(total_bytes))


def summarize_optional_numeric_values(
    values: list[float] | tuple[float, ...],
) -> NumericSummary | None:
    if not values:
        return None
    sorted_values = sorted(values)
    p95_index = max(0, int(round((len(sorted_values) - 1) * 0.95)))
    mean_value = round(sum(sorted_values) / len(sorted_values), 6)
    return NumericSummary(
        min=round(sorted_values[0], 6),
        median=round(_median(sorted_values), 6),
        p95=round(sorted_values[p95_index], 6),
        max=round(sorted_values[-1], 6),
        mean=mean_value,
    )


def _median(values: list[float]) -> float:
    midpoint = len(values) // 2
    if len(values) % 2 == 1:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _parse_optional_percent(value: str) -> float | None:
    value = value.strip()
    if not value or value == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _max_optional(left: float | None, right: float) -> float:
    if left is None:
        return right
    return max(left, right)


def _mps_memory_stat(name: str) -> float | None:
    if not hasattr(torch, "mps"):
        return None
    function = getattr(torch.mps, name, None)
    if function is None or not callable(function):
        return None
    try:
        value = function()
    except RuntimeError:
        return None
    if not isinstance(value, int | float):
        return None
    return bytes_to_mb(float(value))

"""Process memory snapshots for runtime benchmark reporting."""

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ollm.async_io import path_read_text, subprocess_run_process


@dataclass(frozen=True, slots=True)
class _ProcessMemorySnapshot:
    current_rss_mb: float | None
    peak_rss_mb: float | None
    peak_rss_source: str | None


def current_process_memory_snapshot() -> _ProcessMemorySnapshot:
    if sys.platform == "linux":
        current_rss_mb = _linux_current_rss_mb()
        peak_rss_mb = _resource_peak_rss_mb()
        return _ProcessMemorySnapshot(
            current_rss_mb=current_rss_mb,
            peak_rss_mb=peak_rss_mb,
            peak_rss_source="native" if peak_rss_mb is not None else None,
        )
    if sys.platform == "darwin":
        current_rss_mb = _darwin_current_rss_mb()
        peak_rss_mb = _resource_peak_rss_mb()
        return _ProcessMemorySnapshot(
            current_rss_mb=current_rss_mb,
            peak_rss_mb=peak_rss_mb,
            peak_rss_source="native" if peak_rss_mb is not None else None,
        )
    if sys.platform == "win32":
        return _windows_process_memory_snapshot()
    return _ProcessMemorySnapshot(None, None, None)


def _linux_current_rss_mb() -> float | None:
    try:
        status = path_read_text(Path("/proc/self/status"), encoding="utf-8")
    except OSError:
        return None
    for line in status.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) < 2:
            return None
        try:
            return round(int(fields[1]) / 1024.0, 6)
        except ValueError:
            return None
    return None


def _posix_ps_current_rss_mb() -> float | None:
    try:
        completed = subprocess_run_process(
            ("ps", "-o", "rss=", "-p", str(os.getpid())),
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    if not value:
        return None
    try:
        return round(int(value) / 1024.0, 6)
    except ValueError:
        return None


def _darwin_current_rss_mb() -> float | None:
    class _TimeValue(ctypes.Structure):
        _fields_ = [
            ("seconds", ctypes.c_int),
            ("microseconds", ctypes.c_int),
        ]

    class _MachTaskBasicInfo(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_max", ctypes.c_uint64),
            ("user_time", _TimeValue),
            ("system_time", _TimeValue),
            ("policy", ctypes.c_int),
            ("suspend_count", ctypes.c_int),
        ]

    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    except OSError:
        return _posix_ps_current_rss_mb()
    task_info = libc.task_info
    task_info.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    task_info.restype = ctypes.c_int
    mach_task_self = libc.mach_task_self
    mach_task_self.argtypes = []
    mach_task_self.restype = ctypes.c_uint32
    task_info_flavor = 5
    info = _MachTaskBasicInfo()
    count = ctypes.c_uint32(
        ctypes.sizeof(_MachTaskBasicInfo) // ctypes.sizeof(ctypes.c_int)
    )
    result = task_info(
        mach_task_self(),
        task_info_flavor,
        ctypes.byref(info),
        ctypes.byref(count),
    )
    if result != 0:
        return _posix_ps_current_rss_mb()
    return _bytes_to_mb(float(info.resident_size))


def _resource_peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    try:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, ValueError):
        return None
    if peak_rss <= 0:
        return None
    if sys.platform == "darwin":
        return _bytes_to_mb(float(peak_rss))
    return round(float(peak_rss) / 1024.0, 6)


def _windows_process_memory_snapshot() -> _ProcessMemorySnapshot:
    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return _ProcessMemorySnapshot(None, None, None)
    kernel32 = windll.kernel32
    psapi = windll.psapi
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    process = kernel32.GetCurrentProcess()
    success = psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    if success == 0:
        return _ProcessMemorySnapshot(None, None, None)
    return _ProcessMemorySnapshot(
        current_rss_mb=_bytes_to_mb(float(counters.WorkingSetSize)),
        peak_rss_mb=_bytes_to_mb(float(counters.PeakWorkingSetSize)),
        peak_rss_source="native",
    )


def _bytes_to_mb(value: float) -> float:
    return round(value / (1024.0 * 1024.0), 6)

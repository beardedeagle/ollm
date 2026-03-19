import ctypes
import os
import sys
from dataclasses import dataclass

import torch

_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024


@dataclass(slots=True, frozen=True)
class KVCacheResourceSnapshot:
    platform: str
    available_system_memory_bytes: int | None
    available_accelerator_memory_bytes: int | None


@dataclass(slots=True, frozen=True)
class KVCachePolicy:
    policy_id: str
    flush_token_threshold: int
    flush_byte_threshold: int

    def should_flush(
        self,
        *,
        pending_tokens: int,
        pending_bytes: int,
    ) -> bool:
        return (
            pending_tokens >= self.flush_token_threshold
            or pending_bytes >= self.flush_byte_threshold
        )


def detect_kv_cache_resources(device: torch.device) -> KVCacheResourceSnapshot:
    return KVCacheResourceSnapshot(
        platform=sys.platform,
        available_system_memory_bytes=_available_system_memory_bytes(),
        available_accelerator_memory_bytes=_available_accelerator_memory_bytes(device),
    )


def select_kv_cache_policy(
    device: torch.device,
    *,
    resource_snapshot: KVCacheResourceSnapshot | None = None,
) -> KVCachePolicy:
    snapshot = (
        detect_kv_cache_resources(device)
        if resource_snapshot is None
        else resource_snapshot
    )
    available_ram = snapshot.available_system_memory_bytes or 0
    available_accelerator = snapshot.available_accelerator_memory_bytes or 0

    if device.type == "cpu":
        if snapshot.platform == "darwin":
            if available_ram >= 16 * _GIB:
                return KVCachePolicy(
                    policy_id="darwin-cpu-buffered",
                    flush_token_threshold=128,
                    flush_byte_threshold=8 * _MIB,
                )
            return KVCachePolicy(
                policy_id="darwin-cpu-balanced",
                flush_token_threshold=64,
                flush_byte_threshold=4 * _MIB,
            )
        if snapshot.platform == "win32":
            return KVCachePolicy(
                policy_id="windows-cpu-balanced",
                flush_token_threshold=128,
                flush_byte_threshold=8 * _MIB,
            )
        return KVCachePolicy(
            policy_id="cpu-balanced",
            flush_token_threshold=128,
            flush_byte_threshold=8 * _MIB,
        )

    if device.type == "cuda":
        if snapshot.platform == "win32" and available_accelerator >= 12 * _GIB:
            return KVCachePolicy(
                policy_id="windows-cuda-balanced",
                flush_token_threshold=128,
                flush_byte_threshold=8 * _MIB,
            )
        return KVCachePolicy(
            policy_id="cuda-balanced",
            flush_token_threshold=64,
            flush_byte_threshold=4 * _MIB,
        )

    if device.type == "mps":
        return KVCachePolicy(
            policy_id="darwin-mps-buffered",
            flush_token_threshold=128,
            flush_byte_threshold=8 * _MIB,
        )

    return KVCachePolicy(
        policy_id="default-balanced",
        flush_token_threshold=64,
        flush_byte_threshold=4 * _MIB,
    )


def _available_system_memory_bytes() -> int | None:
    if sys.platform == "win32":
        return _windows_available_memory_bytes()
    page_size = _sysconf_int("SC_PAGE_SIZE")
    available_pages = _sysconf_int("SC_AVPHYS_PAGES")
    if page_size is None or available_pages is None:
        return None
    return page_size * available_pages


def _sysconf_int(name: str) -> int | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        value = os.sysconf(name)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, int) and value > 0 else None


def _windows_available_memory_bytes() -> int | None:
    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    kernel32 = getattr(ctypes, "windll", None)
    if kernel32 is None:
        return None
    if kernel32.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) == 0:
        return None
    return int(status.ullAvailPhys)


def _available_accelerator_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
    return int(free_bytes)

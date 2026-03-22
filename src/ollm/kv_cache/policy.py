import ctypes
import math
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
    write_back_retained_tokens: int = 64
    write_back_retained_bytes: int = 4 * _MIB
    journal_compaction_entry_threshold: int = 0

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

    def write_back_spill_token_count(
        self,
        *,
        pending_tokens: int,
        pending_bytes: int,
    ) -> int:
        if pending_tokens <= 0 or not self.should_flush(
            pending_tokens=pending_tokens,
            pending_bytes=pending_bytes,
        ):
            return 0

        spill_tokens = max(0, pending_tokens - self.write_back_retained_tokens)
        if pending_bytes > self.write_back_retained_bytes:
            bytes_per_token = max(1, math.ceil(pending_bytes / pending_tokens))
            spill_tokens = max(
                spill_tokens,
                math.ceil(
                    (pending_bytes - self.write_back_retained_bytes) / bytes_per_token
                ),
            )
        if spill_tokens <= 0:
            return 0
        return min(pending_tokens, spill_tokens)


def detect_kv_cache_resources(device: torch.device) -> KVCacheResourceSnapshot:
    return KVCacheResourceSnapshot(
        platform=sys.platform,
        available_system_memory_bytes=_available_system_memory_bytes(),
        available_accelerator_memory_bytes=available_accelerator_memory_bytes(device),
    )


def select_kv_cache_policy(
    device: torch.device,
    *,
    strategy: str | None = None,
    resource_snapshot: KVCacheResourceSnapshot | None = None,
) -> KVCachePolicy:
    snapshot = (
        detect_kv_cache_resources(device)
        if resource_snapshot is None
        else resource_snapshot
    )
    available_ram = snapshot.available_system_memory_bytes or 0
    available_accelerator = snapshot.available_accelerator_memory_bytes or 0
    if strategy == "tiered-write-back":
        return _select_tiered_write_back_policy(
            device=device,
            snapshot=snapshot,
            available_ram=available_ram,
            available_accelerator=available_accelerator,
        )
    if strategy == "log-structured-journal":
        return _select_log_structured_journal_policy(
            device=device,
            snapshot=snapshot,
            available_ram=available_ram,
            available_accelerator=available_accelerator,
        )
    if strategy == "sliding-window-ring-buffer":
        return _select_sliding_window_ring_buffer_policy(
            device=device,
            snapshot=snapshot,
        )
    if strategy == "quantized-cold-tier":
        return _select_quantized_cold_tier_policy(
            device=device,
            snapshot=snapshot,
            available_ram=available_ram,
            available_accelerator=available_accelerator,
        )

    if device.type == "cpu":
        if snapshot.platform == "darwin":
            if available_ram >= 16 * _GIB:
                return KVCachePolicy(
                    policy_id="darwin-cpu-buffered",
                    flush_token_threshold=128,
                    flush_byte_threshold=8 * _MIB,
                    write_back_retained_tokens=64,
                    write_back_retained_bytes=4 * _MIB,
                )
            return KVCachePolicy(
                policy_id="darwin-cpu-balanced",
                flush_token_threshold=64,
                flush_byte_threshold=4 * _MIB,
                write_back_retained_tokens=32,
                write_back_retained_bytes=2 * _MIB,
            )
        if snapshot.platform == "win32":
            return KVCachePolicy(
                policy_id="windows-cpu-balanced",
                flush_token_threshold=128,
                flush_byte_threshold=8 * _MIB,
                write_back_retained_tokens=64,
                write_back_retained_bytes=4 * _MIB,
            )
        return KVCachePolicy(
            policy_id="cpu-balanced",
            flush_token_threshold=128,
            flush_byte_threshold=8 * _MIB,
            write_back_retained_tokens=64,
            write_back_retained_bytes=4 * _MIB,
        )

    if device.type == "cuda":
        if snapshot.platform == "win32" and available_accelerator >= 12 * _GIB:
            return KVCachePolicy(
                policy_id="windows-cuda-balanced",
                flush_token_threshold=128,
                flush_byte_threshold=8 * _MIB,
                write_back_retained_tokens=64,
                write_back_retained_bytes=4 * _MIB,
            )
        return KVCachePolicy(
            policy_id="cuda-balanced",
            flush_token_threshold=64,
            flush_byte_threshold=4 * _MIB,
            write_back_retained_tokens=32,
            write_back_retained_bytes=2 * _MIB,
        )

    if device.type == "mps":
        return KVCachePolicy(
            policy_id="darwin-mps-buffered",
            flush_token_threshold=128,
            flush_byte_threshold=8 * _MIB,
            write_back_retained_tokens=64,
            write_back_retained_bytes=4 * _MIB,
        )

    return KVCachePolicy(
        policy_id="default-balanced",
        flush_token_threshold=64,
        flush_byte_threshold=4 * _MIB,
        write_back_retained_tokens=32,
        write_back_retained_bytes=2 * _MIB,
    )


def _select_tiered_write_back_policy(
    *,
    device: torch.device,
    snapshot: KVCacheResourceSnapshot,
    available_ram: int,
    available_accelerator: int,
) -> KVCachePolicy:
    if device.type == "cpu":
        if snapshot.platform == "darwin":
            if available_ram >= 16 * _GIB:
                return KVCachePolicy(
                    policy_id="darwin-cpu-tiered",
                    flush_token_threshold=24,
                    flush_byte_threshold=1 * _MIB,
                    write_back_retained_tokens=8,
                    write_back_retained_bytes=256 * 1024,
                )
            return KVCachePolicy(
                policy_id="darwin-cpu-tiered-compact",
                flush_token_threshold=16,
                flush_byte_threshold=768 * 1024,
                write_back_retained_tokens=4,
                write_back_retained_bytes=128 * 1024,
            )
        if snapshot.platform == "win32":
            return KVCachePolicy(
                policy_id="windows-cpu-tiered",
                flush_token_threshold=24,
                flush_byte_threshold=1 * _MIB,
                write_back_retained_tokens=8,
                write_back_retained_bytes=256 * 1024,
            )
        return KVCachePolicy(
            policy_id="cpu-tiered",
            flush_token_threshold=24,
            flush_byte_threshold=1 * _MIB,
            write_back_retained_tokens=8,
            write_back_retained_bytes=256 * 1024,
        )

    if device.type == "cuda":
        if snapshot.platform == "win32" and available_accelerator >= 12 * _GIB:
            return KVCachePolicy(
                policy_id="windows-cuda-tiered",
                flush_token_threshold=24,
                flush_byte_threshold=1 * _MIB,
                write_back_retained_tokens=8,
                write_back_retained_bytes=256 * 1024,
            )
        return KVCachePolicy(
            policy_id="cuda-tiered",
            flush_token_threshold=16,
            flush_byte_threshold=768 * 1024,
            write_back_retained_tokens=4,
            write_back_retained_bytes=128 * 1024,
        )

    if device.type == "mps":
        return KVCachePolicy(
            policy_id="darwin-mps-tiered",
            flush_token_threshold=16,
            flush_byte_threshold=768 * 1024,
            write_back_retained_tokens=4,
            write_back_retained_bytes=128 * 1024,
        )

    return KVCachePolicy(
        policy_id="default-tiered",
        flush_token_threshold=16,
        flush_byte_threshold=768 * 1024,
        write_back_retained_tokens=4,
        write_back_retained_bytes=128 * 1024,
    )


def _select_sliding_window_ring_buffer_policy(
    *,
    device: torch.device,
    snapshot: KVCacheResourceSnapshot,
) -> KVCachePolicy:
    if device.type == "cpu":
        if snapshot.platform == "darwin":
            return KVCachePolicy(
                policy_id="darwin-cpu-sliding-window",
                flush_token_threshold=1,
                flush_byte_threshold=1,
            )
        if snapshot.platform == "win32":
            return KVCachePolicy(
                policy_id="windows-cpu-sliding-window",
                flush_token_threshold=1,
                flush_byte_threshold=1,
            )
        return KVCachePolicy(
            policy_id="cpu-sliding-window",
            flush_token_threshold=1,
            flush_byte_threshold=1,
        )
    if device.type == "cuda":
        if snapshot.platform == "win32":
            return KVCachePolicy(
                policy_id="windows-cuda-sliding-window",
                flush_token_threshold=1,
                flush_byte_threshold=1,
            )
        return KVCachePolicy(
            policy_id="cuda-sliding-window",
            flush_token_threshold=1,
            flush_byte_threshold=1,
        )
    if device.type == "mps":
        return KVCachePolicy(
            policy_id="darwin-mps-sliding-window",
            flush_token_threshold=1,
            flush_byte_threshold=1,
        )
    return KVCachePolicy(
        policy_id="default-sliding-window",
        flush_token_threshold=1,
        flush_byte_threshold=1,
    )


def _select_log_structured_journal_policy(
    *,
    device: torch.device,
    snapshot: KVCacheResourceSnapshot,
    available_ram: int,
    available_accelerator: int,
) -> KVCachePolicy:
    compaction_entry_threshold = _journal_compaction_entry_threshold(
        available_ram=available_ram,
        available_accelerator=available_accelerator,
    )
    if device.type == "cpu":
        if snapshot.platform == "darwin":
            return KVCachePolicy(
                policy_id="darwin-cpu-journal",
                flush_token_threshold=64,
                flush_byte_threshold=4 * _MIB,
                write_back_retained_tokens=32,
                write_back_retained_bytes=2 * _MIB,
                journal_compaction_entry_threshold=compaction_entry_threshold,
            )
        if snapshot.platform == "win32":
            return KVCachePolicy(
                policy_id="windows-cpu-journal",
                flush_token_threshold=128,
                flush_byte_threshold=8 * _MIB,
                write_back_retained_tokens=64,
                write_back_retained_bytes=4 * _MIB,
                journal_compaction_entry_threshold=compaction_entry_threshold,
            )
        return KVCachePolicy(
            policy_id="cpu-journal",
            flush_token_threshold=128,
            flush_byte_threshold=8 * _MIB,
            write_back_retained_tokens=64,
            write_back_retained_bytes=4 * _MIB,
            journal_compaction_entry_threshold=compaction_entry_threshold,
        )

    if device.type == "cuda":
        if snapshot.platform == "win32" and available_accelerator >= 12 * _GIB:
            return KVCachePolicy(
                policy_id="windows-cuda-journal",
                flush_token_threshold=128,
                flush_byte_threshold=8 * _MIB,
                write_back_retained_tokens=64,
                write_back_retained_bytes=4 * _MIB,
                journal_compaction_entry_threshold=compaction_entry_threshold,
            )
        return KVCachePolicy(
            policy_id="cuda-journal",
            flush_token_threshold=64,
            flush_byte_threshold=4 * _MIB,
            write_back_retained_tokens=32,
            write_back_retained_bytes=2 * _MIB,
            journal_compaction_entry_threshold=compaction_entry_threshold,
        )

    if device.type == "mps":
        return KVCachePolicy(
            policy_id="darwin-mps-journal",
            flush_token_threshold=128,
            flush_byte_threshold=8 * _MIB,
            write_back_retained_tokens=64,
            write_back_retained_bytes=4 * _MIB,
            journal_compaction_entry_threshold=compaction_entry_threshold,
        )

    return KVCachePolicy(
        policy_id="default-journal",
        flush_token_threshold=64,
        flush_byte_threshold=4 * _MIB,
        write_back_retained_tokens=32,
        write_back_retained_bytes=2 * _MIB,
        journal_compaction_entry_threshold=compaction_entry_threshold,
    )


def _select_quantized_cold_tier_policy(
    *,
    device: torch.device,
    snapshot: KVCacheResourceSnapshot,
    available_ram: int,
    available_accelerator: int,
) -> KVCachePolicy:
    base_policy = _select_log_structured_journal_policy(
        device=device,
        snapshot=snapshot,
        available_ram=available_ram,
        available_accelerator=available_accelerator,
    )
    return KVCachePolicy(
        policy_id=base_policy.policy_id.replace("journal", "quantized-cold-tier"),
        flush_token_threshold=base_policy.flush_token_threshold,
        flush_byte_threshold=base_policy.flush_byte_threshold,
        write_back_retained_tokens=base_policy.write_back_retained_tokens,
        write_back_retained_bytes=base_policy.write_back_retained_bytes,
        journal_compaction_entry_threshold=base_policy.journal_compaction_entry_threshold,
    )


def _journal_compaction_entry_threshold(
    *,
    available_ram: int,
    available_accelerator: int,
) -> int:
    if available_accelerator >= 16 * _GIB or available_ram >= 32 * _GIB:
        return 6
    return 4


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


def available_accelerator_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
    return int(free_bytes)

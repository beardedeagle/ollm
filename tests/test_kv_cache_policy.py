import torch

from ollm.kv_cache_policy import KVCacheResourceSnapshot, select_kv_cache_policy


def test_select_kv_cache_policy_prefers_buffering_for_darwin_cpu() -> None:
    policy = select_kv_cache_policy(
        torch.device("cpu"),
        resource_snapshot=KVCacheResourceSnapshot(
            platform="darwin",
            available_system_memory_bytes=32 * 1024 * 1024 * 1024,
            available_accelerator_memory_bytes=None,
        ),
    )

    assert policy.policy_id == "darwin-cpu-buffered"
    assert policy.flush_token_threshold == 128


def test_select_kv_cache_policy_keeps_windows_cpu_balanced() -> None:
    policy = select_kv_cache_policy(
        torch.device("cpu"),
        resource_snapshot=KVCacheResourceSnapshot(
            platform="win32",
            available_system_memory_bytes=32 * 1024 * 1024 * 1024,
            available_accelerator_memory_bytes=None,
        ),
    )

    assert policy.policy_id == "windows-cpu-balanced"
    assert policy.flush_token_threshold == 128


def test_select_kv_cache_policy_uses_windows_cuda_profile() -> None:
    policy = select_kv_cache_policy(
        torch.device("cuda:0"),
        resource_snapshot=KVCacheResourceSnapshot(
            platform="win32",
            available_system_memory_bytes=32 * 1024 * 1024 * 1024,
            available_accelerator_memory_bytes=16 * 1024 * 1024 * 1024,
        ),
    )

    assert policy.policy_id == "windows-cuda-balanced"
    assert policy.flush_token_threshold == 128

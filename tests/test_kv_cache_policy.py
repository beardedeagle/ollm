import torch

from ollm.kv_cache.policy import KVCacheResourceSnapshot, select_kv_cache_policy


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


def test_select_kv_cache_policy_uses_tiered_profile_for_mps() -> None:
    policy = select_kv_cache_policy(
        torch.device("mps"),
        strategy="tiered-write-back",
        resource_snapshot=KVCacheResourceSnapshot(
            platform="darwin",
            available_system_memory_bytes=32 * 1024 * 1024 * 1024,
            available_accelerator_memory_bytes=None,
        ),
    )

    assert policy.policy_id == "darwin-mps-tiered"
    assert policy.flush_token_threshold == 16
    assert policy.write_back_retained_tokens == 4


def test_select_kv_cache_policy_uses_journal_profile_for_cpu() -> None:
    policy = select_kv_cache_policy(
        torch.device("cpu"),
        strategy="log-structured-journal",
        resource_snapshot=KVCacheResourceSnapshot(
            platform="darwin",
            available_system_memory_bytes=8 * 1024 * 1024 * 1024,
            available_accelerator_memory_bytes=None,
        ),
    )

    assert policy.policy_id == "darwin-cpu-journal"
    assert policy.journal_compaction_entry_threshold == 4


def test_select_kv_cache_policy_uses_larger_journal_threshold_for_roomy_hosts() -> None:
    policy = select_kv_cache_policy(
        torch.device("cuda:0"),
        strategy="log-structured-journal",
        resource_snapshot=KVCacheResourceSnapshot(
            platform="win32",
            available_system_memory_bytes=64 * 1024 * 1024 * 1024,
            available_accelerator_memory_bytes=24 * 1024 * 1024 * 1024,
        ),
    )

    assert policy.policy_id == "windows-cuda-journal"
    assert policy.journal_compaction_entry_threshold == 6


def test_select_kv_cache_policy_uses_quantized_profile_for_cpu() -> None:
    policy = select_kv_cache_policy(
        torch.device("cpu"),
        strategy="quantized-cold-tier",
        resource_snapshot=KVCacheResourceSnapshot(
            platform="darwin",
            available_system_memory_bytes=8 * 1024 * 1024 * 1024,
            available_accelerator_memory_bytes=None,
        ),
    )

    assert policy.policy_id == "darwin-cpu-quantized-cold-tier"
    assert policy.journal_compaction_entry_threshold == 4

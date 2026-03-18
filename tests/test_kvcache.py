from pathlib import Path

from ollm.kvcache import KVCache


def test_kvcache_creates_nested_cache_directory(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache-root"

    cache = KVCache(cache_dir=cache_root, device="cpu", stats=None)

    assert cache.cache_folder == cache_root / "kv_cache"
    assert cache.cache_folder.exists()

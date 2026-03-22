from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from ollm.kv_cache.store_common import atomic_write_bytes, atomic_write_text


def test_atomic_write_text_supports_concurrent_writers(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    barrier = Barrier(8)

    def _write(index: int) -> None:
        barrier.wait()
        atomic_write_text(path, f"content-{index}\n")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_write, range(8)))

    final_content = path.read_text(encoding="utf-8")
    assert final_content in {f"content-{index}\n" for index in range(8)}


def test_atomic_write_bytes_supports_concurrent_writers(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    barrier = Barrier(8)

    def _write(index: int) -> None:
        barrier.wait()
        atomic_write_bytes(path, f"payload-{index}".encode("utf-8"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_write, range(8)))

    final_content = path.read_bytes()
    assert final_content in {f"payload-{index}".encode("utf-8") for index in range(8)}

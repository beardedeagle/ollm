"""Async-backed I/O helpers with synchronous bridge wrappers."""

import asyncio
import io
import shutil
import subprocess
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import TypeVar, cast

import torch

T = TypeVar("T")


def run_async_operation(operation: Coroutine[object, object, T]) -> T:
    """Run an async operation from sync code, even if a loop is already active."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation)

    result: list[T] = []
    errors: list[BaseException] = []
    finished = threading.Event()

    def runner() -> None:
        try:
            result.append(asyncio.run(operation))
        except BaseException as exc:  # pragma: no cover - passthrough
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    finished.wait()
    if errors:
        raise errors[0]
    return result[0]


async def path_read_text_async(path: Path, *, encoding: str = "utf-8") -> str:
    return await asyncio.to_thread(path.read_text, encoding=encoding)


def path_read_text(path: Path, *, encoding: str = "utf-8") -> str:
    return run_async_operation(path_read_text_async(path, encoding=encoding))


async def path_read_bytes_async(path: Path) -> bytes:
    return await asyncio.to_thread(path.read_bytes)


def path_read_bytes(path: Path) -> bytes:
    return run_async_operation(path_read_bytes_async(path))


async def path_write_text_async(
    path: Path, content: str, *, encoding: str = "utf-8"
) -> None:
    await asyncio.to_thread(path.write_text, content, encoding=encoding)


def path_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    run_async_operation(path_write_text_async(path, content, encoding=encoding))


async def path_write_bytes_async(path: Path, content: bytes) -> None:
    await asyncio.to_thread(path.write_bytes, content)


def path_write_bytes(path: Path, content: bytes) -> None:
    run_async_operation(path_write_bytes_async(path, content))


async def path_exists_async(path: Path) -> bool:
    return await asyncio.to_thread(path.exists)


def path_exists(path: Path) -> bool:
    return run_async_operation(path_exists_async(path))


async def path_mkdir_async(
    path: Path,
    *,
    parents: bool = False,
    exist_ok: bool = False,
    mode: int = 0o777,
) -> None:
    await asyncio.to_thread(path.mkdir, parents=parents, exist_ok=exist_ok, mode=mode)


def path_mkdir(
    path: Path,
    *,
    parents: bool = False,
    exist_ok: bool = False,
    mode: int = 0o777,
) -> None:
    run_async_operation(
        path_mkdir_async(path, parents=parents, exist_ok=exist_ok, mode=mode)
    )


async def path_touch_async(path: Path, *, exist_ok: bool = True) -> None:
    await asyncio.to_thread(path.touch, exist_ok=exist_ok)


def path_touch(path: Path, *, exist_ok: bool = True) -> None:
    run_async_operation(path_touch_async(path, exist_ok=exist_ok))


async def path_replace_async(source: Path, target: Path) -> None:
    await asyncio.to_thread(source.replace, target)


def path_replace(source: Path, target: Path) -> None:
    run_async_operation(path_replace_async(source, target))


async def chmod_async(path: Path, mode: int) -> None:
    await asyncio.to_thread(path.chmod, mode)


def chmod_path(path: Path, mode: int) -> None:
    run_async_operation(chmod_async(path, mode))


async def remove_tree_async(path: Path) -> None:
    await asyncio.to_thread(shutil.rmtree, path)


def remove_tree(path: Path) -> None:
    run_async_operation(remove_tree_async(path))


async def torch_load_async(
    path: str | Path, *, map_location: str | torch.device
) -> object:
    return await asyncio.to_thread(torch.load, path, map_location=map_location)


def torch_load_file(path: str | Path, *, map_location: str | torch.device) -> object:
    return run_async_operation(torch_load_async(path, map_location=map_location))


async def torch_save_async(value: object, path: str | Path) -> None:
    await asyncio.to_thread(torch.save, value, path)


def torch_save_file(value: object, path: str | Path) -> None:
    run_async_operation(torch_save_async(value, path))


async def open_binary_file_async(path: str | Path) -> io.BufferedReader:
    return cast(io.BufferedReader, await asyncio.to_thread(io.open, path, "rb"))


def open_binary_file(path: str | Path) -> io.BufferedReader:
    return run_async_operation(open_binary_file_async(path))


async def subprocess_run_async(
    command: tuple[str, ...], *, cwd: str | None = None
) -> subprocess.CompletedProcess[str]:
    return cast(
        subprocess.CompletedProcess[str],
        await asyncio.to_thread(
            subprocess.run,
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        ),
    )


def subprocess_run_process(
    command: tuple[str, ...], *, cwd: str | None = None
) -> subprocess.CompletedProcess[str]:
    return run_async_operation(subprocess_run_async(command, cwd=cwd))


async def subprocess_popen_async(
    command: list[str], *, stdout: int | None, stderr: int | None, text: bool
) -> subprocess.Popen[str]:
    return cast(
        subprocess.Popen[str],
        await asyncio.to_thread(
            subprocess.Popen,
            command,
            stdout=stdout,
            stderr=stderr,
            text=text,
        ),
    )


def subprocess_popen_process(
    command: list[str], *, stdout: int | None, stderr: int | None, text: bool
) -> subprocess.Popen[str]:
    return run_async_operation(
        subprocess_popen_async(command, stdout=stdout, stderr=stderr, text=text)
    )

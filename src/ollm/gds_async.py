"""Pending storage-read helpers for GDS and safetensor loaders."""

from collections.abc import Callable
from concurrent.futures import Future
from typing import Protocol

import torch
from torch.utils.dlpack import from_dlpack

from ollm.async_io import open_binary_file
from ollm.gds_loader_types import _CuFile, _CuFileFuture, _CuPyArray


def read_binary_slice(path: str, *, offset: int, nbytes: int) -> bytes:
    with open_binary_file(path) as handle:
        handle.seek(offset)
        buffer = handle.read(nbytes)
    if len(buffer) != nbytes:
        raise IOError(f"Short read: {len(buffer)} of {nbytes} bytes from {path}")
    return buffer


class PendingTensorRead(Protocol):
    def result(self) -> torch.Tensor: ...


class PendingCpuTensorRead:
    def __init__(
        self,
        *,
        future: Future[bytes],
        dtype: torch.dtype,
        shape: tuple[int, ...],
        started_at: float,
        record_stats: Callable[[str, float], None],
    ) -> None:
        self._future = future
        self._dtype = dtype
        self._shape = shape
        self._started_at = started_at
        self._record_stats = record_stats

    def result(self) -> torch.Tensor:
        buffer = self._future.result()
        self._record_stats("safetensor_read", self._started_at)
        writable_buffer = bytearray(buffer)
        return torch.frombuffer(writable_buffer, dtype=self._dtype).reshape(self._shape)


class PendingGpuTensorRead:
    def __init__(
        self,
        *,
        buffer: _CuPyArray,
        future: _CuFileFuture,
        nbytes: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: str,
        started_at: float,
        record_stats: Callable[[str, float], None],
    ) -> None:
        self._buffer = buffer
        self._future = future
        self._nbytes = nbytes
        self._dtype = dtype
        self._shape = shape
        self._device = device
        self._started_at = started_at
        self._record_stats = record_stats

    def result(self) -> torch.Tensor:
        read_bytes = self._future.get()
        self._record_stats("safetensor_pread", self._started_at)
        if read_bytes != self._nbytes:
            raise IOError(f"Expected {self._nbytes} bytes, got {read_bytes}")
        tensor = torch.as_tensor(self._buffer, device=self._device).view(torch.uint8)
        return tensor.view(self._dtype).reshape(self._shape)


class PendingGDSTensorRead:
    def __init__(
        self,
        *,
        buffer: _CuPyArray,
        future: _CuFileFuture,
        nbytes: int,
        shape: tuple[int, ...],
        started_at: float,
        file_handle: _CuFile,
        record_stats: Callable[[str, float], None],
    ) -> None:
        self._buffer = buffer
        self._future = future
        self._nbytes = nbytes
        self._shape = shape
        self._started_at = started_at
        self._file_handle = file_handle
        self._record_stats = record_stats

    def result(self) -> torch.Tensor:
        try:
            read_bytes = self._future.get()
            self._record_stats("gds_read", self._started_at)
            if read_bytes != self._nbytes:
                raise IOError(f"Short read: {read_bytes} of {self._nbytes} bytes")
            reshaped_buffer = self._buffer.reshape(self._shape)
            return from_dlpack(reshaped_buffer.toDlpack())
        finally:
            self._file_handle.close()

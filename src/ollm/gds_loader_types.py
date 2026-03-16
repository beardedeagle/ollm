"""Type definitions shared by the GDS loader implementation."""

from dataclasses import dataclass
from typing import Protocol, Self, TypedDict

import torch


class _StatsProtocol(Protocol):
    def set(self, name: str, started_at: float) -> None: ...


class _CuPyDTypeInfo(Protocol):
    itemsize: int


class _CuPyArray(Protocol):
    def reshape(self, shape: int | tuple[int, ...] | list[int]) -> Self: ...
    def toDlpack(self) -> object: ...


class _CudaDeviceContext(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None: ...


class _CudaNamespace(Protocol):
    def Device(self, device_id: int) -> _CudaDeviceContext: ...


class _CuPyModule(Protocol):
    float16: object
    float32: object
    float64: object
    int8: object
    int32: object
    uint8: object
    cuda: _CudaNamespace

    def dtype(self, dtype: object) -> _CuPyDTypeInfo: ...
    def empty(self, shape: int | tuple[int, ...], dtype: object) -> _CuPyArray: ...


class _CuFileFuture(Protocol):
    def get(self) -> int: ...


class _CuFile(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None: ...
    def read(self, buffer: _CuPyArray) -> int: ...
    def pread(self, buffer: _CuPyArray, file_offset: int) -> _CuFileFuture: ...
    def close(self) -> None: ...


class _CuFileFactory(Protocol):
    def __call__(self, path: str, mode: str) -> _CuFile: ...


class _KvikioModule(Protocol):
    CuFile: _CuFileFactory


class _GDSManifestEntry(TypedDict):
    path: str
    shape: list[int]
    dtype: str
    packed: str | None


class _SafeTensorHeaderEntry(TypedDict):
    dtype: str
    shape: list[int]
    data_offsets: list[int]


@dataclass(slots=True)
class _OffloadedTensorRecord:
    shape: list[int]
    dtype: str
    packed: str | None
    tensor: torch.Tensor | dict[str, torch.Tensor]

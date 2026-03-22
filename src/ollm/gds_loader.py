import importlib
import json
import os
import re
import struct
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Self, cast

import torch

from ollm.async_io import open_binary_file, torch_load_file
from ollm.gds_async import (
    PendingCpuTensorRead,
    PendingGDSTensorRead,
    PendingGpuTensorRead,
    PendingTensorRead,
    read_binary_slice,
)
from ollm.gds_loader_types import (
    _CuPyModule,
    _GDSManifestEntry,
    _KvikioModule,
    _OffloadedTensorRecord,
    _SafeTensorHeaderEntry,
    _StatsProtocol,
)
from ollm.gds_mxfp4 import convert_moe_packed_tensors


def _import_optional_module(module_name: str) -> object | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


stats: _StatsProtocol | None = None
_cupy_module = cast(_CuPyModule | None, _import_optional_module("cupy"))
_kvikio_module = cast(_KvikioModule | None, _import_optional_module("kvikio"))
kvikio_available = _cupy_module is not None and _kvikio_module is not None
_CPU_READ_EXECUTOR = ThreadPoolExecutor(max_workers=min(8, max(2, os.cpu_count() or 2)))


def _require_cupy() -> _CuPyModule:
    if _cupy_module is None:
        raise RuntimeError("cupy is required for GPU-direct storage operations")
    return _cupy_module


def _require_kvikio() -> _KvikioModule:
    if _kvikio_module is None:
        raise RuntimeError("kvikio is required for GPU-direct storage operations")
    return _kvikio_module


def _record_stats(name: str, started_at: float) -> None:
    if stats is not None:
        stats.set(name, started_at)


def _read_json_object(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return cast(dict[str, object], json.load(handle))


def _match_group(
    primary: re.Match[str] | None, secondary: re.Match[str] | None
) -> str | None:
    if primary is not None:
        return primary.group(1)
    if secondary is not None:
        return secondary.group(1)
    return None


class GDSWeights:
    def __init__(self, path: str, device: str = "cuda:0"):
        self.path = path
        manifest_path = Path(path) / "manifest.json"
        raw_manifest = _read_json_object(manifest_path)
        self.manifest = cast(dict[str, _GDSManifestEntry], raw_manifest)
        self.device = torch.device(device)
        self.offloaded_map: dict[str, _OffloadedTensorRecord] = {}
        self._pending_reads: dict[str, PendingTensorRead] = {}

    def load_param_to_cuda(self, name: str) -> torch.Tensor:
        meta = self.manifest[name]
        offloaded_tensor = self.get_offloaded_from_cpu_to_cuda(name)
        if offloaded_tensor is not None:
            return offloaded_tensor

        pending = self._pending_reads.pop(name, None)
        if pending is not None:
            return pending.result()

        if meta.get("packed") == "mxfp4":
            param_path = os.path.join(self.path, meta["path"])
            return self.load_mxfp4_from_disk(param_path)
        if meta["dtype"].startswith("torch"):
            param_path = os.path.join(self.path, meta["path"])
            return self.load_torch_from_disk(param_path)
        return self._submit_param_to_cuda(name).result()

    def preload_params_to_cuda(self, names: Iterable[str]) -> None:
        for name in names:
            if name in self._pending_reads:
                continue
            if name in self.offloaded_map:
                continue
            meta = self.manifest[name]
            if meta.get("packed") == "mxfp4" or meta["dtype"].startswith("torch"):
                continue
            self._pending_reads[name] = self._submit_param_to_cuda(name)

    def get_dtype(self, dtype: str) -> object:
        cupy = _require_cupy()
        return {
            "float16": cupy.float16,
            "bfloat16": cupy.float16,
            "float32": cupy.float32,
            "float64": cupy.float64,
            "int8": cupy.int8,
            "int32": cupy.int32,
        }[dtype]

    def load_from_disk_to_cuda(
        self, path: str, shape: list[int], dtype: str
    ) -> torch.Tensor:
        return self._submit_read_from_disk_to_cuda(path, shape, dtype).result()

    def _submit_param_to_cuda(self, name: str) -> PendingTensorRead:
        meta = self.manifest[name]
        param_path = os.path.join(self.path, meta["path"])
        return self._submit_read_from_disk_to_cuda(
            param_path,
            meta["shape"],
            meta["dtype"],
        )

    def _submit_read_from_disk_to_cuda(
        self, path: str, shape: list[int], dtype: str
    ) -> PendingTensorRead:
        cupy = _require_cupy()
        kvikio = _require_kvikio()
        cupy_dtype = self.get_dtype(dtype)
        num_elements = 1
        for dimension in shape:
            num_elements *= dimension
        nbytes = num_elements * cupy.dtype(cupy_dtype).itemsize

        with cupy.cuda.Device(0):
            buffer = cupy.empty(num_elements, dtype=cupy_dtype)

        started_at = time.perf_counter()
        file_handle = kvikio.CuFile(path, "r")
        future = file_handle.pread(buffer, file_offset=0)
        return PendingGDSTensorRead(
            buffer=buffer,
            future=future,
            nbytes=nbytes,
            shape=tuple(shape),
            started_at=started_at,
            file_handle=file_handle,
            record_stats=_record_stats,
        )

    def has(self, name: str) -> bool:
        return name in self.manifest

    def load_torch_from_disk(self, path: str) -> torch.Tensor:
        started_at = time.perf_counter()
        tensor = cast(torch.Tensor, torch_load_file(path, map_location=self.device))
        _record_stats("torch_file_load", started_at)
        return tensor

    def load_mxfp4_from_disk(self, path: str) -> torch.Tensor:
        started_at = time.perf_counter()
        packed_tensors = cast(
            dict[str, torch.Tensor],
            torch_load_file(path, map_location=self.device),
        )
        _record_stats("torch_file_load", started_at)
        return convert_moe_packed_tensors(
            packed_tensors["_blocks"], packed_tensors["_scales"]
        ).to(self.device)

    def offload_param_to_cpu(self, name: str) -> None:
        meta = self.manifest[name]
        path = os.path.join(self.path, meta["path"])
        shape = meta["shape"]
        dtype = meta["dtype"]
        packed = meta.get("packed")
        if packed == "mxfp4" or dtype.startswith("torch"):
            tensor = cast(
                torch.Tensor | dict[str, torch.Tensor],
                torch_load_file(path, map_location="cpu"),
            )
        else:
            tensor = self.load_from_disk_to_cuda(path, shape, dtype).cpu()
        self.offloaded_map[name] = _OffloadedTensorRecord(
            shape=shape, dtype=dtype, packed=packed, tensor=tensor
        )

    def get_offloaded_from_cpu_to_cuda(self, name: str) -> torch.Tensor | None:
        record = self.offloaded_map.get(name)
        if record is None:
            return None
        started_at = time.perf_counter()
        if record.packed == "mxfp4":
            packed_tensors = cast(dict[str, torch.Tensor], record.tensor)
            tensor = convert_moe_packed_tensors(
                packed_tensors["_blocks"].to(self.device),
                packed_tensors["_scales"].to(self.device),
            )
        else:
            tensor = cast(torch.Tensor, record.tensor).to(self.device)
        _record_stats("offloaded_cpu_to_cuda", started_at)
        return tensor


class SafeTensorReader:
    def __init__(self, path: str):
        self.path = path
        with open_binary_file(path) as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            self.header = cast(
                dict[str, _SafeTensorHeaderEntry],
                json.loads(handle.read(header_length)),
            )
            self.data_offset = 8 + header_length
        self._pending_reads: dict[str, PendingTensorRead] = {}
        self._dtype_map = {
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I32": torch.int32,
            "I64": torch.int64,
        }

    def close(self) -> None:
        self._pending_reads.clear()

    def keys(self) -> list[str]:
        return list(self.header.keys())

    def preload_tensors(self, names: Iterable[str]) -> None:
        for name in names:
            if name in self._pending_reads:
                continue
            self._pending_reads[name] = self._submit_tensor_read(name)

    def _submit_tensor_read(self, name: str) -> PendingTensorRead:
        info = self.header[name]
        dtype = self._dtype_map[info["dtype"]]
        shape = tuple(info["shape"])
        offset_start, offset_end = info["data_offsets"]
        nbytes = offset_end - offset_start
        started_at = time.perf_counter()
        future = _CPU_READ_EXECUTOR.submit(
            read_binary_slice,
            self.path,
            offset=self.data_offset + offset_start,
            nbytes=nbytes,
        )
        return PendingCpuTensorRead(
            future=future,
            dtype=dtype,
            shape=shape,
            started_at=started_at,
            record_stats=_record_stats,
        )

    def get_tensor(self, name: str) -> torch.Tensor:
        pending = self._pending_reads.pop(name, None)
        if pending is None:
            pending = self._submit_tensor_read(name)
        return pending.result()


class SafeTensorReaderGPU:
    def __init__(self, path: str, device: str = "cuda:0"):
        self._dtype_map = {
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
        }
        self.path = path
        self.device = device
        with open_binary_file(path) as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            self.header = cast(
                dict[str, _SafeTensorHeaderEntry],
                json.loads(handle.read(header_length)),
            )
            self.data_offset = 8 + header_length
        self._file_pointer = _require_kvikio().CuFile(path, "rb")
        self._pending_reads: dict[str, PendingTensorRead] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._pending_reads.clear()
        self._file_pointer.close()

    def keys(self) -> list[str]:
        return list(self.header.keys())

    def preload_tensors(self, names: Iterable[str]) -> None:
        for name in names:
            if name in self._pending_reads:
                continue
            self._pending_reads[name] = self._submit_tensor_read(name)

    def _submit_tensor_read(self, name: str) -> PendingTensorRead:
        if name not in self.header:
            raise KeyError(f"Tensor '{name}' not found in {self.path}")
        info = self.header[name]
        dtype = self._dtype_map[info["dtype"]]
        shape = tuple(info["shape"])
        offset_start, offset_end = info["data_offsets"]
        nbytes = offset_end - offset_start
        cupy = _require_cupy()
        buffer = cupy.empty((nbytes,), dtype=cupy.uint8)
        started_at = time.perf_counter()
        future = self._file_pointer.pread(
            buffer, file_offset=self.data_offset + offset_start
        )
        return PendingGpuTensorRead(
            buffer=buffer,
            future=future,
            nbytes=nbytes,
            dtype=dtype,
            shape=shape,
            device=self.device,
            started_at=started_at,
            record_stats=_record_stats,
        )

    def get_tensor(self, name: str) -> torch.Tensor:
        pending = self._pending_reads.pop(name, None)
        if pending is None:
            pending = self._submit_tensor_read(name)
        return pending.result()


ReaderType = SafeTensorReader | SafeTensorReaderGPU


def get_optimal_safetensor_reader(
    filepath: str, device: torch.device | str | None = None
) -> ReaderType:
    if kvikio_available:
        selected_device = "cuda:0" if device is None else str(device)
        return SafeTensorReaderGPU(filepath, device=selected_device)
    return SafeTensorReader(filepath)


class DenseWeightsLoader:
    def __init__(self, path: str, device: str = "cuda:0"):
        self.path = path
        index_path = Path(path) / "model.safetensors.index.json"
        indexes = _read_json_object(index_path)
        weight_map = cast(dict[str, str], indexes["weight_map"])
        self.manifest: dict[str, dict[str, str]] = {}
        self.safetensors: dict[str, ReaderType] = {}
        for manifest_name, filename in weight_map.items():
            match1 = re.search(r"(language_model.model\.layers\.\d+\.)", manifest_name)
            match2 = re.search(r"(model\.layers\.\d+\.)", manifest_name)
            base = _match_group(match1, match2)
            if base is None:
                continue
            self.manifest.setdefault(base, {})[manifest_name.replace(base, "")] = (
                filename
            )
        self.device = torch.device(device)
        self.offloaded_map: dict[str, dict[str, torch.Tensor]] = {}

    def _preload_base_tensors(self, base: str) -> None:
        filenames_by_tensor: dict[str, list[str]] = {}
        for attr_path, filename in self.manifest[base].items():
            filenames_by_tensor.setdefault(filename, []).append(base + attr_path)
        for filename, tensor_names in filenames_by_tensor.items():
            if filename not in self.safetensors:
                filepath = os.path.join(self.path, filename)
                self.safetensors[filename] = get_optimal_safetensor_reader(
                    filepath, device=self.device
                )
            self.safetensors[filename].preload_tensors(tensor_names)

    def load_dict_to_cuda(self, base: str) -> dict[str, torch.Tensor]:
        offloaded = self.get_offloaded_dict_to_cuda(base)
        if offloaded is not None:
            return offloaded
        return self.load_dict_from_disk(base, device=self.device)

    def prefetch_layer_weights(self, base: str) -> None:
        self._preload_base_tensors(base)

    def offload_dict_to_gpu_cpu(self, base: str, gpu: bool = False) -> None:
        device = self.device if gpu else "cpu"
        self.offloaded_map[base] = self.load_dict_from_disk(base, device=device)

    def get_offloaded_dict_to_cuda(self, base: str) -> dict[str, torch.Tensor] | None:
        if base not in self.offloaded_map:
            return None
        started_at = time.perf_counter()
        tensors = {
            attr_path: tensor.to(self.device, non_blocking=True)
            for attr_path, tensor in self.offloaded_map[base].items()
        }
        _record_stats("offloaded_cpu_to_cuda", started_at)
        return tensors

    def load_dict_from_disk(
        self, base: str, device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        return {
            attr_path: self.safetensors[filename]
            .get_tensor(base + attr_path)
            .to(device)
            for attr_path, filename in self.manifest[base].items()
        }

    def preload_layer_safetensors(self, base: str) -> None:
        self.prefetch_layer_weights(base)


class SingleDenseWeightsLoader(DenseWeightsLoader):
    def __init__(self, path: str, device: str = "cuda:0"):
        self.path = path
        self.device = torch.device(device)
        self.offloaded_map: dict[str, dict[str, torch.Tensor]] = {}
        self.manifest: dict[str, dict[str, str]] = {}
        self.safetensors: dict[str, ReaderType] = {}
        filename = "model.safetensors"
        filepath = os.path.join(self.path, filename)
        self.safetensors[filename] = get_optimal_safetensor_reader(
            filepath, device=self.device
        )
        for manifest_name in self.safetensors[filename].keys():
            match = re.search(r"(model\.layers\.\d+\.)", manifest_name)
            if match is None:
                continue
            base = match.group(1)
            self.manifest.setdefault(base, {})[manifest_name.replace(base, "")] = (
                filename
            )

    def preload_layer_safetensors(self, base: str) -> None:
        self._preload_base_tensors(base)


class MoEWeightsLoader(DenseWeightsLoader):
    def __init__(self, path: str, device: str = "cuda:0"):
        self.path = path
        index_path = Path(path) / "model.safetensors.index.json"
        indexes = _read_json_object(index_path)
        weight_map = cast(dict[str, str], indexes["weight_map"])
        self.manifest: dict[str, dict[str, str]] = {}
        self.safetensors: dict[str, ReaderType] = {}
        for manifest_name, filename in weight_map.items():
            match1 = re.search(
                r"(model\.layers\.\d+\.mlp\.experts\.\d+\.)", manifest_name
            )
            match2 = re.search(r"(model\.layers\.\d+\.)", manifest_name)
            base = _match_group(match1, match2)
            if base is None:
                continue
            self.manifest.setdefault(base, {})[manifest_name.replace(base, "")] = (
                filename
            )
        self.device = torch.device(device)
        self.offloaded_map: dict[str, dict[str, torch.Tensor]] = {}

    def preload_layer_safetensors(self, base: str) -> None:
        for nested_base in list(self.manifest.keys()):
            if not nested_base.startswith(base):
                continue
            self._preload_base_tensors(nested_base)

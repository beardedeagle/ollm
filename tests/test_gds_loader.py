import json
import struct
from pathlib import Path
from typing import Self, cast

import torch

import ollm.gds_async as gds_async
import ollm.gds_loader as gds_loader
from ollm.gds_async import read_binary_slice


def _write_minimal_safetensor(
    path: Path, tensor_name: str, values: torch.Tensor
) -> None:
    tensor_bytes = values.contiguous().numpy().tobytes()
    header = json.dumps(
        {
            tensor_name: {
                "dtype": "F32",
                "shape": list(values.shape),
                "data_offsets": [0, len(tensor_bytes)],
            }
        }
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + tensor_bytes)


class _FakeDType:
    def __init__(self, *, itemsize: int, torch_dtype: torch.dtype) -> None:
        self.itemsize = itemsize
        self.torch_dtype = torch_dtype


class _FakeCudaDeviceContext:
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        del exc_type, exc_val, exc_tb


class _FakeCudaNamespace:
    def Device(self, device_id: int) -> _FakeCudaDeviceContext:
        del device_id
        return _FakeCudaDeviceContext()


class _FakeCuPyArray:
    def __init__(self, element_count: int, dtype: _FakeDType) -> None:
        self._dtype = dtype
        self._element_count = element_count
        self._shape = (element_count,)
        self.raw = bytearray(element_count * dtype.itemsize)

    def reshape(self, shape: int | tuple[int, ...] | list[int]) -> Self:
        if isinstance(shape, int):
            self._shape = (shape,)
        else:
            self._shape = tuple(shape)
        return self

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def torch_dtype(self) -> torch.dtype:
        return self._dtype.torch_dtype

    def toDlpack(self) -> Self:
        return self


class _FakeCuPyModule:
    float16 = _FakeDType(itemsize=2, torch_dtype=torch.float16)
    float32 = _FakeDType(itemsize=4, torch_dtype=torch.float32)
    float64 = _FakeDType(itemsize=8, torch_dtype=torch.float64)
    int8 = _FakeDType(itemsize=1, torch_dtype=torch.int8)
    int32 = _FakeDType(itemsize=4, torch_dtype=torch.int32)
    uint8 = _FakeDType(itemsize=1, torch_dtype=torch.uint8)
    cuda = _FakeCudaNamespace()

    def dtype(self, dtype: object) -> _FakeDType:
        if not isinstance(dtype, _FakeDType):
            raise TypeError("expected fake dtype")
        return dtype

    def empty(self, shape: int | tuple[int, ...], dtype: object) -> _FakeCuPyArray:
        dtype_info = self.dtype(dtype)
        if isinstance(shape, int):
            element_count = shape
        else:
            element_count = 1
            for dimension in shape:
                element_count *= dimension
        return _FakeCuPyArray(element_count, dtype_info)


class _FakeCuFileFuture:
    def __init__(self, path: str, buffer: _FakeCuPyArray, offset: int) -> None:
        self._path = path
        self._buffer = buffer
        self._offset = offset
        self.get_calls = 0

    def get(self) -> int:
        self.get_calls += 1
        data = read_binary_slice(
            self._path,
            offset=self._offset,
            nbytes=len(self._buffer.raw),
        )
        self._buffer.raw[: len(data)] = data
        return len(data)


class _FakeCuFile:
    def __init__(self, path: str, mode: str, futures: list[_FakeCuFileFuture]) -> None:
        self._path = path
        self._mode = mode
        self._futures = futures

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        del exc_type, exc_val, exc_tb

    def read(self, buffer: _FakeCuPyArray) -> int:
        future = _FakeCuFileFuture(self._path, buffer, 0)
        self._futures.append(future)
        return future.get()

    def pread(self, buffer: _FakeCuPyArray, file_offset: int) -> _FakeCuFileFuture:
        future = _FakeCuFileFuture(self._path, buffer, file_offset)
        self._futures.append(future)
        return future

    def close(self) -> None:
        del self._mode


class _FakeKvikioModule:
    def __init__(self) -> None:
        self.futures: list[_FakeCuFileFuture] = []

    def CuFile(self, path: str, mode: str) -> _FakeCuFile:
        return _FakeCuFile(path, mode, self.futures)


def _fake_torch_as_tensor(buffer: _FakeCuPyArray, *, device: str) -> torch.Tensor:
    del device
    return torch.frombuffer(buffer.raw, dtype=torch.uint8)


def _fake_from_dlpack(buffer: _FakeCuPyArray) -> torch.Tensor:
    tensor = torch.frombuffer(buffer.raw, dtype=buffer.torch_dtype).clone()
    return tensor.reshape(buffer.shape)


def test_safe_tensor_reader_preload_submits_cpu_reads(tmp_path: Path) -> None:
    tensor_path = tmp_path / "model.safetensors"
    expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
    _write_minimal_safetensor(tensor_path, "weight", expected)

    reader = gds_loader.SafeTensorReader(str(tensor_path))

    reader.preload_tensors(["weight"])

    assert "weight" in reader._pending_reads
    result = reader.get_tensor("weight")

    assert "weight" not in reader._pending_reads
    assert torch.equal(result, expected)


def test_safe_tensor_reader_gpu_preload_defers_future_wait(
    monkeypatch, tmp_path: Path
) -> None:
    tensor_path = tmp_path / "model.safetensors"
    expected = torch.tensor([3.0, 4.0], dtype=torch.float32)
    _write_minimal_safetensor(tensor_path, "weight", expected)
    fake_kvikio = _FakeKvikioModule()

    monkeypatch.setattr(gds_loader, "_cupy_module", _FakeCuPyModule())
    monkeypatch.setattr(gds_loader, "_kvikio_module", fake_kvikio)
    monkeypatch.setattr(gds_loader.torch, "as_tensor", _fake_torch_as_tensor)

    reader = gds_loader.SafeTensorReaderGPU(str(tensor_path), device="cuda:0")
    reader.preload_tensors(["weight"])

    future = fake_kvikio.futures[-1]
    assert future.get_calls == 0

    result = reader.get_tensor("weight")

    assert future.get_calls == 1
    assert torch.equal(result, expected)


def test_single_dense_weights_loader_preload_submits_layer_tensors(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "single-dense"
    model_dir.mkdir()
    _write_minimal_safetensor(
        model_dir / "model.safetensors",
        "model.layers.0.mlp.gate_proj.weight",
        torch.tensor([5.0, 6.0], dtype=torch.float32),
    )

    loader = gds_loader.SingleDenseWeightsLoader(str(model_dir), device="cpu")
    reader = cast(gds_loader.SafeTensorReader, loader.safetensors["model.safetensors"])

    loader.preload_layer_safetensors("model.layers.0.")

    assert "model.layers.0.mlp.gate_proj.weight" in reader._pending_reads


def test_single_dense_weights_loader_load_dict_to_cuda_without_prefetch(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "single-dense-sync"
    model_dir.mkdir()
    expected = torch.tensor([9.0, 10.0], dtype=torch.float32)
    _write_minimal_safetensor(
        model_dir / "model.safetensors",
        "model.layers.0.mlp.gate_proj.weight",
        expected,
    )

    loader = gds_loader.SingleDenseWeightsLoader(str(model_dir), device="cpu")

    loaded = loader.load_dict_to_cuda("model.layers.0.")

    assert torch.equal(loaded["mlp.gate_proj.weight"], expected)


def test_dense_weights_loader_prefetch_layer_weights_stages_sharded_tensors(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "sharded-dense"
    model_dir.mkdir()
    expected = torch.tensor([11.0, 12.0], dtype=torch.float32)
    filename = "model-00001-of-00001.safetensors"
    tensor_name = "model.layers.0.mlp.gate_proj.weight"
    _write_minimal_safetensor(model_dir / filename, tensor_name, expected)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": int(expected.numel() * expected.element_size())
                },
                "weight_map": {tensor_name: filename},
            }
        ),
        encoding="utf-8",
    )

    loader = gds_loader.DenseWeightsLoader(str(model_dir), device="cpu")
    loader.prefetch_layer_weights("model.layers.0.")
    reader = cast(gds_loader.SafeTensorReader, loader.safetensors[filename])

    assert tensor_name in reader._pending_reads

    loaded = loader.load_dict_to_cuda("model.layers.0.")

    assert torch.equal(loaded["mlp.gate_proj.weight"], expected)


def test_dense_weights_loader_prefetch_layer_weights_skips_unsupported_device(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "sharded-dense-mps"
    model_dir.mkdir()
    filename = "model-00001-of-00001.safetensors"
    tensor_name = "model.layers.0.mlp.gate_proj.weight"
    _write_minimal_safetensor(
        model_dir / filename,
        tensor_name,
        torch.tensor([13.0, 14.0], dtype=torch.float32),
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {tensor_name: filename},
            }
        ),
        encoding="utf-8",
    )

    loader = gds_loader.DenseWeightsLoader(str(model_dir), device="mps")
    loader.prefetch_layer_weights("model.layers.0.")

    assert loader.safetensors == {}


def test_gds_weights_preload_defers_future_wait(monkeypatch, tmp_path: Path) -> None:
    export_dir = tmp_path / "gds_export"
    export_dir.mkdir()
    expected = torch.tensor([7.0, 8.0], dtype=torch.float32)
    data_path = export_dir / "weight.bin"
    data_path.write_bytes(expected.contiguous().numpy().tobytes())
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "weight": {
                    "path": "weight.bin",
                    "shape": [2],
                    "dtype": "float32",
                    "packed": None,
                }
            }
        ),
        encoding="utf-8",
    )
    fake_kvikio = _FakeKvikioModule()

    monkeypatch.setattr(gds_loader, "_cupy_module", _FakeCuPyModule())
    monkeypatch.setattr(gds_loader, "_kvikio_module", fake_kvikio)
    monkeypatch.setattr(gds_async, "from_dlpack", _fake_from_dlpack)

    weights = gds_loader.GDSWeights(str(export_dir), device="cuda:0")
    weights.preload_params_to_cuda(["weight"])

    future = fake_kvikio.futures[-1]
    assert future.get_calls == 0

    result = weights.load_param_to_cuda("weight")

    assert future.get_calls == 1
    assert torch.equal(result, expected)

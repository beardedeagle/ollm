from pathlib import Path

import torch
from scripts.gpt_oss_export import (
    build_tensor_spec_index,
    load_optional_tensor,
)
from scripts.llama_export import _write_raw_tensor_file
from scripts.scripts import benchmark_tensor_storage


def test_write_raw_tensor_file_supports_bfloat16(tmp_path: Path) -> None:
    output_path = tmp_path / "tensor.bin"
    tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).contiguous()

    _write_raw_tensor_file(tensor, output_path)

    assert output_path.read_bytes() == tensor.view(torch.uint8).numpy().tobytes()


def test_benchmark_tensor_storage_rejects_large_defaults_by_default(
    tmp_path: Path,
) -> None:
    try:
        benchmark_tensor_storage(
            output_dir=tmp_path,
            expert_count=512,
            hidden_size=1024,
            expert_ids=(1, 3, 9),
            allow_large_allocation=False,
        )
    except ValueError as exc:
        assert "Benchmark allocation is too large" in str(exc)
    else:
        raise AssertionError("expected ValueError for oversized benchmark inputs")


def test_build_tensor_spec_index_keeps_tensor_loading_lazy(
    monkeypatch, tmp_path: Path
) -> None:
    shard_path = tmp_path / "model-00000-of-00002.safetensors"
    shard_path.write_bytes(b"placeholder")
    calls: list[tuple[str, str]] = []

    class _FakeSafeOpen:
        def __init__(self, path: Path, *, framework: str) -> None:
            del framework
            self._path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def keys(self) -> list[str]:
            calls.append(("keys", self._path.name))
            return ["weight_blocks", "weight_scales"]

        def get_tensor(self, key: str) -> torch.Tensor:
            calls.append(("get_tensor", key))
            return torch.tensor([1], dtype=torch.int32)

    monkeypatch.setattr(
        "scripts.gpt_oss_export.safe_open",
        lambda path, framework="pt": _FakeSafeOpen(path, framework=framework),
    )

    index = build_tensor_spec_index(tmp_path, ("model-00000-of-00002",))

    assert index == {
        "weight_blocks": shard_path,
        "weight_scales": shard_path,
    }
    assert calls == [("keys", shard_path.name)]

    tensor = load_optional_tensor("weight_blocks", index)

    assert isinstance(tensor, torch.Tensor)
    assert calls == [("keys", shard_path.name), ("get_tensor", "weight_blocks")]

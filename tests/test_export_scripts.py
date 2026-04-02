from pathlib import Path

import torch
from scripts.llama_export import _write_raw_tensor_file


def test_write_raw_tensor_file_supports_bfloat16(tmp_path: Path) -> None:
    output_path = tmp_path / "tensor.bin"
    tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).contiguous()

    _write_raw_tensor_file(tensor, output_path)

    assert output_path.read_bytes() == tensor.view(torch.uint8).numpy().tobytes()

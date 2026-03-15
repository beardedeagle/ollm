import json
from pathlib import Path

import pytest

from ollm.runtime.safety import validate_safe_gds_export_artifacts


def test_validate_safe_gds_export_artifacts_rejects_torch_serialized_entries(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "gds_export"
    export_dir.mkdir()
    (export_dir / "tensor.pt").write_text("unsafe", encoding="utf-8")
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "weight": {
                    "path": "tensor.pt",
                    "shape": [1],
                    "dtype": "torch.float16",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refuses"):
        validate_safe_gds_export_artifacts(export_dir)


def test_validate_safe_gds_export_artifacts_rejects_path_traversal(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "gds_export"
    export_dir.mkdir()
    outside_file = tmp_path / "tensor.bin"
    outside_file.write_text("unsafe", encoding="utf-8")
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "weight": {
                    "path": "../tensor.bin",
                    "shape": [1],
                    "dtype": "float16",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes"):
        validate_safe_gds_export_artifacts(export_dir)

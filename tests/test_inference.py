import json
import subprocess
import sys
from pathlib import Path

import pytest

from ollm.inference import AutoInference


def test_auto_inference_rejects_missing_local_model_dir(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-model"
    with pytest.raises(ValueError, match="Local model directory does not exist"):
        AutoInference(str(missing_path), device="cpu", logging=False)


class StubAutoInference(AutoInference):
    def load_model(self, model_dir: str) -> None:
        del model_dir
        self.model = object()


def test_auto_inference_rejects_unsafe_adapter_artifacts(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.bin").write_text("unsafe", encoding="utf-8")

    with pytest.raises(ValueError, match="safetensors"):
        StubAutoInference(str(model_dir), adapter_dir=str(adapter_dir), device="cpu", logging=False)


def test_importing_ollm_does_not_require_peft() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockPeft(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'peft' or fullname.startswith('peft.'):
            raise ModuleNotFoundError('blocked peft import')
        return None

sys.meta_path.insert(0, BlockPeft())
sys.path.insert(0, r'REPO_SRC')
import ollm
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script.replace("REPO_SRC", str(repo_root / "src"))],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"

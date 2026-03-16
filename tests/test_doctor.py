import json
from pathlib import Path

import torch

from ollm.app.doctor import DoctorService
from ollm.runtime.config import RuntimeConfig


def test_doctor_service_reports_download_readiness(tmp_path: Path) -> None:
    service = DoctorService()
    config = RuntimeConfig(
        model_reference="llama3-1B-chat", models_dir=tmp_path / "models"
    )
    report = service.run(
        config,
        include_imports=False,
        include_runtime=False,
        include_paths=False,
        include_download=True,
    )
    checks = {check.name: check for check in report.checks}
    assert checks["download:ready"].ok is True
    assert (tmp_path / "models").exists()


def test_doctor_service_reports_missing_model_path(tmp_path: Path) -> None:
    service = DoctorService()
    config = RuntimeConfig(
        model_reference="llama3-1B-chat", models_dir=tmp_path / "models"
    )
    report = service.run(
        config,
        include_imports=False,
        include_runtime=False,
        include_paths=True,
        include_download=False,
    )
    checks = {check.name: check for check in report.checks}
    assert checks["path:models-dir"].ok is False
    assert checks["model:resolution"].ok is True
    assert checks["model:path"].ok is False


def test_doctor_service_imports_only_excludes_model_checks(tmp_path: Path) -> None:
    service = DoctorService()
    config = RuntimeConfig(
        model_reference="llama3-1B-chat", models_dir=tmp_path / "models"
    )
    report = service.run(
        config,
        include_imports=True,
        include_runtime=False,
        include_paths=False,
        include_download=False,
    )
    names = {check.name for check in report.checks}
    assert "model:path" not in names


def test_doctor_service_marks_opaque_references_as_unsupported(
    tmp_path: Path,
) -> None:
    service = DoctorService()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    config = RuntimeConfig(model_reference="qwen3.5:9b-bf16", models_dir=models_dir)
    report = service.run(
        config,
        include_imports=False,
        include_runtime=False,
        include_paths=True,
        include_download=False,
    )

    checks = {check.name: check for check in report.checks}
    assert checks["model:resolution"].ok is False
    assert "is not a built-in alias, local directory, or Hugging Face repository" in (
        checks["model:resolution"].message
    )
    assert checks["model:path"].ok is False


def test_doctor_service_reports_generic_local_model_as_executable(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models" / "qwen2"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
        encoding="utf-8",
    )
    service = DoctorService()
    config = RuntimeConfig(
        model_reference=str(model_dir), models_dir=tmp_path / "models", device="cpu"
    )
    report = service.run(
        config,
        include_imports=False,
        include_runtime=False,
        include_paths=True,
        include_download=False,
    )
    checks = {check.name: check for check in report.checks}
    assert checks["model:resolution"].ok is True


def test_doctor_service_reports_planned_specialization_passes_for_optimized_local_model(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models" / "llama"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    service = DoctorService()
    config = RuntimeConfig(
        model_reference=str(model_dir), models_dir=tmp_path / "models", device="cpu"
    )
    report = service.run(
        config,
        include_imports=False,
        include_runtime=False,
        include_paths=True,
        include_download=False,
    )
    checks = {check.name: check for check in report.checks}

    assert checks["model:resolution"].ok is True
    assert checks["model:resolution"].message.startswith(
        "Selected specialization provider 'llama-native'"
    )
    assert (
        checks["model:resolution"].details["specialization_provider_id"]
        == "llama-native"
    )
    assert checks["model:resolution"].details["support_level"] == "optimized"
    assert checks["model:resolution"].details["specialization_state"] == "planned"
    assert (
        checks["model:resolution"].details["planned_specialization_pass_ids"]
        == "disk-cache,cpu-offload,mlp-chunking"
    )


def test_doctor_service_fails_when_requested_cuda_device_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    service = DoctorService()
    config = RuntimeConfig(
        model_reference="llama3-1B-chat",
        models_dir=tmp_path / "models",
        device="cuda:0",
    )
    report = service.run(
        config,
        include_imports=False,
        include_runtime=True,
        include_paths=False,
        include_download=False,
    )
    checks = {check.name: check for check in report.checks}

    assert checks["runtime:requested-device"].ok is False
    assert report.ok() is False

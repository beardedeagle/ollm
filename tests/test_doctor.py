import json
from pathlib import Path

from ollm.app.doctor import DoctorService
from ollm.runtime.config import RuntimeConfig



def test_doctor_service_reports_download_readiness(tmp_path: Path) -> None:
    service = DoctorService()
    config = RuntimeConfig(model_reference="llama3-1B-chat", models_dir=tmp_path / "models")
    report = service.run(config, include_imports=False, include_runtime=False, include_paths=False, include_download=True)
    checks = {check.name: check for check in report.checks}
    assert checks["download:ready"].ok is True
    assert (tmp_path / "models").exists()



def test_doctor_service_reports_missing_model_path(tmp_path: Path) -> None:
    service = DoctorService()
    config = RuntimeConfig(model_reference="llama3-1B-chat", models_dir=tmp_path / "models")
    report = service.run(config, include_imports=False, include_runtime=False, include_paths=True, include_download=False)
    checks = {check.name: check for check in report.checks}
    assert checks["path:models-dir"].ok is False
    assert checks["model:resolution"].ok is True
    assert checks["model:path"].ok is False



def test_doctor_service_imports_only_excludes_model_checks(tmp_path: Path) -> None:
    service = DoctorService()
    config = RuntimeConfig(model_reference="llama3-1B-chat", models_dir=tmp_path / "models")
    report = service.run(config, include_imports=True, include_runtime=False, include_paths=False, include_download=False)
    names = {check.name for check in report.checks}
    assert "model:path" not in names


def test_doctor_service_marks_provider_backed_references_as_not_executable(tmp_path: Path) -> None:
    service = DoctorService()
    config = RuntimeConfig(model_reference="ollama:qwen3.5:9b-bf16", models_dir=tmp_path / "models")
    report = service.run(config, include_imports=False, include_runtime=False, include_paths=True, include_download=False)
    checks = {check.name: check for check in report.checks}
    assert checks["model:resolution"].ok is False


def test_doctor_service_reports_generic_local_model_as_executable(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "qwen2"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
        encoding="utf-8",
    )
    service = DoctorService()
    config = RuntimeConfig(model_reference=str(model_dir), models_dir=tmp_path / "models", device="cpu")
    report = service.run(config, include_imports=False, include_runtime=False, include_paths=True, include_download=False)
    checks = {check.name: check for check in report.checks}
    assert checks["model:resolution"].ok is True

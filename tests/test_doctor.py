from pathlib import Path

from ollm.app.doctor import DoctorService
from ollm.runtime.config import RuntimeConfig


def test_doctor_service_reports_download_readiness(tmp_path: Path) -> None:
    service = DoctorService()
    config = RuntimeConfig(model_id="llama3-1B-chat", models_dir=tmp_path / "models")
    report = service.run(config, include_imports=False, include_runtime=False, include_paths=False, include_download=True)
    checks = {check.name: check for check in report.checks}
    assert checks["download:ready"].ok is True
    assert (tmp_path / "models").exists()


def test_doctor_service_reports_missing_model_path(tmp_path: Path) -> None:
    service = DoctorService()
    config = RuntimeConfig(model_id="llama3-1B-chat", models_dir=tmp_path / "models")
    report = service.run(config, include_imports=False, include_runtime=False, include_paths=True, include_download=False)
    checks = {check.name: check for check in report.checks}
    assert checks["path:models-dir"].ok is False
    assert checks["model:path"].ok is False


def test_doctor_service_imports_only_excludes_model_checks(tmp_path: Path) -> None:
    service = DoctorService()
    config = RuntimeConfig(model_id="llama3-1B-chat", models_dir=tmp_path / "models")
    report = service.run(config, include_imports=True, include_runtime=False, include_paths=False, include_download=False)
    names = {check.name for check in report.checks}
    assert "model:path" not in names

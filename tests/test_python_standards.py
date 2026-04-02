import sys
from pathlib import Path

from ollm.async_io import subprocess_run_process

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_python_standards.py"
)


def test_standards_checker_passes_clean_tree(tmp_path: Path) -> None:
    (tmp_path / "src" / "sample").mkdir(parents=True)
    (tmp_path / "src" / "sample" / "module.py").write_text(
        '"""Sample module."""\n\nimport math\n\n\ndef area(radius: float) -> float:\n    """Return the area of a circle."""\n    return math.pi * radius * radius\n',
        encoding="utf-8",
    )

    completed = subprocess_run_process(
        (sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path))
    )

    assert completed.returncode == 0
    assert "no violations" in completed.stdout.lower()


def test_standards_checker_reports_mechanical_failures(tmp_path: Path) -> None:
    package_root = tmp_path / "src" / "sample"
    package_root.mkdir(parents=True)
    oversized_file = "\n".join("value = 1" for _ in range(1001))
    (package_root / "module.py").write_text(
        (
            "from __future__ import annotations\n"
            "from typing import Any, Optional\n"
            "from .other import helper\n"
            "value = 1\n"
            "import math\n"
            "def area(radius: 'Radius', payload: Any | None = None) -> Optional[int]:\n"
            "    return 1\n\n"
            f"{oversized_file}\n"
        ),
        encoding="utf-8",
    )

    completed = subprocess_run_process(
        (sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path))
    )

    assert completed.returncode == 1
    assert "future-annotations" in completed.stdout
    assert "relative-import" in completed.stdout
    assert "typing-alias" in completed.stdout
    assert "forward-reference" in completed.stdout
    assert "late-import" in completed.stdout
    assert "file-too-large-hard" in completed.stdout


def test_standards_checker_can_emit_markdown_report(tmp_path: Path) -> None:
    (tmp_path / "src" / "sample").mkdir(parents=True)
    (tmp_path / "src" / "sample" / "module.py").write_text(
        "from typing import Optional\n\n\ndef sample(value: Optional[int]) -> int | None:\n    return value\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "audit.md"

    completed = subprocess_run_process(
        (
            sys.executable,
            str(SCRIPT_PATH),
            "--root",
            str(tmp_path),
            "--format",
            "markdown",
            "--output",
            str(output_path),
        )
    )

    assert completed.returncode == 1
    rendered = output_path.read_text(encoding="utf-8")
    assert "# Python Standards Audit" in rendered
    assert "Mechanical failures" in rendered
    assert "typing-alias" in rendered


def test_standards_checker_reports_audit_only_findings(tmp_path: Path) -> None:
    package_root = tmp_path / "src" / "sample"
    package_root.mkdir(parents=True)
    oversized_file = "\n".join("value = 1" for _ in range(501))
    (package_root / "module.py").write_text(
        (
            "import subprocess\n\n"
            "def run_probe() -> None:\n"
            "    subprocess.run(['echo', 'hi'], check=False)\n\n"
            f"{oversized_file}\n"
        ),
        encoding="utf-8",
    )

    completed = subprocess_run_process(
        (sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path))
    )

    assert completed.returncode == 0
    assert "file-too-large-soft" in completed.stdout
    assert "sync-io" in completed.stdout


def test_standards_checker_reports_partial_work_markers(tmp_path: Path) -> None:
    package_root = tmp_path / "src" / "sample"
    package_root.mkdir(parents=True)
    (package_root / "module.py").write_text(
        ("def sample() -> int:\n    # TODO remove placeholder path\n    return 1\n"),
        encoding="utf-8",
    )

    completed = subprocess_run_process(
        (sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path))
    )

    assert completed.returncode == 1
    assert "partial-work-marker" in completed.stdout


def test_standards_checker_reports_machine_specific_paths(tmp_path: Path) -> None:
    script_root = tmp_path / "scripts"
    script_root.mkdir(parents=True)
    (script_root / "manual.py").write_text(
        (
            "MODEL_DIR = '/media/example/models/demo'\n"
            "def main() -> None:\n"
            "    print(MODEL_DIR)\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
        encoding="utf-8",
    )

    completed = subprocess_run_process(
        (sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path))
    )

    assert completed.returncode == 1
    assert "machine-specific-path" in completed.stdout


def test_standards_checker_reports_script_import_side_effects(tmp_path: Path) -> None:
    script_root = tmp_path / "scripts"
    script_root.mkdir(parents=True)
    (script_root / "manual.py").write_text(
        ("counter = 0\nfor value in range(3):\n    counter += value\n"),
        encoding="utf-8",
    )

    completed = subprocess_run_process(
        (sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path))
    )

    assert completed.returncode == 1
    assert "script-import-side-effect" in completed.stdout


def test_standards_checker_reports_callable_top_level_assignments(
    tmp_path: Path,
) -> None:
    script_root = tmp_path / "scripts"
    script_root.mkdir(parents=True)
    (script_root / "manual.py").write_text(
        ("def build_value() -> int:\n    return 1\n\nVALUE = build_value()\n"),
        encoding="utf-8",
    )

    completed = subprocess_run_process(
        (sys.executable, str(SCRIPT_PATH), "--root", str(tmp_path))
    )

    assert completed.returncode == 1
    assert "script-import-side-effect" in completed.stdout

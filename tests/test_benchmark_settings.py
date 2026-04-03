from pathlib import Path

from ollm.runtime.settings import BenchmarkSettings, load_app_settings


def _write_benchmark_settings_file(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[benchmark]",
                'history_dir = "file-benchmark-history"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_app_settings_reads_benchmark_history_dir_from_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "ollm.toml"
    _write_benchmark_settings_file(config_path)

    settings = load_app_settings(config_file=config_path)

    assert settings.benchmark.history_dir == Path("file-benchmark-history")


def test_load_app_settings_reads_benchmark_history_dir_from_environment(
    monkeypatch, tmp_path: Path
) -> None:
    history_dir = tmp_path / "env-benchmark-history"
    monkeypatch.setenv("OLLM_BENCHMARK__HISTORY_DIR", str(history_dir))

    settings = load_app_settings()

    assert settings.benchmark.history_dir == history_dir


def test_load_app_settings_environment_overrides_benchmark_history_file(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "ollm.toml"
    history_dir = tmp_path / "env-benchmark-history"
    _write_benchmark_settings_file(config_path)
    monkeypatch.setenv("OLLM_BENCHMARK__HISTORY_DIR", str(history_dir))

    settings = load_app_settings(config_file=config_path)

    assert settings.benchmark.history_dir == history_dir


def test_benchmark_settings_accept_history_dir(tmp_path: Path) -> None:
    history_dir = tmp_path / "benchmark-history"
    settings = BenchmarkSettings(history_dir=history_dir)

    assert settings.history_dir == history_dir

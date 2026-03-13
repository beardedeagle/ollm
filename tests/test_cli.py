import stat
import json
from pathlib import Path

from typer.testing import CliRunner

from ollm.cli.main import create_app
from ollm.cli.services import CommandServices

from tests.fakes import FakeDoctorService, FakeRuntimeExecutor, FakeRuntimeLoader


def build_test_app():
    loader = FakeRuntimeLoader()
    services = CommandServices(
        runtime_loader=loader,
        runtime_executor=FakeRuntimeExecutor(),
        doctor_service=FakeDoctorService(),
    )
    return CliRunner(), loader, create_app(services)


def test_prompt_command_supports_text_and_json_output(tmp_path: Path) -> None:
    runner, _, app = build_test_app()
    result = runner.invoke(app, ["prompt", "hello world", "--no-stream", "--no-color"])
    assert result.exit_code == 0
    assert "echo:hello world" in result.output

    output_path = tmp_path / "prompt.json"
    json_result = runner.invoke(
        app,
        ["prompt", "hello json", "--format", "json", "--no-stream", "--no-color", "--output", str(output_path)],
    )
    assert json_result.exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["text"] == "echo:hello json"

    text_output = tmp_path / "prompt.txt"
    text_result = runner.invoke(
        app,
        ["prompt", "hello file", "--no-stream", "--no-color", "--output", str(text_output)],
    )
    assert text_result.exit_code == 0
    assert text_output.read_text(encoding="utf-8") == "echo:hello file"
    assert "echo:hello file" not in text_result.output
    assert stat.S_IMODE(text_output.stat().st_mode) == 0o600


def test_prompt_command_supports_stdin() -> None:
    runner, _, app = build_test_app()
    result = runner.invoke(app, ["prompt", "--stdin", "--no-stream", "--no-color"], input="from stdin")
    assert result.exit_code == 0
    assert "echo:from stdin" in result.output


def test_prompt_command_validates_multimodal_and_json_stream() -> None:
    runner, _, app = build_test_app()

    multimodal_error = runner.invoke(app, ["prompt", "hello", "--image", "image.png", "--no-stream", "--no-color"])
    assert multimodal_error.exit_code != 0
    assert "--image and --audio require --multimodal" in multimodal_error.output

    multimodal_ok = runner.invoke(
        app,
        ["prompt", "hello", "--image", "image.png", "--multimodal", "--no-stream", "--no-color"],
    )
    assert multimodal_ok.exit_code == 0
    assert "echo:hello" in multimodal_ok.output

    json_stream_error = runner.invoke(app, ["prompt", "hello", "--format", "json", "--no-color"])
    assert json_stream_error.exit_code != 0
    assert "--format json cannot be combined with --stream" in json_stream_error.output


def test_root_command_requires_interactive_tty() -> None:
    runner, _, app = build_test_app()
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "Use `ollm prompt`" in result.output


def test_doctor_and_models_commands(tmp_path: Path) -> None:
    runner, loader, app = build_test_app()
    model_dir = tmp_path / "models"
    (model_dir / "llama3-1B-chat").mkdir(parents=True)

    doctor_result = runner.invoke(app, ["doctor", "--json", "--no-color"])
    assert doctor_result.exit_code == 0
    assert "doctor:fake" in doctor_result.output

    models_result = runner.invoke(app, ["models", "list", "--json", "--models-dir", str(model_dir), "--no-color"])
    assert models_result.exit_code == 0
    assert "llama3-1B-chat" in models_result.output

    download_result = runner.invoke(app, ["models", "download", "llama3-3B-chat", "--models-dir", str(model_dir), "--no-color"])
    assert download_result.exit_code == 0
    assert loader.download_calls[0][0] == "llama3-3B-chat"

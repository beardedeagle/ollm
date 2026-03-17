import json
import stat
from pathlib import Path

from ollm.cli import chat as chat_module
from tests.cli_support import build_real_app, build_test_app, strip_ansi


def test_prompt_command_supports_text_and_json_output(tmp_path: Path) -> None:
    runner, _, app = build_test_app()
    result = runner.invoke(app, ["prompt", "hello world", "--no-stream", "--no-color"])
    assert result.exit_code == 0
    assert "echo:hello world" in result.output

    output_path = tmp_path / "prompt.json"
    json_result = runner.invoke(
        app,
        [
            "prompt",
            "hello json",
            "--format",
            "json",
            "--no-stream",
            "--no-color",
            "--output",
            str(output_path),
        ],
    )
    assert json_result.exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["text"] == "echo:hello json"

    text_output = tmp_path / "prompt.txt"
    text_result = runner.invoke(
        app,
        [
            "prompt",
            "hello file",
            "--no-stream",
            "--no-color",
            "--output",
            str(text_output),
        ],
    )
    assert text_result.exit_code == 0
    assert text_output.read_text(encoding="utf-8") == "echo:hello file"
    assert "echo:hello file" not in text_result.output
    assert stat.S_IMODE(text_output.stat().st_mode) == 0o600


def test_prompt_command_supports_stdin() -> None:
    runner, _, app = build_test_app()
    result = runner.invoke(
        app, ["prompt", "--stdin", "--no-stream", "--no-color"], input="from stdin"
    )
    assert result.exit_code == 0
    assert "echo:from stdin" in result.output


def test_prompt_command_validates_multimodal_and_json_stream() -> None:
    runner, _, app = build_test_app()

    multimodal_error = runner.invoke(
        app, ["prompt", "hello", "--image", "image.png", "--no-stream", "--no-color"]
    )
    assert multimodal_error.exit_code != 0
    assert "--image and --audio require --multimodal" in strip_ansi(
        multimodal_error.output
    )

    multimodal_ok = runner.invoke(
        app,
        [
            "prompt",
            "hello",
            "--image",
            "image.png",
            "--multimodal",
            "--no-stream",
            "--no-color",
        ],
    )
    assert multimodal_ok.exit_code == 0
    assert "echo:hello" in multimodal_ok.output

    json_stream_error = runner.invoke(
        app, ["prompt", "hello", "--format", "json", "--no-color"]
    )
    assert json_stream_error.exit_code != 0
    assert "--format json cannot be combined with --stream" in strip_ansi(
        json_stream_error.output
    )


def test_prompt_command_accepts_non_catalog_model_reference() -> None:
    runner, loader, app = build_test_app()
    result = runner.invoke(
        app,
        [
            "prompt",
            "hello world",
            "--model",
            "Qwen/Qwen2.5-7B-Instruct",
            "--no-stream",
            "--no-color",
        ],
    )
    assert result.exit_code == 0
    assert loader.load_calls == ["Qwen/Qwen2.5-7B-Instruct"]


def test_prompt_command_honors_backend_override_and_no_specialization() -> None:
    runner, loader, app = build_test_app()
    result = runner.invoke(
        app,
        [
            "prompt",
            "hello world",
            "--backend",
            "transformers-generic",
            "--no-specialization",
            "--no-stream",
            "--no-color",
        ],
    )
    assert result.exit_code == 0
    assert loader.loaded_configs[-1].backend == "transformers-generic"
    assert loader.loaded_configs[-1].use_specialization is False


def test_prompt_command_rejects_optimized_backend_with_no_specialization() -> None:
    runner, _, app = build_test_app()
    result = runner.invoke(
        app,
        [
            "prompt",
            "hello world",
            "--backend",
            "optimized-native",
            "--no-specialization",
            "--no-stream",
            "--no-color",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "cannot be combined with --no-specialization" in str(result.exception)


def test_prompt_command_plan_json_prints_without_loading_runtime() -> None:
    runner, loader, app = build_test_app()
    result = runner.invoke(
        app,
        [
            "prompt",
            "--model",
            "llama3-1B-chat",
            "--backend",
            "transformers-generic",
            "--no-specialization",
            "--plan-json",
            "--no-color",
        ],
    )
    assert result.exit_code == 0
    assert loader.load_calls == []
    assert loader.plan_calls[-1].backend == "transformers-generic"
    assert '"runtime_plan"' in result.output
    assert '"specialization_enabled": false' in result.output


def test_chat_command_plan_json_does_not_require_tty() -> None:
    runner, loader, app = build_test_app()
    result = runner.invoke(
        app,
        [
            "chat",
            "--backend",
            "transformers-generic",
            "--no-specialization",
            "--plan-json",
            "--no-color",
        ],
    )
    assert result.exit_code == 0
    assert loader.load_calls == []
    assert loader.plan_calls[-1].backend == "transformers-generic"
    assert '"runtime_plan"' in result.output


def test_run_chat_command_executes_session_through_application_service(
    monkeypatch, tmp_path: Path
) -> None:
    runner, loader, app = build_test_app()
    events: list[str] = []

    class FakeShell:
        def __init__(self, session, console, history_file, plain):
            del console, history_file, plain
            self._session = session

        def run(self) -> None:
            response = self._session.prompt_text("List planets")
            events.append(response.text)

    monkeypatch.setattr(chat_module, "ensure_interactive_terminal", lambda: None)
    monkeypatch.setattr(chat_module, "InteractiveChatShell", FakeShell)

    result = runner.invoke(
        app,
        [
            "chat",
            "--models-dir",
            str(tmp_path / "models"),
            "--no-stream",
            "--no-color",
        ],
    )

    assert result.exit_code == 0
    assert events == ["echo:List planets"]
    assert loader.load_calls == ["llama3-1B-chat"]


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

    models_result = runner.invoke(
        app, ["models", "list", "--json", "--models-dir", str(model_dir), "--no-color"]
    )
    assert models_result.exit_code == 0
    assert "llama3-1B-chat" in models_result.output

    info_result = runner.invoke(
        app,
        [
            "models",
            "info",
            "Qwen/Qwen2.5-7B-Instruct",
            "--json",
            "--models-dir",
            str(model_dir),
            "--no-color",
        ],
    )
    assert info_result.exit_code == 0
    assert '"source_kind": "hugging-face"' in info_result.output

    installed_info_result = runner.invoke(
        app,
        [
            "models",
            "info",
            "llama3-1B-chat",
            "--json",
            "--models-dir",
            str(model_dir),
            "--no-color",
        ],
    )
    assert installed_info_result.exit_code == 0
    assert '"support_level": "optimized"' in installed_info_result.output
    assert '"resolved_support_level": "optimized"' in installed_info_result.output
    assert '"materialized": true' in installed_info_result.output
    assert '"resolved_supports_disk_cache": true' in installed_info_result.output
    assert '"specialization_state": "planned"' in installed_info_result.output
    assert '"planned_specialization_pass_ids": [' in installed_info_result.output

    download_result = runner.invoke(
        app,
        [
            "models",
            "download",
            "llama3-3B-chat",
            "--models-dir",
            str(model_dir),
            "--no-color",
        ],
    )
    assert download_result.exit_code == 0
    assert loader.download_calls[0][0] == "llama3-3B-chat"


def test_doctor_and_models_info_support_plan_json_and_backend_override(
    tmp_path: Path,
) -> None:
    runner, loader, app = build_test_app()

    doctor_result = runner.invoke(
        app,
        [
            "doctor",
            "--model",
            "llama3-1B-chat",
            "--backend",
            "transformers-generic",
            "--no-specialization",
            "--plan-json",
            "--no-color",
        ],
    )
    assert doctor_result.exit_code == 0
    assert loader.plan_calls[-1].backend == "transformers-generic"
    assert '"backend": "transformers-generic"' in doctor_result.output
    assert '"specialization_enabled": false' in doctor_result.output

    info_result = runner.invoke(
        app,
        [
            "models",
            "info",
            "llama3-1B-chat",
            "--backend",
            "transformers-generic",
            "--no-specialization",
            "--plan-json",
            "--models-dir",
            str(tmp_path / "models"),
            "--no-color",
        ],
    )
    assert info_result.exit_code == 0
    assert loader.plan_calls[-1].backend == "transformers-generic"
    assert '"runtime_config"' in info_result.output
    assert '"backend_id": "transformers-generic"' in info_result.output


def test_models_list_applies_backend_override_to_runtime_plans(tmp_path: Path) -> None:
    runner, _, app = build_test_app()
    model_dir = tmp_path / "models"
    (model_dir / "llama3-1B-chat").mkdir(parents=True)

    result = runner.invoke(
        app,
        [
            "models",
            "list",
            "--backend",
            "transformers-generic",
            "--no-specialization",
            "--json",
            "--models-dir",
            str(model_dir),
            "--no-color",
        ],
    )
    assert result.exit_code == 0
    assert '"backend_id": "transformers-generic"' in result.output


def test_models_path_returns_resolved_local_path(tmp_path: Path) -> None:
    runner, _, app = build_test_app()
    model_dir = tmp_path / "models"
    materialized_path = model_dir / "llama3-1B-chat"
    materialized_path.mkdir(parents=True)

    result = runner.invoke(
        app,
        [
            "models",
            "path",
            "llama3-1B-chat",
            "--models-dir",
            str(model_dir),
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == str(materialized_path)


def test_opaque_models_info_and_doctor_commands_are_unsupported(
    tmp_path: Path,
) -> None:
    runner, app = build_real_app()
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    info_result = runner.invoke(
        app,
        [
            "models",
            "info",
            "qwen3.5:9b-bf16",
            "--json",
            "--models-dir",
            str(model_dir),
            "--no-color",
        ],
    )
    assert info_result.exit_code == 0
    assert '"source_kind": "opaque"' in info_result.output
    assert '"support_level": "unsupported"' in info_result.output
    assert "could not be resolved" in info_result.output

    doctor_result = runner.invoke(
        app,
        [
            "doctor",
            "--json",
            "--model",
            "qwen3.5:9b-bf16",
            "--models-dir",
            str(model_dir),
            "--no-color",
        ],
    )
    assert doctor_result.exit_code == 1
    assert '"model:resolution"' in doctor_result.output
    assert "is not a built-in alias, local directory, or Hugging Face repository" in (
        doctor_result.output
    )


def test_prompt_command_rejects_opaque_model_reference(
    tmp_path: Path,
) -> None:
    runner, app = build_real_app()
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "prompt",
            "hello world",
            "--model",
            "qwen3.5:9b-bf16",
            "--models-dir",
            str(model_dir),
            "--no-stream",
            "--no-color",
        ],
    )
    assert result.exit_code == 1
    assert "could not be resolved" in strip_ansi(result.output)

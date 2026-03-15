import json
import re
import stat
from pathlib import Path
from typing import cast

from typer.testing import CliRunner

from ollm.app.doctor import DoctorService
from ollm.cli.services import CommandServices
from ollm.cli.main import create_app
from ollm.runtime.backends.openai_compatible import OpenAICompatibleBackend
from ollm.runtime.backends.ollama import OllamaBackend
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader
from ollm.runtime.providers.openai_compatible_client import OpenAICompatibleClient
from ollm.runtime.providers.ollama_client import OllamaClient

from tests.openai_compatible_server import OpenAICompatibleFixtureServer
from tests.fakes import FakeDoctorService, FakeRuntimeExecutor, FakeRuntimeLoader
from tests.media_server import MediaFixtureServer, MediaResponse
from tests.ollama_server import OllamaFixtureServer


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so assertions work regardless of terminal."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def build_test_app():
    loader = FakeRuntimeLoader()
    services = CommandServices(
        runtime_loader=cast(RuntimeLoader, loader),
        runtime_executor=cast(RuntimeExecutor, FakeRuntimeExecutor()),
        doctor_service=cast(DoctorService, FakeDoctorService()),
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
    assert "--image and --audio require --multimodal" in _strip_ansi(
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
    assert "--format json cannot be combined with --stream" in json_stream_error.output


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
    assert '"availability_status": "materialized"' in installed_info_result.output
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


def test_provider_backed_models_info_and_doctor_commands(tmp_path: Path) -> None:
    server = OllamaFixtureServer(
        models={"llama3.2": {"capabilities": ["completion"], "response_text": "ready"}}
    )
    server.start()
    try:
        runtime_loader = RuntimeLoader(
            backends=(OllamaBackend(client=OllamaClient(base_url=server.base_url)),),
        )
        services = CommandServices(
            runtime_loader=runtime_loader,
            runtime_executor=RuntimeExecutor(),
            doctor_service=DoctorService(runtime_loader=runtime_loader),
        )
        runner = CliRunner()
        app = create_app(services)

        info_result = runner.invoke(
            app,
            [
                "models",
                "info",
                "ollama:llama3.2",
                "--json",
                "--models-dir",
                str(tmp_path / "models"),
                "--no-color",
            ],
        )
        assert info_result.exit_code == 0
        assert '"backend_id": "ollama"' in info_result.output
        assert '"materialized": false' in info_result.output
        assert '"available": true' in info_result.output
        assert '"availability_status": "available"' in info_result.output
        assert '"support_level": "provider-backed"' in info_result.output

        doctor_result = runner.invoke(
            app,
            [
                "doctor",
                "--json",
                "--model",
                "ollama:llama3.2",
                "--models-dir",
                str(tmp_path / "models"),
                "--no-color",
            ],
        )
        assert doctor_result.exit_code == 0
        assert '"runtime:requested-device"' in doctor_result.output
        assert (
            '"Provider-backed model references for ollama ignore local device'
            in doctor_result.output
        )
    finally:
        server.stop()


def test_msty_provider_models_info_and_doctor_commands(tmp_path: Path) -> None:
    server = OllamaFixtureServer(
        models={"llama3.2": {"capabilities": ["completion"], "response_text": "ready"}}
    )
    server.start()
    try:
        runtime_loader = RuntimeLoader(
            backends=(
                OllamaBackend(
                    client_factory=lambda endpoint: OllamaClient(base_url=endpoint)
                ),
            ),
        )
        services = CommandServices(
            runtime_loader=runtime_loader,
            runtime_executor=RuntimeExecutor(),
            doctor_service=DoctorService(runtime_loader=runtime_loader),
        )
        runner = CliRunner()
        app = create_app(services)

        info_result = runner.invoke(
            app,
            [
                "models",
                "info",
                "msty:llama3.2",
                "--provider-endpoint",
                server.base_url,
                "--json",
                "--models-dir",
                str(tmp_path / "models"),
                "--no-color",
            ],
        )
        assert info_result.exit_code == 0
        assert '"backend_id": "ollama"' in info_result.output
        assert '"materialized": false' in info_result.output
        assert '"available": true' in info_result.output
        assert '"availability_status": "available"' in info_result.output
        assert '"provider_name": "msty"' in info_result.output

        doctor_result = runner.invoke(
            app,
            [
                "doctor",
                "--json",
                "--model",
                "msty:llama3.2",
                "--provider-endpoint",
                server.base_url,
                "--models-dir",
                str(tmp_path / "models"),
                "--no-color",
            ],
        )
        assert doctor_result.exit_code == 0
        assert '"runtime:requested-device"' in doctor_result.output
        assert (
            '"Provider-backed model references for msty ignore local device'
            in doctor_result.output
        )
    finally:
        server.stop()


def test_prompt_command_executes_ollama_provider_reference_with_remote_image_url(
    tmp_path: Path,
) -> None:
    ollama_server = OllamaFixtureServer(
        models={
            "llava": {
                "capabilities": ["completion", "vision"],
                "response_text": "remote vision ready",
            }
        }
    )
    media_server = MediaFixtureServer(
        responses={
            "/diagram.png": MediaResponse(
                body=b"remote-png-bytes", content_type="image/png"
            ),
        }
    )
    ollama_server.start()
    media_server.start()
    try:
        runtime_loader = RuntimeLoader(
            backends=(
                OllamaBackend(client=OllamaClient(base_url=ollama_server.base_url)),
            ),
        )
        services = CommandServices(
            runtime_loader=runtime_loader,
            runtime_executor=RuntimeExecutor(),
            doctor_service=DoctorService(runtime_loader=runtime_loader),
        )
        runner = CliRunner()
        app = create_app(services)

        result = runner.invoke(
            app,
            [
                "prompt",
                "describe the image",
                "--model",
                "ollama:llava",
                "--image",
                f"{media_server.base_url}/diagram.png",
                "--multimodal",
                "--no-stream",
                "--no-color",
            ],
        )
        assert result.exit_code == 0
        assert "remote vision ready" in result.output
    finally:
        media_server.stop()
        ollama_server.stop()


def test_models_list_discovers_provider_models(tmp_path: Path) -> None:
    ollama_server = OllamaFixtureServer(
        models={"llama3.2": {"capabilities": ["completion"], "response_text": "ready"}}
    )
    lmstudio_server = OpenAICompatibleFixtureServer(
        models={"local-model": {"response_text": "ready"}}
    )
    ollama_server.start()
    lmstudio_server.start()
    try:
        runtime_loader = RuntimeLoader(
            backends=(
                OllamaBackend(client=OllamaClient(base_url=ollama_server.base_url)),
                OpenAICompatibleBackend(
                    client_factory=lambda endpoint: OpenAICompatibleClient(
                        base_url=endpoint
                    )
                ),
            ),
        )
        services = CommandServices(
            runtime_loader=runtime_loader,
            runtime_executor=RuntimeExecutor(),
            doctor_service=DoctorService(runtime_loader=runtime_loader),
        )
        runner = CliRunner()
        app = create_app(services)

        result = runner.invoke(
            app,
            [
                "models",
                "list",
                "--json",
                "--discover-provider",
                "ollama",
                "--discover-provider",
                "lmstudio",
                "--provider-endpoint",
                lmstudio_server.base_url,
                "--models-dir",
                str(tmp_path / "models"),
                "--no-color",
            ],
        )
        assert result.exit_code == 0
        assert '"model_reference": "ollama:llama3.2"' in result.output
        assert '"model_reference": "lmstudio:local-model"' in result.output
        assert '"discovery_source": "discovered-provider"' in result.output
        assert '"availability_status": "available"' in result.output
    finally:
        lmstudio_server.stop()
        ollama_server.stop()


def test_models_list_discovers_msty_provider_models(tmp_path: Path) -> None:
    server = OllamaFixtureServer(
        models={"llama3.2": {"capabilities": ["completion"], "response_text": "ready"}}
    )
    server.start()
    try:
        runtime_loader = RuntimeLoader(
            backends=(
                OllamaBackend(
                    client_factory=lambda endpoint: OllamaClient(base_url=endpoint)
                ),
            ),
        )
        services = CommandServices(
            runtime_loader=runtime_loader,
            runtime_executor=RuntimeExecutor(),
            doctor_service=DoctorService(runtime_loader=runtime_loader),
        )
        runner = CliRunner()
        app = create_app(services)

        result = runner.invoke(
            app,
            [
                "models",
                "list",
                "--json",
                "--discover-provider",
                "msty",
                "--provider-endpoint",
                server.base_url,
                "--models-dir",
                str(tmp_path / "models"),
                "--no-color",
            ],
        )
        assert result.exit_code == 0
        assert '"model_reference": "msty:llama3.2"' in result.output
        assert '"provider_endpoint": "' + server.base_url + '"' in result.output
        assert '"discovery_source": "discovered-provider"' in result.output
        assert '"availability_status": "available"' in result.output
    finally:
        server.stop()


def test_models_list_installed_filters_out_provider_entries(tmp_path: Path) -> None:
    server = OllamaFixtureServer(
        models={"llama3.2": {"capabilities": ["completion"], "response_text": "ready"}}
    )
    server.start()
    try:
        runtime_loader = RuntimeLoader(
            backends=(OllamaBackend(client=OllamaClient(base_url=server.base_url)),),
        )
        services = CommandServices(
            runtime_loader=runtime_loader,
            runtime_executor=RuntimeExecutor(),
            doctor_service=DoctorService(runtime_loader=runtime_loader),
        )
        runner = CliRunner()
        app = create_app(services)
        model_dir = tmp_path / "models"
        (model_dir / "llama3-1B-chat").mkdir(parents=True)

        result = runner.invoke(
            app,
            [
                "models",
                "list",
                "--installed",
                "--discover-provider",
                "ollama",
                "--json",
                "--models-dir",
                str(model_dir),
                "--no-color",
            ],
        )
        assert result.exit_code == 0
        assert '"model_reference": "llama3-1B-chat"' in result.output
        assert '"model_reference": "ollama:llama3.2"' not in result.output
    finally:
        server.stop()


def test_models_list_requires_endpoint_for_openai_compatible_discovery(
    tmp_path: Path,
) -> None:
    runner, _, app = build_test_app()
    result = runner.invoke(
        app,
        [
            "models",
            "list",
            "--discover-provider",
            "openai-compatible",
            "--models-dir",
            str(tmp_path / "models"),
            "--no-color",
        ],
    )
    assert result.exit_code != 0
    assert "--provider-endpoint" in _strip_ansi(result.output)


def test_models_list_requires_endpoint_for_msty_discovery(tmp_path: Path) -> None:
    runner, _, app = build_test_app()
    result = runner.invoke(
        app,
        [
            "models",
            "list",
            "--discover-provider",
            "msty",
            "--models-dir",
            str(tmp_path / "models"),
            "--no-color",
        ],
    )
    assert result.exit_code != 0
    assert "--provider-endpoint" in _strip_ansi(result.output)


def test_models_list_rejects_invalid_provider_endpoint(tmp_path: Path) -> None:
    runner, _, app = build_test_app()
    result = runner.invoke(
        app,
        [
            "models",
            "list",
            "--discover-provider",
            "openai-compatible",
            "--provider-endpoint",
            "not-a-url",
            "--models-dir",
            str(tmp_path / "models"),
            "--no-color",
        ],
    )
    assert result.exit_code != 0
    assert "absolute http or https URL" in result.output


def test_models_list_rejects_provider_endpoint_with_credentials(tmp_path: Path) -> None:
    runner, _, app = build_test_app()
    result = runner.invoke(
        app,
        [
            "models",
            "list",
            "--discover-provider",
            "openai-compatible",
            "--provider-endpoint",
            "https://user:pass@example.test/v1",
            "--models-dir",
            str(tmp_path / "models"),
            "--no-color",
        ],
    )
    assert result.exit_code != 0
    assert "must not include credentials" in result.output


def test_prompt_command_rejects_provider_endpoint_with_credentials() -> None:
    runner, _, app = build_test_app()
    result = runner.invoke(
        app,
        [
            "prompt",
            "hello world",
            "--model",
            "openai-compatible:local-model",
            "--provider-endpoint",
            "https://user:pass@example.test/v1",
            "--no-stream",
            "--no-color",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "must not include credentials" in str(result.exception)


def test_openai_compatible_provider_models_info_and_doctor_commands(
    tmp_path: Path,
) -> None:
    server = OpenAICompatibleFixtureServer(
        models={"local-model": {"response_text": "ready"}}
    )
    server.start()
    try:
        runtime_loader = RuntimeLoader(
            backends=(OpenAICompatibleBackend(),),
        )
        services = CommandServices(
            runtime_loader=runtime_loader,
            runtime_executor=RuntimeExecutor(),
            doctor_service=DoctorService(runtime_loader=runtime_loader),
        )
        runner = CliRunner()
        app = create_app(services)

        info_result = runner.invoke(
            app,
            [
                "models",
                "info",
                "openai-compatible:local-model",
                "--provider-endpoint",
                server.base_url,
                "--json",
                "--models-dir",
                str(tmp_path / "models"),
                "--no-color",
            ],
        )
        assert info_result.exit_code == 0
        assert '"backend_id": "openai-compatible"' in info_result.output
        assert '"materialized": false' in info_result.output
        assert '"available": true' in info_result.output
        assert '"availability_status": "available"' in info_result.output
        assert '"support_level": "provider-backed"' in info_result.output
        assert '"modalities": [' in info_result.output
        assert '"audio_input_support": "request-capable"' in info_result.output

        doctor_result = runner.invoke(
            app,
            [
                "doctor",
                "--json",
                "--model",
                "openai-compatible:local-model",
                "--provider-endpoint",
                server.base_url,
                "--models-dir",
                str(tmp_path / "models"),
                "--no-color",
            ],
        )
        assert doctor_result.exit_code == 0
        assert '"runtime:requested-device"' in doctor_result.output
        assert (
            '"Provider-backed model references for openai-compatible ignore local device'
            in doctor_result.output
        )
        assert '"modalities": "text"' in doctor_result.output
        assert '"audio_input_support": "request-capable"' in doctor_result.output
    finally:
        server.stop()


def test_prompt_command_executes_openai_compatible_provider_reference(
    tmp_path: Path,
) -> None:
    server = OpenAICompatibleFixtureServer(
        models={"local-model": {"response_text": "ready from provider"}}
    )
    server.start()
    try:
        runtime_loader = RuntimeLoader(
            backends=(OpenAICompatibleBackend(),),
        )
        services = CommandServices(
            runtime_loader=runtime_loader,
            runtime_executor=RuntimeExecutor(),
            doctor_service=DoctorService(runtime_loader=runtime_loader),
        )
        runner = CliRunner()
        app = create_app(services)

        result = runner.invoke(
            app,
            [
                "prompt",
                "say hi",
                "--model",
                "openai-compatible:local-model",
                "--provider-endpoint",
                server.base_url,
                "--no-stream",
                "--no-color",
            ],
        )
        assert result.exit_code == 0
        assert "ready from provider" in result.output
    finally:
        server.stop()


def test_prompt_command_executes_openai_compatible_provider_audio_reference(
    tmp_path: Path,
) -> None:
    provider_server = OpenAICompatibleFixtureServer(
        models={"audio-model": {"response_text": "heard audio from cli"}}
    )
    media_server = MediaFixtureServer(
        responses={
            "/sample.wav": MediaResponse(body=b"wav-bytes", content_type="audio/wav"),
        }
    )
    provider_server.start()
    media_server.start()
    try:
        runtime_loader = RuntimeLoader(
            backends=(OpenAICompatibleBackend(),),
        )
        services = CommandServices(
            runtime_loader=runtime_loader,
            runtime_executor=RuntimeExecutor(),
            doctor_service=DoctorService(runtime_loader=runtime_loader),
        )
        runner = CliRunner()
        app = create_app(services)

        result = runner.invoke(
            app,
            [
                "prompt",
                "describe the clip",
                "--model",
                "openai-compatible:audio-model",
                "--provider-endpoint",
                provider_server.base_url,
                "--multimodal",
                "--audio",
                f"{media_server.base_url}/sample.wav",
                "--no-stream",
                "--no-color",
            ],
        )
        assert result.exit_code == 0
        assert "heard audio from cli" in result.output
    finally:
        media_server.stop()
        provider_server.stop()


def test_prompt_command_executes_msty_provider_reference(tmp_path: Path) -> None:
    server = OllamaFixtureServer(
        models={
            "llama3.2": {
                "capabilities": ["completion"],
                "response_text": "ready from msty",
            }
        }
    )
    server.start()
    try:
        runtime_loader = RuntimeLoader(
            backends=(
                OllamaBackend(
                    client_factory=lambda endpoint: OllamaClient(base_url=endpoint)
                ),
            ),
        )
        services = CommandServices(
            runtime_loader=runtime_loader,
            runtime_executor=RuntimeExecutor(),
            doctor_service=DoctorService(runtime_loader=runtime_loader),
        )
        runner = CliRunner()
        app = create_app(services)

        result = runner.invoke(
            app,
            [
                "prompt",
                "say hi",
                "--model",
                "msty:llama3.2",
                "--provider-endpoint",
                server.base_url,
                "--no-stream",
                "--no-color",
            ],
        )
        assert result.exit_code == 0
        assert "ready from msty" in result.output
    finally:
        server.stop()

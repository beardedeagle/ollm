from typing import cast

from ollm.runtime.settings import DEFAULT_SERVER_HOST, ServerSettings
from ollm.server.runtime import SERVER_EXTRA_INSTALL_HINT, ServerDependenciesError
from tests.cli_support import build_test_app


def test_serve_command_is_registered() -> None:
    runner, _, app = build_test_app()

    result = runner.invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output


def test_serve_command_uses_loaded_server_settings_when_flags_are_omitted(
    monkeypatch,
) -> None:
    runner, _, app = build_test_app()
    captured: dict[str, object] = {}

    def fake_serve_application(server_settings, application_service) -> None:
        captured["settings"] = server_settings
        captured["application_service"] = application_service

    monkeypatch.setattr(
        "ollm.cli.server.serve_application",
        fake_serve_application,
    )

    result = runner.invoke(
        app,
        ["serve"],
        env={
            "OLLM_SERVER__PORT": "8123",
            "OLLM_SERVER__LOG_LEVEL": "debug",
        },
    )

    assert result.exit_code == 0
    server_settings = cast(ServerSettings, captured["settings"])
    assert server_settings.host == DEFAULT_SERVER_HOST
    assert server_settings.port == 8123
    assert server_settings.log_level == "debug"
    assert captured["application_service"] is not None


def test_serve_command_cli_flags_override_loaded_server_settings(
    monkeypatch,
) -> None:
    runner, _, app = build_test_app()
    captured: dict[str, object] = {}

    def fake_serve_application(server_settings, application_service) -> None:
        del application_service
        captured["settings"] = server_settings

    monkeypatch.setattr(
        "ollm.cli.server.serve_application",
        fake_serve_application,
    )

    result = runner.invoke(
        app,
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--reload",
            "--log-level",
            "warning",
        ],
        env={
            "OLLM_SERVER__HOST": "127.0.0.1",
            "OLLM_SERVER__PORT": "8123",
            "OLLM_SERVER__LOG_LEVEL": "debug",
            "OLLM_SERVER__RELOAD": "false",
        },
    )

    assert result.exit_code == 0
    server_settings = cast(ServerSettings, captured["settings"])
    assert server_settings.host == "0.0.0.0"
    assert server_settings.port == 9001
    assert server_settings.reload is True
    assert server_settings.log_level == "warning"


def test_serve_command_reports_missing_server_dependencies_cleanly(
    monkeypatch,
) -> None:
    runner, _, app = build_test_app()

    def missing_server_dependencies(server_settings, application_service) -> None:
        del server_settings, application_service
        raise ServerDependenciesError(SERVER_EXTRA_INSTALL_HINT)

    monkeypatch.setattr(
        "ollm.cli.server.serve_application",
        missing_server_dependencies,
    )

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 1
    assert SERVER_EXTRA_INSTALL_HINT in result.output

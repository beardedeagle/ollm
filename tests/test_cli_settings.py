from pathlib import Path

from tests.cli_support import build_test_app


def test_prompt_command_uses_env_defaults_when_flags_are_omitted(
    tmp_path: Path,
) -> None:
    runner, loader, app = build_test_app()
    env_models_dir = tmp_path / "env-models"
    result = runner.invoke(
        app,
        ["prompt", "hello world", "--print-config", "--no-color"],
        env={
            "OLLM_RUNTIME__MODEL_REFERENCE": "env-model",
            "OLLM_RUNTIME__MODELS_DIR": str(env_models_dir),
            "OLLM_RUNTIME__DEVICE": "cpu",
            "OLLM_GENERATION__MAX_NEW_TOKENS": "17",
            "OLLM_GENERATION__STREAM": "false",
        },
    )

    assert result.exit_code == 0
    assert loader.loaded_configs[-1].model_reference == "env-model"
    assert loader.loaded_configs[-1].models_dir == env_models_dir
    assert loader.loaded_configs[-1].device == "cpu"
    assert '"max_new_tokens": 17' in result.output
    assert '"stream": false' in result.output


def test_prompt_command_cli_flags_override_env_defaults(
    tmp_path: Path,
) -> None:
    runner, loader, app = build_test_app()
    env_models_dir = tmp_path / "env-models"
    cli_models_dir = tmp_path / "cli-models"
    result = runner.invoke(
        app,
        [
            "prompt",
            "hello world",
            "--model",
            "cli-model",
            "--models-dir",
            str(cli_models_dir),
            "--device",
            "mps",
            "--max-new-tokens",
            "9",
            "--stream",
            "--print-config",
            "--no-color",
        ],
        env={
            "OLLM_RUNTIME__MODEL_REFERENCE": "env-model",
            "OLLM_RUNTIME__MODELS_DIR": str(env_models_dir),
            "OLLM_RUNTIME__DEVICE": "cpu",
            "OLLM_GENERATION__MAX_NEW_TOKENS": "17",
            "OLLM_GENERATION__STREAM": "false",
        },
    )

    assert result.exit_code == 0
    assert loader.loaded_configs[-1].model_reference == "cli-model"
    assert loader.loaded_configs[-1].models_dir == cli_models_dir
    assert loader.loaded_configs[-1].device == "mps"
    assert '"max_new_tokens": 9' in result.output
    assert '"stream": true' in result.output


def test_models_info_uses_loaded_runtime_defaults_for_backend_and_models_dir(
    tmp_path: Path,
) -> None:
    runner, loader, app = build_test_app()
    model_dir = tmp_path / "models-from-env"
    model_dir.mkdir()

    result = runner.invoke(
        app,
        ["models", "info", "llama3-1B-chat", "--plan-json", "--no-color"],
        env={
            "OLLM_RUNTIME__MODELS_DIR": str(model_dir),
            "OLLM_RUNTIME__BACKEND": "transformers-generic",
        },
    )

    assert result.exit_code == 0
    assert loader.plan_calls[-1].models_dir == model_dir
    assert loader.plan_calls[-1].backend == "transformers-generic"


def test_doctor_command_uses_loaded_runtime_defaults_when_flags_are_omitted(
    tmp_path: Path,
) -> None:
    runner, loader, app = build_test_app()
    model_dir = tmp_path / "doctor-models"
    model_dir.mkdir()

    result = runner.invoke(
        app,
        ["doctor", "--plan-json", "--no-color"],
        env={
            "OLLM_RUNTIME__MODEL_REFERENCE": "env-model",
            "OLLM_RUNTIME__MODELS_DIR": str(model_dir),
            "OLLM_RUNTIME__BACKEND": "transformers-generic",
            "OLLM_RUNTIME__DEVICE": "cpu",
        },
    )

    assert result.exit_code == 0
    assert loader.plan_calls[-1].model_reference == "env-model"
    assert loader.plan_calls[-1].models_dir == model_dir
    assert loader.plan_calls[-1].backend == "transformers-generic"
    assert loader.plan_calls[-1].device == "cpu"

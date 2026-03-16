from pathlib import Path
from typing import cast

from ollm import GenerationConfig, RuntimeClient, RuntimeConfig
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader
from tests.fakes import FakeRuntimeExecutor, FakeRuntimeLoader


def build_client() -> tuple[RuntimeClient, FakeRuntimeLoader, FakeRuntimeExecutor]:
    loader = FakeRuntimeLoader()
    executor = FakeRuntimeExecutor()
    return (
        RuntimeClient(
            runtime_loader=cast(RuntimeLoader, loader),
            runtime_executor=cast(RuntimeExecutor, executor),
        ),
        loader,
        executor,
    )


def test_runtime_client_describe_plan_reports_backend_override_and_specialization_choice(
    tmp_path: Path,
) -> None:
    client, loader, _ = build_client()
    runtime_config = RuntimeConfig(
        model_reference="llama3-1B-chat",
        models_dir=tmp_path / "models",
        backend="transformers-generic",
        use_specialization=False,
    )

    payload = client.describe_plan(runtime_config)

    assert loader.plan_calls[-1].backend == "transformers-generic"
    assert payload["runtime_config"]["backend"] == "transformers-generic"
    assert payload["runtime_config"]["use_specialization"] is False
    assert payload["runtime_plan"]["backend_id"] == "transformers-generic"
    assert payload["runtime_plan"]["specialization_enabled"] is False


def test_runtime_client_prompt_loads_runtime_and_executes_request(
    tmp_path: Path,
) -> None:
    client, loader, executor = build_client()
    runtime_config = RuntimeConfig(
        model_reference="Qwen/Qwen2.5-7B-Instruct",
        models_dir=tmp_path / "models",
    )
    generation_config = GenerationConfig(stream=False)

    response = client.prompt(
        "hello from client",
        runtime_config=runtime_config,
        generation_config=generation_config,
    )

    assert response.text == "echo:hello from client"
    assert loader.load_calls == ["Qwen/Qwen2.5-7B-Instruct"]
    assert executor.prompts == ["hello from client"]


def test_runtime_client_prompt_enables_multimodal_when_parts_require_it(
    tmp_path: Path,
) -> None:
    client, loader, _ = build_client()
    runtime_config = RuntimeConfig(
        model_reference="llama3-1B-chat",
        models_dir=tmp_path / "models",
        multimodal=False,
    )

    client.prompt(
        "describe the clip",
        runtime_config=runtime_config,
        generation_config=GenerationConfig(stream=False),
        audio=("https://example.test/sample.wav",),
    )

    assert loader.loaded_configs[-1].multimodal is True
    assert runtime_config.multimodal is False


def test_runtime_client_session_reuses_runtime_stack(tmp_path: Path) -> None:
    client, loader, executor = build_client()
    session = client.session(
        runtime_config=RuntimeConfig(
            model_reference="llama3-1B-chat",
            models_dir=tmp_path / "models",
        ),
        generation_config=GenerationConfig(stream=False),
        system_prompt="You are concise.",
    )

    response = session.prompt_text("List planets")

    assert response.text == "echo:List planets"
    assert loader.load_calls == ["llama3-1B-chat"]
    assert executor.prompts == ["List planets"]


def test_runtime_client_prompt_validates_generation_config(tmp_path: Path) -> None:
    client, loader, _ = build_client()

    try:
        client.prompt(
            "hello",
            runtime_config=RuntimeConfig(
                model_reference="llama3-1B-chat",
                models_dir=tmp_path / "models",
            ),
            generation_config=GenerationConfig(max_new_tokens=0),
        )
    except ValueError as exc:
        assert "--max-new-tokens must be greater than zero" in str(exc)
    else:
        raise AssertionError("RuntimeClient.prompt should validate GenerationConfig")

    assert loader.load_calls == []


def test_runtime_client_session_validates_generation_config(tmp_path: Path) -> None:
    client, _, _ = build_client()

    try:
        client.session(
            runtime_config=RuntimeConfig(
                model_reference="llama3-1B-chat",
                models_dir=tmp_path / "models",
            ),
            generation_config=GenerationConfig(max_new_tokens=0),
        )
    except ValueError as exc:
        assert "--max-new-tokens must be greater than zero" in str(exc)
    else:
        raise AssertionError("RuntimeClient.session should validate GenerationConfig")

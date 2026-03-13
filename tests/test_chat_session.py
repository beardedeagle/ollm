from pathlib import Path

from ollm.app.history import load_transcript
from ollm.app.session import ChatSession
from ollm.runtime.config import GenerationConfig, RuntimeConfig

from tests.fakes import FakeRuntimeExecutor, FakeRuntimeLoader


def test_chat_session_round_trip_and_retry(tmp_path: Path) -> None:
    loader = FakeRuntimeLoader()
    executor = FakeRuntimeExecutor()
    session = ChatSession(
        runtime_loader=loader,
        runtime_executor=executor,
        runtime_config=RuntimeConfig(),
        generation_config=GenerationConfig(stream=False),
        autosave_path=tmp_path / "session.json",
    )

    first = session.prompt_text("hello")
    assert first.text == "echo:hello"
    assert [message.text_content() for message in session.messages] == ["hello", "echo:hello"]
    assert loader.load_calls == ["llama3-1B-chat"]

    retried = session.retry_last()
    assert retried.text == "echo:hello"
    assert executor.prompts == ["hello", "hello"]
    assert [message.text_content() for message in session.messages] == ["hello", "echo:hello"]

    session.undo_last_exchange()
    assert session.messages == []

    saved = load_transcript(tmp_path / "session.json")
    assert saved.model_id == "llama3-1B-chat"
    assert saved.system_prompt == session.system_prompt


def test_chat_session_loads_and_switches_model(tmp_path: Path) -> None:
    loader = FakeRuntimeLoader()
    executor = FakeRuntimeExecutor()
    session = ChatSession(
        runtime_loader=loader,
        runtime_executor=executor,
        runtime_config=RuntimeConfig(),
        generation_config=GenerationConfig(stream=False),
        session_name="chat-a",
    )
    session.prompt_text("hello")
    path = tmp_path / "saved.json"
    session.save(path)

    restored = ChatSession(
        runtime_loader=loader,
        runtime_executor=executor,
        runtime_config=RuntimeConfig(model_id="llama3-3B-chat"),
        generation_config=GenerationConfig(stream=False),
    )
    restored.load(path)
    assert restored.session_name == "chat-a"
    assert restored.runtime_config.model_id == "llama3-1B-chat"
    restored.set_model("llama3-3B-chat")
    assert restored.runtime_config.model_id == "llama3-3B-chat"

import io
import stat
from typing import cast

import pytest
from rich.console import Console

from ollm.app.session import ChatSession
from ollm.runtime.config import GenerationConfig, RuntimeConfig
from ollm.runtime.generation import RuntimeExecutor
from ollm.runtime.loader import RuntimeLoader
from ollm.ui.chat_shell import InteractiveChatShell, SlashCommand, parse_slash_command
from tests.fakes import FakeRuntimeExecutor, FakeRuntimeLoader


def test_parse_slash_command() -> None:
    command = parse_slash_command("/model llama3-3B-chat")
    assert command is not None
    assert command.name == "model"
    assert command.argument == "llama3-3B-chat"


def test_parse_slash_command_requires_name() -> None:
    with pytest.raises(ValueError):
        parse_slash_command("/")


def test_shell_queues_attachments_and_send_clears_queue() -> None:
    session = ChatSession(
        runtime_loader=cast(RuntimeLoader, FakeRuntimeLoader()),
        runtime_executor=cast(RuntimeExecutor, FakeRuntimeExecutor()),
        runtime_config=RuntimeConfig(multimodal=True),
        generation_config=GenerationConfig(stream=False),
    )
    output = io.StringIO()
    shell = InteractiveChatShell(
        session=session, console=Console(file=output, force_terminal=False)
    )
    shell._handle_command(SlashCommand(name="image", argument="diagram.png"))
    shell._handle_command(SlashCommand(name="send", argument="describe this"))
    assert [part.kind.value for part in session.messages[0].content] == [
        "image",
        "text",
    ]
    assert shell._pending_parts == []


def test_retry_prints_text_in_non_stream_mode() -> None:
    session = ChatSession(
        runtime_loader=cast(RuntimeLoader, FakeRuntimeLoader()),
        runtime_executor=cast(RuntimeExecutor, FakeRuntimeExecutor()),
        runtime_config=RuntimeConfig(),
        generation_config=GenerationConfig(stream=False),
    )
    session.prompt_text("hello")
    output = io.StringIO()
    shell = InteractiveChatShell(
        session=session, console=Console(file=output, force_terminal=False)
    )
    shell._handle_command(SlashCommand(name="retry", argument=""))
    assert "echo:hello" in output.getvalue()


def test_history_file_is_opt_in_and_private(tmp_path) -> None:
    session = ChatSession(
        runtime_loader=cast(RuntimeLoader, FakeRuntimeLoader()),
        runtime_executor=cast(RuntimeExecutor, FakeRuntimeExecutor()),
        runtime_config=RuntimeConfig(),
        generation_config=GenerationConfig(stream=False),
    )
    history_path = tmp_path / "history" / "chat.txt"
    shell = InteractiveChatShell(
        session=session,
        console=Console(file=io.StringIO(), force_terminal=False),
        history_file=history_path,
    )
    del shell
    assert history_path.exists()
    assert stat.S_IMODE(history_path.stat().st_mode) == 0o600

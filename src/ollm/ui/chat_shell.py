import os
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory, InMemoryHistory
from rich.console import Console

from ollm.app.session import ChatSession
from ollm.app.types import ContentPart
from ollm.runtime.streaming import StreamSink


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    argument: str


class ConsoleStreamSink(StreamSink):
    def __init__(self, console: Console):
        self._console = console
        self._line_open = False

    def on_status(self, message: str) -> None:
        self._console.print(f"[cyan]{message}[/cyan]")

    def on_text(self, text: str) -> None:
        if not self._line_open:
            self._console.print("[bold green]assistant>[/bold green] ", end="")
            self._line_open = True
        self._console.print(text, end="")

    def on_complete(self, text: str) -> None:
        del text
        if self._line_open:
            self._console.print()
            self._line_open = False


def parse_slash_command(raw_text: str) -> SlashCommand | None:
    stripped = raw_text.strip()
    if not stripped.startswith("/"):
        return None
    body = stripped[1:]
    if not body:
        raise ValueError("Slash commands require a command name")
    name, _, argument = body.partition(" ")
    return SlashCommand(name=name.strip().lower(), argument=argument.strip())


class InteractiveChatShell:
    def __init__(
        self,
        session: ChatSession,
        console: Console,
        history_file: Path | None = None,
        plain: bool = False,
    ):
        self._session = session
        self._console = console
        self._plain = plain
        self._pending_parts: list[ContentPart] = []
        history = InMemoryHistory()
        if history_file is not None:
            history_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(history_file, os.O_CREAT | os.O_APPEND, 0o600)
            os.close(fd)
            os.chmod(history_file, 0o600)
            history = FileHistory(str(history_file))
        self._prompt_session = PromptSession(history=history)

    def run(self) -> None:
        self._render_banner()
        while True:
            try:
                text = self._prompt_session.prompt("user> ")
            except (EOFError, KeyboardInterrupt):
                self._console.print()
                self._console.print("Exiting chat.")
                return

            stripped = text.strip()
            if not stripped:
                continue
            command = parse_slash_command(stripped)
            if command is not None:
                try:
                    if self._handle_command(command):
                        return
                except Exception as exc:
                    self._console.print(f"[red]{exc}[/red]")
                continue

            self._submit_parts([ContentPart.text(stripped)])

    def _handle_command(self, command: SlashCommand) -> bool:
        if command.name == "help":
            self._console.print(
                "Commands: /help /clear /reset /system /model /stats /save /load "
                "/retry /undo /image /audio /attachments /clear-attachments /send /exit"
            )
            return False
        if command.name == "clear":
            self._session.clear()
            self._pending_parts.clear()
            self._console.print("Conversation cleared.")
            return False
        if command.name == "reset":
            self._session.reset()
            self._pending_parts.clear()
            self._console.print("Conversation and system prompt reset.")
            return False
        if command.name == "system":
            if command.argument:
                self._session.set_system_prompt(command.argument)
                self._console.print("System prompt updated.")
            else:
                self._console.print(self._session.system_prompt)
            return False
        if command.name == "model":
            if command.argument:
                self._session.set_model(command.argument)
                self._pending_parts.clear()
                self._console.print(f"Model switched to {command.argument}.")
            else:
                self._console.print(self._session.runtime_config.model_reference)
            return False
        if command.name == "stats":
            self._console.print(
                f"stats={self._session.runtime_config.stats} "
                f"model={self._session.runtime_config.model_reference} "
                f"messages={len(self._session.messages)}"
            )
            return False
        if command.name == "save":
            if not command.argument:
                raise ValueError("/save requires a path")
            path = Path(command.argument).expanduser().resolve()
            self._session.save(path)
            self._console.print(f"Saved transcript to {path}")
            return False
        if command.name == "load":
            if not command.argument:
                raise ValueError("/load requires a path")
            path = Path(command.argument).expanduser().resolve()
            self._session.load(path)
            self._pending_parts.clear()
            self._console.print(f"Loaded transcript from {path}")
            return False
        if command.name == "retry":
            sink = ConsoleStreamSink(self._console)
            response = self._session.retry_last(sink=sink)
            if not response.text and not self._session.generation_config.stream:
                self._console.print("[yellow]Assistant returned no text.[/yellow]")
            elif not self._session.generation_config.stream:
                self._render_assistant_message(response.text)
            return False
        if command.name == "undo":
            self._session.undo_last_exchange()
            self._console.print("Removed the last user/assistant exchange.")
            return False
        if command.name == "image":
            if not command.argument:
                raise ValueError("/image requires a path or URL")
            self._pending_parts.append(ContentPart.image(command.argument))
            self._console.print(f"Queued image attachment: {command.argument}")
            return False
        if command.name == "audio":
            if not command.argument:
                raise ValueError("/audio requires a path or URL")
            self._pending_parts.append(ContentPart.audio(command.argument))
            self._console.print(f"Queued audio attachment: {command.argument}")
            return False
        if command.name == "attachments":
            self._render_pending_attachments()
            return False
        if command.name == "clear-attachments":
            self._pending_parts.clear()
            self._console.print("Cleared queued attachments.")
            return False
        if command.name == "send":
            if not self._pending_parts and not command.argument:
                raise ValueError("/send requires queued attachments or inline text")
            parts: list[ContentPart] = []
            if command.argument:
                parts.append(ContentPart.text(command.argument))
            self._submit_parts(parts)
            return False
        if command.name == "exit":
            self._console.print("Exiting chat.")
            return True
        raise ValueError(f"Unknown slash command '/{command.name}'")

    def _render_banner(self) -> None:
        if self._plain:
            self._console.print(f"oLLM chat ({self._session.runtime_config.model_reference})")
            return
        self._console.print(f"[bold]oLLM chat[/bold] using [cyan]{self._session.runtime_config.model_reference}[/cyan]")
        self._console.print("Type /help for commands.")

    def _render_assistant_message(self, text: str) -> None:
        if self._plain:
            self._console.print(f"assistant> {text}")
            return
        self._console.print(f"[bold green]assistant>[/bold green] {text}")

    def _render_pending_attachments(self) -> None:
        if not self._pending_parts:
            self._console.print("No queued attachments.")
            return
        self._console.print("Queued attachments:")
        for part in self._pending_parts:
            self._console.print(f"- {part.kind.value}: {part.value}")

    def _submit_parts(self, input_parts: list[ContentPart]) -> None:
        parts = list(self._pending_parts)
        parts.extend(input_parts)
        if not parts:
            raise ValueError("There is nothing to send")
        sink = ConsoleStreamSink(self._console)
        try:
            response = self._session.prompt_parts(parts, sink=sink)
        except Exception as exc:
            self._console.print(f"[red]{exc}[/red]")
            return
        self._pending_parts.clear()
        if not response.text:
            self._console.print("[yellow]Assistant returned no text.[/yellow]")
        elif not self._session.generation_config.stream:
            self._render_assistant_message(response.text)
        if "stats" in response.metadata:
            self._console.print(f"[dim]{response.metadata['stats']}[/dim]")

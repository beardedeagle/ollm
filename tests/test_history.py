import stat
from pathlib import Path

from ollm.app.history import TRANSCRIPT_VERSION, load_transcript, save_transcript
from ollm.app.types import Message, Transcript



def test_transcript_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "transcript.json"
    transcript = Transcript(
        version=TRANSCRIPT_VERSION,
        session_name="demo",
        model_reference="llama3-1B-chat",
        system_prompt="You are helpful.",
        messages=[Message.user_text("hello"), Message.assistant_text("hi")],
    )
    save_transcript(path, transcript)
    loaded = load_transcript(path)
    assert loaded.session_name == "demo"
    assert loaded.model_reference == "llama3-1B-chat"
    assert [message.text_content() for message in loaded.messages] == ["hello", "hi"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

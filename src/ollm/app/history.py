import json
import os
from pathlib import Path

from ollm.app.types import Transcript

TRANSCRIPT_VERSION = 2


def write_private_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
    finally:
        try:
            os.chmod(path, 0o600)
        except FileNotFoundError:
            pass


def save_transcript(path: Path, transcript: Transcript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.tmp"
    write_private_text(temp_path, json.dumps(transcript.as_dict(), indent=2) + "\n")
    temp_path.replace(path)
    os.chmod(path, 0o600)


def load_transcript(path: Path) -> Transcript:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transcript = Transcript.from_dict(payload)
    if transcript.version != TRANSCRIPT_VERSION:
        raise ValueError(
            f"Unsupported transcript version {transcript.version}. "
            f"Expected version {TRANSCRIPT_VERSION}."
        )
    return transcript

import json
from pathlib import Path

from ollm.app.types import Transcript
from ollm.async_io import (
    chmod_path,
    path_mkdir,
    path_read_text,
    path_replace,
    path_write_text,
)

TRANSCRIPT_VERSION = 2


def write_private_text(path: Path, content: str) -> None:
    path_mkdir(path.parent, parents=True, exist_ok=True, mode=0o700)
    path_write_text(path, content, encoding="utf-8")
    try:
        chmod_path(path, 0o600)
    except FileNotFoundError:
        pass


def save_transcript(path: Path, transcript: Transcript) -> None:
    path_mkdir(path.parent, parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.tmp"
    write_private_text(temp_path, json.dumps(transcript.as_dict(), indent=2) + "\n")
    path_replace(temp_path, path)
    chmod_path(path, 0o600)


def load_transcript(path: Path) -> Transcript:
    payload = json.loads(path_read_text(path, encoding="utf-8"))
    transcript = Transcript.from_dict(payload)
    if transcript.version != TRANSCRIPT_VERSION:
        raise ValueError(
            f"Unsupported transcript version {transcript.version}. "
            f"Expected version {TRANSCRIPT_VERSION}."
        )
    return transcript

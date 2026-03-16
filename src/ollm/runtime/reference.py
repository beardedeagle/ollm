"""Parsing for opaque model references accepted by the oLLM CLI and library APIs."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self

KNOWN_REFERENCE_SCHEMES = {
    "hf",
    "path",
}
_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/].+")


@dataclass(frozen=True, slots=True)
class ModelReference:
    """Normalized representation of a user-supplied model reference string."""

    raw: str
    scheme: str | None
    identifier: str
    revision: str | None
    local_path: Path | None

    @classmethod
    def parse(cls, raw_reference: str) -> Self:
        """Parse a raw model reference into a structured form."""
        raw = raw_reference.strip()
        if not raw:
            raise ValueError("Model reference cannot be empty")

        scheme, value = _split_reference(raw)
        if scheme == "path":
            path = Path(value).expanduser().resolve()
            return cls(
                raw=raw,
                scheme=scheme,
                identifier=str(path),
                revision=None,
                local_path=path,
            )

        if scheme is None and _looks_like_local_path(raw):
            path = Path(raw).expanduser().resolve()
            return cls(
                raw=raw,
                scheme=None,
                identifier=str(path),
                revision=None,
                local_path=path,
            )

        identifier, revision = _split_revision(value)
        return cls(
            raw=raw,
            scheme=scheme,
            identifier=identifier,
            revision=revision,
            local_path=None,
        )

    def is_huggingface_reference(self) -> bool:
        """Return whether the reference should be treated as a Hugging Face repo ID."""
        if self.scheme == "hf":
            return True
        if self.scheme is not None:
            return False
        if "/" not in self.identifier:
            return False
        return " " not in self.identifier

    def materialization_name(self) -> str:
        """Return the directory name used for local materialization of this reference."""
        base = self.identifier.replace("/", "--").replace(":", "--")
        if self.revision is None:
            return base
        revision = self.revision.replace("/", "--").replace(":", "--")
        return f"{base}--{revision}"


def _split_reference(raw_reference: str) -> tuple[str | None, str]:
    if "://" in raw_reference:
        scheme, _, remainder = raw_reference.partition("://")
        if not scheme:
            raise ValueError("Model reference scheme cannot be empty")
        return scheme.lower(), remainder

    scheme, separator, remainder = raw_reference.partition(":")
    if (
        separator
        and scheme.lower() in KNOWN_REFERENCE_SCHEMES
        and not _WINDOWS_PATH_RE.match(raw_reference)
    ):
        return scheme.lower(), remainder
    return None, raw_reference


def _looks_like_local_path(raw_reference: str) -> bool:
    if raw_reference.startswith(("/", "./", "../", "~")):
        return True
    if _WINDOWS_PATH_RE.match(raw_reference):
        return True
    candidate = Path(raw_reference)
    return candidate.exists() and candidate.is_dir()


def _split_revision(value: str) -> tuple[str, str | None]:
    identifier, separator, revision = value.partition("@")
    if not separator:
        return value, None
    if not identifier or not revision:
        raise ValueError(
            f"Invalid model reference '{value}'. Revisions must use <reference>@<revision>."
        )
    return identifier, revision

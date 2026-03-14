import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

DEFAULT_REMOTE_IMAGE_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_REMOTE_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_REMOTE_AUDIO_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_REMOTE_AUDIO_BYTES = 20 * 1024 * 1024
_CHUNK_SIZE = 64 * 1024
_SUPPORTED_AUDIO_MEDIA_TYPES = {
	"audio/mpeg": "mp3",
	"audio/mp3": "mp3",
	"audio/wav": "wav",
	"audio/wave": "wav",
	"audio/x-wav": "wav",
	"audio/vnd.wave": "wav",
}
_SUPPORTED_AUDIO_SUFFIXES = {
	".mp3": "mp3",
	".wav": "wav",
}


@dataclass(frozen=True, slots=True)
class ResolvedAudioInput:
	base64_data: str
	audio_format: str


def encode_image_input_base64(
	value: str,
	*,
	timeout_seconds: float = DEFAULT_REMOTE_IMAGE_TIMEOUT_SECONDS,
	max_bytes: int = DEFAULT_MAX_REMOTE_IMAGE_BYTES,
) -> str:
	return base64.b64encode(
		resolve_image_input_bytes(
			value,
			timeout_seconds=timeout_seconds,
			max_bytes=max_bytes,
		)
	).decode("ascii")


def resolve_image_input_bytes(
	value: str,
	*,
	timeout_seconds: float = DEFAULT_REMOTE_IMAGE_TIMEOUT_SECONDS,
	max_bytes: int = DEFAULT_MAX_REMOTE_IMAGE_BYTES,
) -> bytes:
	if value.startswith("data:"):
		return _decode_data_url_image(value, max_bytes=max_bytes)

	parsed = urlparse(value)
	if parsed.scheme in {"http", "https"}:
		return _download_remote_image(
			value,
			timeout_seconds=timeout_seconds,
			max_bytes=max_bytes,
		)
	if parsed.scheme == "file":
		return _read_local_image(path=Path(unquote(parsed.path)), max_bytes=max_bytes)

	image_path = Path(value).expanduser()
	if image_path.exists() and image_path.is_file():
		return _read_local_image(path=image_path, max_bytes=max_bytes)

	raise ValueError(
		"Image inputs require a local file path, a base64 data URL, or an http/https image URL"
	)


def resolve_audio_input(
	value: str,
	*,
	timeout_seconds: float = DEFAULT_REMOTE_AUDIO_TIMEOUT_SECONDS,
	max_bytes: int = DEFAULT_MAX_REMOTE_AUDIO_BYTES,
) -> ResolvedAudioInput:
	if value.startswith("data:"):
		audio_bytes, audio_format = _decode_data_url_audio(value, max_bytes=max_bytes)
		return _resolved_audio_input(audio_bytes, audio_format)

	parsed = urlparse(value)
	if parsed.scheme in {"http", "https"}:
		audio_bytes, audio_format = _download_remote_audio(
			value,
			timeout_seconds=timeout_seconds,
			max_bytes=max_bytes,
		)
		return _resolved_audio_input(audio_bytes, audio_format)
	if parsed.scheme == "file":
		audio_bytes, audio_format = _read_local_audio(path=Path(unquote(parsed.path)), max_bytes=max_bytes)
		return _resolved_audio_input(audio_bytes, audio_format)

	audio_path = Path(value).expanduser()
	if audio_path.exists() and audio_path.is_file():
		audio_bytes, audio_format = _read_local_audio(path=audio_path, max_bytes=max_bytes)
		return _resolved_audio_input(audio_bytes, audio_format)

	raise ValueError(
		"Audio inputs require a local .wav/.mp3 file path, a base64 data URL, or an http/https audio URL"
	)


def _decode_data_url_image(value: str, *, max_bytes: int) -> bytes:
	header, separator, encoded = value.partition(",")
	if not separator or not encoded:
		raise ValueError("Data URL image inputs must contain a base64 payload")
	if not header.lower().startswith("data:image/"):
		raise ValueError("Data URL image inputs must declare an image media type")
	if ";base64" not in header.lower():
		raise ValueError("Data URL image inputs for Ollama must be base64-encoded")
	try:
		decoded_bytes = base64.b64decode(encoded, validate=True)
	except (ValueError, binascii.Error) as exc:
		raise ValueError("Data URL image inputs for Ollama contain invalid base64 data") from exc
	if not decoded_bytes:
		raise ValueError("Data URL image inputs must contain image bytes")
	if len(decoded_bytes) > max_bytes:
		raise ValueError(f"Image input exceeds the {max_bytes} byte size limit")
	return decoded_bytes


def _decode_data_url_audio(value: str, *, max_bytes: int) -> tuple[bytes, str]:
	header, separator, encoded = value.partition(",")
	if not separator or not encoded:
		raise ValueError("Data URL audio inputs must contain a base64 payload")
	audio_format = _audio_format_from_media_type(_data_url_media_type(header))
	if audio_format is None:
		raise ValueError("Data URL audio inputs currently support only WAV and MP3 media types")
	if ";base64" not in header.lower():
		raise ValueError("Data URL audio inputs must be base64-encoded")
	try:
		decoded_bytes = base64.b64decode(encoded, validate=True)
	except (ValueError, binascii.Error) as exc:
		raise ValueError("Data URL audio inputs contain invalid base64 data") from exc
	if not decoded_bytes:
		raise ValueError("Data URL audio inputs must contain audio bytes")
	if len(decoded_bytes) > max_bytes:
		raise ValueError(f"Audio input exceeds the {max_bytes} byte size limit")
	return decoded_bytes, audio_format


def _read_local_image(path: Path, *, max_bytes: int) -> bytes:
	return _read_local_bytes(path=path, max_bytes=max_bytes, kind_label="Image")


def _read_local_audio(path: Path, *, max_bytes: int) -> tuple[bytes, str]:
	resolved_path = path.expanduser().resolve()
	audio_format = _audio_format_from_path(resolved_path)
	if audio_format is None:
		raise ValueError("Audio inputs currently support only local .wav and .mp3 files")
	return (
		_read_local_bytes(path=resolved_path, max_bytes=max_bytes, kind_label="Audio"),
		audio_format,
	)


def _download_remote_image(
	value: str,
	*,
	timeout_seconds: float,
	max_bytes: int,
) -> bytes:
	request = Request(
		value,
		headers={
			"Accept": "image/*",
			"User-Agent": "ollm/remote-image-fetch",
		},
		method="GET",
	)
	try:
		with urlopen(request, timeout=timeout_seconds) as response:
			content_type = response.headers.get_content_type()
			if not isinstance(content_type, str) or not content_type.startswith("image/"):
				raise ValueError(
					f"Remote image URL must return an image content type; received {content_type!r}"
				)
			content_length = response.headers.get("Content-Length")
			if content_length is not None:
				try:
					resolved_content_length = int(content_length)
				except ValueError as exc:
					raise ValueError(
						f"Remote image URL returned an invalid Content-Length header: {content_length!r}"
					) from exc
				if resolved_content_length > max_bytes:
					raise ValueError(
						f"Remote image URL exceeds the {max_bytes} byte download limit"
					)
			return _read_bounded_bytes(response, max_bytes=max_bytes, kind_label="image")
	except HTTPError as exc:
		raise ValueError(
			f"Failed to fetch remote image URL {value}: HTTP {exc.code}"
		) from exc
	except URLError as exc:
		raise ValueError(
			f"Failed to fetch remote image URL {value}: {exc.reason}"
		) from exc


def _download_remote_audio(
	value: str,
	*,
	timeout_seconds: float,
	max_bytes: int,
) -> tuple[bytes, str]:
	request = Request(
		value,
		headers={
			"Accept": "audio/mpeg,audio/wav,audio/x-wav",
			"User-Agent": "ollm/remote-audio-fetch",
		},
		method="GET",
	)
	try:
		with urlopen(request, timeout=timeout_seconds) as response:
			content_type = response.headers.get_content_type()
			audio_format = _audio_format_from_media_type(content_type)
			if audio_format is None:
				raise ValueError(
					f"Remote audio URL must return a WAV or MP3 audio content type; received {content_type!r}"
				)
			content_length = response.headers.get("Content-Length")
			if content_length is not None:
				try:
					resolved_content_length = int(content_length)
				except ValueError as exc:
					raise ValueError(
						f"Remote audio URL returned an invalid Content-Length header: {content_length!r}"
					) from exc
				if resolved_content_length > max_bytes:
					raise ValueError(
						f"Remote audio URL exceeds the {max_bytes} byte download limit"
					)
			audio_bytes = _read_bounded_bytes(response, max_bytes=max_bytes, kind_label="audio")
			return audio_bytes, audio_format
	except HTTPError as exc:
		raise ValueError(
			f"Failed to fetch remote audio URL {value}: HTTP {exc.code}"
		) from exc
	except URLError as exc:
		raise ValueError(
			f"Failed to fetch remote audio URL {value}: {exc.reason}"
		) from exc


def _read_local_bytes(path: Path, *, max_bytes: int, kind_label: str) -> bytes:
	resolved_path = path.expanduser().resolve()
	if not resolved_path.exists() or not resolved_path.is_file():
		raise ValueError(f"{kind_label} file does not exist: {resolved_path}")
	if resolved_path.stat().st_size > max_bytes:
		raise ValueError(f"{kind_label} input exceeds the {max_bytes} byte size limit")
	return resolved_path.read_bytes()


def _read_bounded_bytes(response, *, max_bytes: int, kind_label: str = "media") -> bytes:
	buffer = bytearray()
	while True:
		chunk = response.read(_CHUNK_SIZE)
		if not chunk:
			break
		buffer.extend(chunk)
		if len(buffer) > max_bytes:
			raise ValueError(f"Remote {kind_label} URL exceeds the {max_bytes} byte download limit")
	if not buffer:
		raise ValueError(f"Remote {kind_label} URL returned an empty response body")
	return bytes(buffer)


def _resolved_audio_input(audio_bytes: bytes, audio_format: str) -> ResolvedAudioInput:
	return ResolvedAudioInput(
		base64_data=base64.b64encode(audio_bytes).decode("ascii"),
		audio_format=audio_format,
	)


def _data_url_media_type(header: str) -> str:
	media_type = header.partition(",")[0].partition(";")[0]
	if not media_type:
		return ""
	return media_type.removeprefix("data:").lower()


def _audio_format_from_media_type(media_type: str | None) -> str | None:
	if media_type is None:
		return None
	return _SUPPORTED_AUDIO_MEDIA_TYPES.get(media_type.lower())


def _audio_format_from_path(path: Path) -> str | None:
	return _SUPPORTED_AUDIO_SUFFIXES.get(path.suffix.lower())

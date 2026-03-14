import base64
import binascii
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

DEFAULT_REMOTE_IMAGE_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_REMOTE_IMAGE_BYTES = 20 * 1024 * 1024
_CHUNK_SIZE = 64 * 1024


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


def _read_local_image(path: Path, *, max_bytes: int) -> bytes:
	resolved_path = path.expanduser().resolve()
	if not resolved_path.exists() or not resolved_path.is_file():
		raise ValueError(f"Image file does not exist: {resolved_path}")
	if resolved_path.stat().st_size > max_bytes:
		raise ValueError(f"Image input exceeds the {max_bytes} byte size limit")
	return resolved_path.read_bytes()


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
			return _read_bounded_bytes(response, max_bytes=max_bytes)
	except HTTPError as exc:
		raise ValueError(
			f"Failed to fetch remote image URL {value}: HTTP {exc.code}"
		) from exc
	except URLError as exc:
		raise ValueError(
			f"Failed to fetch remote image URL {value}: {exc.reason}"
		) from exc


def _read_bounded_bytes(response, *, max_bytes: int) -> bytes:
	buffer = bytearray()
	while True:
		chunk = response.read(_CHUNK_SIZE)
		if not chunk:
			break
		buffer.extend(chunk)
		if len(buffer) > max_bytes:
			raise ValueError(f"Remote image URL exceeds the {max_bytes} byte download limit")
	if not buffer:
		raise ValueError("Remote image URL returned an empty response body")
	return bytes(buffer)

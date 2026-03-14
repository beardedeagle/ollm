import base64
from pathlib import Path

import pytest

from ollm.runtime.media_inputs import (
	DEFAULT_MAX_REMOTE_IMAGE_BYTES,
	encode_image_input_base64,
)

from tests.media_server import MediaFixtureServer, MediaResponse


def test_encode_image_input_base64_reads_local_file(tmp_path: Path) -> None:
	image_path = tmp_path / "diagram.png"
	image_path.write_bytes(b"png-bytes")

	assert encode_image_input_base64(str(image_path)) == base64.b64encode(b"png-bytes").decode("ascii")


def test_encode_image_input_base64_decodes_data_url() -> None:
	assert (
		encode_image_input_base64("data:image/png;base64,cG5nLWJ5dGVz")
		== base64.b64encode(b"png-bytes").decode("ascii")
	)


def test_encode_image_input_base64_rejects_large_data_url() -> None:
	with pytest.raises(ValueError, match="size limit"):
		encode_image_input_base64(
			"data:image/png;base64," + base64.b64encode(b"x" * (DEFAULT_MAX_REMOTE_IMAGE_BYTES + 1)).decode("ascii")
		)


def test_encode_image_input_base64_fetches_remote_image() -> None:
	server = MediaFixtureServer(
		responses={
			"/image.png": MediaResponse(body=b"remote-png", content_type="image/png"),
		}
	)
	server.start()
	try:
		assert (
			encode_image_input_base64(f"{server.base_url}/image.png")
			== base64.b64encode(b"remote-png").decode("ascii")
		)
	finally:
		server.stop()


def test_encode_image_input_base64_rejects_large_local_file(tmp_path: Path) -> None:
	image_path = tmp_path / "large.png"
	image_path.write_bytes(b"x" * (DEFAULT_MAX_REMOTE_IMAGE_BYTES + 1))

	with pytest.raises(ValueError, match="size limit"):
		encode_image_input_base64(str(image_path))


def test_encode_image_input_base64_rejects_non_image_response() -> None:
	server = MediaFixtureServer(
		responses={
			"/not-image": MediaResponse(body=b"hello", content_type="text/plain"),
		}
	)
	server.start()
	try:
		with pytest.raises(ValueError, match="image content type"):
			encode_image_input_base64(f"{server.base_url}/not-image")
	finally:
		server.stop()


def test_encode_image_input_base64_rejects_http_error() -> None:
	server = MediaFixtureServer(responses={})
	server.start()
	try:
		with pytest.raises(ValueError, match="HTTP 404"):
			encode_image_input_base64(f"{server.base_url}/missing.png")
	finally:
		server.stop()


def test_encode_image_input_base64_surfaces_http_fetch_failures() -> None:
	server = MediaFixtureServer(responses={})
	server.start()
	try:
		with pytest.raises(ValueError, match="HTTP 404"):
			encode_image_input_base64(f"{server.base_url}/missing.png")
	finally:
		server.stop()


def test_encode_image_input_base64_rejects_large_remote_images() -> None:
	server = MediaFixtureServer(
		responses={
			"/large.png": MediaResponse(
				body=b"x" * (DEFAULT_MAX_REMOTE_IMAGE_BYTES + 1),
				content_type="image/png",
			),
		}
	)
	server.start()
	try:
		with pytest.raises(ValueError, match="download limit"):
			encode_image_input_base64(f"{server.base_url}/large.png")
	finally:
		server.stop()


def test_encode_image_input_base64_rejects_unsupported_scheme() -> None:
	with pytest.raises(ValueError, match="local file path, a base64 data URL, or an http/https image URL"):
		encode_image_input_base64("ftp://example.com/image.png")

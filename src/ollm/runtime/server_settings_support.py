"""Helpers for local-server settings validation."""

DEFAULT_RESPONSE_STORE_BACKEND = "none"
KNOWN_RESPONSE_STORE_BACKENDS = ("none", "memory", "plugin")


def normalize_response_store_backend(backend: str | None) -> str | None:
    """Validate and normalize a response-store backend identifier."""
    if backend is None:
        return None
    normalized_backend = backend.strip().lower()
    if not normalized_backend:
        raise ValueError("response_store_backend cannot be empty")
    if normalized_backend not in KNOWN_RESPONSE_STORE_BACKENDS:
        allowed_backends = ", ".join(KNOWN_RESPONSE_STORE_BACKENDS)
        raise ValueError(f"response_store_backend must be one of: {allowed_backends}")
    return normalized_backend

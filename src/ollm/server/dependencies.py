"""Shared optional-dependency errors for the oLLM server surface."""

SERVER_EXTRA_INSTALL_HINT = (
    "Install server support with `uv sync --extra server` or "
    '`pip install --no-build-isolation -e ".[server]"`.'
)


class ServerDependenciesError(RuntimeError):
    """Raised when optional server transport dependencies are unavailable."""

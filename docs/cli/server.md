# `ollm serve`

`ollm serve` starts oLLM's optional local-only REST API server.

## Key options

- `--host`
- `--port`
- `--reload / --no-reload`
- `--log-level`

## Settings precedence

`ollm serve` uses the same precedence contract as the rest of oLLM:

1. CLI flags
2. `OLLM_SERVER__*` environment variables
3. TOML config values under `[server]`
4. built-in defaults

The default bind is `127.0.0.1:8000`.

## OpenAPI and docs endpoints

When the server is running locally, FastAPI publishes:

- `/openapi.json`
- `/docs`
- `/redoc`

## Examples

```bash
uv sync --extra server
ollm serve
ollm serve --port 9001 --log-level debug
OLLM_SERVER__PORT=8123 ollm serve
```

See [Local Server API](../guides/local-server.md) for the current HTTP route
surface.

# `ollm serve`

`ollm serve` starts oLLM's optional local-only REST API server.

## Key options

- `--host`
- `--port`
- `--reload / --no-reload`
- `--log-level`
- `--response-store-backend`
- `--response-store-factory`

## Settings precedence

`ollm serve` uses the same precedence contract as the rest of oLLM:

1. CLI flags
2. `OLLM_SERVER__*` environment variables
3. TOML config values under `[server]`
4. built-in defaults

The default bind is `127.0.0.1:8000`.

Responses storage is disabled by default. Use:

- `--response-store-backend memory` for process-scoped dev/test retrieval
- `--response-store-backend plugin --response-store-factory package.module:factory`
  for a custom backend

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
ollm serve --response-store-backend memory
ollm serve --response-store-backend plugin \
  --response-store-factory custom.module:build_store
OLLM_SERVER__PORT=8123 ollm serve
```

See [Local Server API](../guides/local-server.md) for the HTTP route
surface.

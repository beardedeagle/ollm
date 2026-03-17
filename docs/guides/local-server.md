# Local Server API

The optional server extra exposes a local-only FastAPI application on top of the
same planning and execution stack used by the CLI.

## Start the server

```bash
uv sync --extra server
ollm serve
```

The default bind is `127.0.0.1:8000`.

## OpenAPI surfaces

The local server publishes:

- `/openapi.json` for machine-readable schema access
- `/docs` for Swagger UI
- `/redoc` for ReDoc

## Current routes

- `GET /v1/health`
- `GET /v1/models`
- `GET /v1/models/{model_reference}`
- `POST /v1/plan`
- `POST /v1/prompt`
- `POST /v1/prompt/stream`
- `POST /v1/sessions`
- `GET /v1/sessions/{session_id}`
- `POST /v1/sessions/{session_id}/prompt`
- `POST /v1/sessions/{session_id}/prompt/stream`

## Behavior notes

- The bind is local-only by default.
- Streaming prompt responses use SSE.
- Server-side sessions are in-memory only in the current slice.
- Runtime and generation defaults still follow the standard config layering
  contract.

## Example requests

```bash
curl http://127.0.0.1:8000/v1/health

curl -X POST http://127.0.0.1:8000/v1/plan \
  -H "content-type: application/json" \
  -d '{"runtime":{"model_reference":"llama3-1B-chat"}}'

curl -N -X POST http://127.0.0.1:8000/v1/prompt/stream \
  -H "content-type: application/json" \
  -d '{"prompt":"List planets","runtime":{"model_reference":"llama3-1B-chat"}}'
```

Use `examples/ollm.toml` from the repository root as a starting point when you
want shared CLI and server defaults.

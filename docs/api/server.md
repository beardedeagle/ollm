# Local Server API Reference

oLLM's optional server transport is a local-only FastAPI application. Start it
with `ollm serve`.

## Schema and docs endpoints

- `/openapi.json`
- `/docs`
- `/redoc`

## Route reference

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

## Semantics

- The server bind is local-only by default.
- Streaming responses use SSE.
- Server-side sessions are in-memory only.
- The HTTP transport reuses the same `ApplicationService` runtime planning and
  prompt execution logic as the CLI.

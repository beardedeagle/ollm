# Local Server API Reference

oLLM's optional server transport is a local-only FastAPI application. Start it
with `ollm serve`.

## Schema and docs endpoints

- `/openapi.json`
- `/docs`
- `/redoc`

## OpenAI-compatible routes

- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/chat/completions`

## Native oLLM routes

- `GET /v1/health`
- `GET /v1/ollm/models`
- `GET /v1/ollm/models/{model_reference}`
- `POST /v1/plan`
- `POST /v1/prompt`
- `POST /v1/prompt/stream`
- `POST /v1/sessions`
- `GET /v1/sessions/{session_id}`
- `POST /v1/sessions/{session_id}/prompt`
- `POST /v1/sessions/{session_id}/prompt/stream`

## Compatibility scope

- The OpenAI-compatible surface currently targets text chat and model discovery.
- Chat-completions requests currently support plain string content and structured
  text-part arrays only.
- `POST /v1/chat/completions` supports both standard JSON responses and SSE
  streaming responses.
- The server continues to expose native oLLM-only runtime planning, prompt, and
  session endpoints beside the compatibility layer.
- `POST /v1/responses` is intentionally not implemented in this slice. Treat it
  as a follow-up once the chat-completions compatibility contract is stable.

## Semantics

- The server bind is local-only by default.
- OpenAI-compatible streaming responses use `text/event-stream` chat-completion
  chunks plus a final `data: [DONE]` line.
- Native streaming responses still use oLLM's SSE event family.
- Server-side sessions are in-memory only.
- The HTTP transport reuses the same `ApplicationService` runtime planning and
  prompt execution logic as the CLI.

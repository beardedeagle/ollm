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
- `POST /v1/responses`
- `GET /v1/responses/{response_id}`

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

- The OpenAI-compatible surface currently targets text chat, text responses, and
  model discovery.
- Chat-completions requests currently support plain string content and structured
  text-part arrays only.
- Responses requests currently support plain string input or message arrays with
  text, image, and audio content parts.
- `POST /v1/chat/completions` supports both standard JSON responses and SSE
  streaming responses.
- `POST /v1/responses` supports both standard JSON responses and typed SSE
  response events.
- `GET /v1/responses/{response_id}` and `previous_response_id` require a
  configured response-store backend.
- Responses tool definitions, tool-call outputs, and non-text output items are
  not supported yet.
- The server continues to expose native oLLM-only runtime planning, prompt, and
  session endpoints beside the compatibility layer.

## Semantics

- The server bind is local-only by default.
- OpenAI-compatible streaming responses use `text/event-stream` chat-completion
  chunks plus a final `data: [DONE]` line.
- Responses streaming uses typed SSE events such as `response.created`,
  `response.output_item.added`, `response.content_part.added`,
  `response.output_text.delta`, `response.output_text.done`,
  `response.output_item.done`, and `response.completed`.
- Native streaming responses still use oLLM's SSE event family.
- Server-side sessions are in-memory only.
- Responses storage is disabled by default; enable a response-store backend when
  you want retrieval or `previous_response_id` chaining.
- The HTTP transport reuses the same `ApplicationService` runtime planning and
  prompt execution logic as the CLI.

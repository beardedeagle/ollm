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

## OpenAI-compatible routes

- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses/{response_id}`
- `DELETE /v1/responses/{response_id}`

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

## Behavior notes

- The bind is local-only by default.
- The OpenAI-compatible surface currently covers model discovery, text chat
  completions, and text responses.
- Chat-completions requests currently support plain string content and structured
  text-part arrays only.
- Responses requests support plain string input, message arrays with text/image/audio
  and file-reference content parts, `function_call_output` tool-result items, and custom
  `type=function` tools with `tool_choice`.
- OpenAI-compatible chat streaming uses `text/event-stream` with chat-completion
  chunks and a final `data: [DONE]` marker.
- Responses streaming uses typed SSE events such as `response.created`,
  `response.in_progress`,
  `response.output_item.added`, `response.content_part.added`,
  `response.output_text.delta`, `response.output_text.done`,
  `response.content_part.done`,
  `response.function_call_arguments.delta`,
  `response.function_call_arguments.done`,
  `response.output_item.done`, `response.completed`, and `response.failed`.
- Native prompt streaming continues to use the oLLM-specific SSE event shape.
- Server-side sessions are in-memory only in the current slice.
- Responses storage is disabled by default. Configure a response-store backend
  if you want `GET /v1/responses/{response_id}`,
  `DELETE /v1/responses/{response_id}`, or `previous_response_id` chaining.
- Runtime and generation defaults still follow the standard config layering
  contract for native endpoints.

## Example requests

```bash
curl http://127.0.0.1:8000/v1/health

curl http://127.0.0.1:8000/v1/models

curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "llama3-1B-chat",
    "messages": [{"role": "user", "content": "List three planets"}]
  }'

curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "llama3-1B-chat",
    "stream": true,
    "messages": [{"role": "user", "content": "List three planets"}]
  }'

curl -X POST http://127.0.0.1:8000/v1/responses \
  -H "content-type: application/json" \
  -d '{
    "model": "llama3-1B-chat",
    "instructions": "Be brief.",
    "input": "List three planets"
  }'

curl -N -X POST http://127.0.0.1:8000/v1/responses \
  -H "content-type: application/json" \
  -d '{
    "model": "llama3-1B-chat",
    "stream": true,
    "input": "List three planets"
  }'

curl -X POST http://127.0.0.1:8000/v1/responses \
  -H "content-type: application/json" \
  -d '{
    "model": "llama3-1B-chat",
    "input": "What is the weather in Paris?",
    "tools": [{
      "type": "function",
      "name": "get_weather",
      "description": "Look up current weather",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }]
  }'

curl -X POST http://127.0.0.1:8000/v1/plan \
  -H "content-type: application/json" \
  -d '{"runtime":{"model_reference":"llama3-1B-chat"}}'
```

Use `examples/ollm.toml` from the repository root as a starting point when you
want shared CLI and server defaults.

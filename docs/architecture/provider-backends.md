# Provider Backends

Provider-backed execution is handled through backend-specific transports rather than native model loading inside oLLM.

## Current backends

- `ollama`
- `openai-compatible`

## Provider aliases

- `msty:` is expressed through the Ollama transport
- `lmstudio:` is expressed through the OpenAI-compatible transport

## Why this boundary exists

It keeps:

- endpoint probing
- provider discovery
- provider request validation
- provider-specific execution metadata

inside the backend layer instead of leaking those concerns into generic runtime loading.

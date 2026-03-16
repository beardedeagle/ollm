# Providers

Provider-backed execution is a first-class runtime path in oLLM.

## Supported provider transports

### Ollama
- backend id: `ollama`
- default endpoint: `http://127.0.0.1:11434`
- reference form: `ollama:<model>`

### Msty Local AI
- backend transport: `ollama`
- reference form: `msty:<model>`
- requires `--provider-endpoint`
- provider identity remains `msty` in runtime metadata

### LM Studio
- backend transport: `openai-compatible`
- default endpoint: `http://127.0.0.1:1234/v1`
- reference form: `lmstudio:<model>`

### Generic OpenAI-compatible
- backend transport: `openai-compatible`
- reference form: `openai-compatible:<model>`
- requires `--provider-endpoint`

## Discovery

`ollm models list` can probe providers and return discovered model references:

```bash
ollm models list
ollm models list --discover-provider ollama
ollm models list --discover-provider msty --provider-endpoint http://127.0.0.1:11434
ollm models list --discover-provider openai-compatible --provider-endpoint http://127.0.0.1:1234/v1
```

## Provider-specific capability notes

- `ollama:` supports text and, for vision-capable models, image inputs
- `msty:` inherits the current Ollama transport behavior
- `openai-compatible:` supports text and request-side audio inputs
- `lmstudio:` remains text-only in oLLM because the current LM Studio OpenAI-compatible docs do not provide a verified audio input contract for this repo

## Provider constraints

Provider-backed execution rejects:

- `--top-k`
- PEFT adapters
- custom CPU/GPU layer offload controls

Use provider-backed refs when the provider is the runtime boundary. Use local or Hugging Face refs when you want native/generic execution inside oLLM itself.

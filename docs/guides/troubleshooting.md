# Troubleshooting

## Start with `ollm doctor`

```bash
ollm doctor --json
```

Useful variants:

```bash
ollm doctor --imports --json
ollm doctor --runtime --json
ollm doctor --model openai-compatible:local-model --provider-endpoint http://127.0.0.1:1234/v1 --json
```

## Common issues

### `--provider-endpoint must not include credentials`
Put credentials in provider configuration or environment instead of embedding them in the URL.

### `--backend optimized-native cannot be combined with --no-specialization`
`optimized-native` explicitly means you want specialization. Use `transformers-generic` if you want a non-specialized path.

### A Hugging Face repo resolves but will not execute yet
Some Hugging Face refs must be materialized locally before generic capability discovery can inspect them.

### Provider model is discovered but unavailable
The provider can return a model name even when it is not currently executable from the endpoint oLLM is probing. Check:

- service availability
- endpoint URL
- provider-side model presence

### Generic runtime rejects a checkpoint
The generic runtime intentionally refuses unsafe `.bin` or pickle-backed local weights. Use `safetensors` artifacts for the generic local path.

### LM Studio audio is rejected
That is currently intentional in oLLM because the current LM Studio OpenAI-compatible docs do not provide a verified audio-input contract for this repo.

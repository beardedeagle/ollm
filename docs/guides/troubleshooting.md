# Troubleshooting

## Start with `ollm doctor`

```bash
ollm doctor --json
```

Useful variants:

```bash
ollm doctor --imports --json
ollm doctor --runtime --json
```

## Common issues

### `--backend optimized-native cannot be combined with --no-specialization`
`optimized-native` explicitly means you want specialization. Use `transformers-generic` if you want a non-specialized path.

### A Hugging Face repo resolves but will not execute yet
Some Hugging Face refs must be materialized locally before generic capability discovery can inspect them.

### Provider-prefixed refs are rejected
That is intentional. oLLM now executes only locally managed models. Use a built-in alias, a local model directory, or a Hugging Face repository reference.

### Generic runtime rejects a checkpoint
The generic runtime intentionally refuses unsafe `.bin` or pickle-backed local weights. Use `safetensors` artifacts for the generic local path.

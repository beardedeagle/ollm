# Model References

A model reference is the user-facing input that tells oLLM what to resolve and run.

## Supported forms

### Built-in aliases
Examples:

- `llama3-1B-chat`
- `llama3-3B-chat`
- `llama3-8B-chat`
- `gpt-oss-20B`
- `qwen3-next-80B`
- `gemma3-12B`
- `voxtral-small-24B`

These aliases are metadata entries, not admission gates for the rest of the runtime.

### Hugging Face repo IDs
Examples:

- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-3-12b-it`

These resolve to local materialization paths under `--models-dir`.

### Hugging Face revisions
Examples:

```text
Qwen/Qwen2.5-7B-Instruct@main
Qwen/Qwen2.5-7B-Instruct@refs/pr/12
```

Revisions are preserved in the materialization path.

### Local paths
Examples:

```text
./models/custom-model
~/models/custom-model
/absolute/path/to/model
```

### Provider references
Examples:

- `ollama:llama3.2`
- `msty:llama3.2`
- `lmstudio:qwen2.5`
- `openai-compatible:local-model`

## Resolution outcomes

oLLM resolves a model reference into one of these source kinds:

- `builtin`
- `hugging-face`
- `local-path`
- `provider`
- `opaque`

Opaque references are parsed, but not currently resolvable to a runnable runtime or materialization path.

## Availability terminology

For local and provider references, oLLM intentionally uses different terms:

- local refs: `materialized` / `not-materialized`
- provider refs: `available` / `unavailable`

This keeps provider executability separate from local on-disk presence.

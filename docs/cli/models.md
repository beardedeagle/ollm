# `ollm models`

`ollm models` covers discovery, inspection, local materialization, and path reporting.

## Subcommands

- `list`
- `info`
- `download`
- `path`

## `list`

Discovery view over:

- built-in aliases
- local materialized models
- provider-discovered entries

Examples:

```bash
ollm models list
ollm models list --installed
ollm models list --discover-provider openai-compatible --provider-endpoint http://127.0.0.1:1234/v1
```

## `info`

Inspect a single model reference and its runtime-plan truth:

```bash
ollm models info llama3-1B-chat --json
ollm models info openai-compatible:local-model --provider-endpoint http://127.0.0.1:1234/v1 --json
```

## `download`

Materialize a downloadable model reference locally:

```bash
ollm models download llama3-3B-chat
ollm models download Qwen/Qwen2.5-7B-Instruct
```

## `path`

Print the local materialization path for a reference that resolves to local storage.

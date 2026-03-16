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

Examples:

```bash
ollm models list
ollm models list --installed
```

## `info`

Inspect a single model reference and its runtime-plan truth:

```bash
ollm models info llama3-1B-chat --json
ollm models info Qwen/Qwen2.5-7B-Instruct --json
```

## `download`

Materialize a downloadable model reference locally. The downloader stores only the
runtime-critical artifacts that `ollm` needs to inspect, plan, and execute the
model locally rather than a full Hugging Face repository snapshot:

```bash
ollm models download llama3-3B-chat
ollm models download Qwen/Qwen2.5-7B-Instruct
```

## `path`

Print the local materialization path for a reference that resolves to local storage.

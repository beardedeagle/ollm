# `ollm doctor`

`ollm doctor` inspects imports, runtime availability, paths, downloads, and model readiness.

## Modes

- `--imports`
- `--runtime`
- `--paths`
- `--download`
- `--json`
- `--plan-json`

## Why use it

- verify optional dependency availability
- verify runtime/device visibility
- inspect provider-backed refs without attempting a full generation
- inspect runtime-plan truth for a specific model reference

## Examples

```bash
ollm doctor --json
ollm doctor --imports --json
ollm doctor --model ollama:llava --json
ollm doctor --model openai-compatible:local-model --provider-endpoint http://127.0.0.1:1234/v1 --json
```

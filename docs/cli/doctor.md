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
- inspect runtime-plan truth for a specific model reference

## Examples

```bash
ollm doctor --json
ollm doctor --imports --json
ollm doctor --model llama3-1B-chat --json
```

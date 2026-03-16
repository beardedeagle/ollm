# Model Discovery

`ollm models list` is a discovery view, not a fixed allowlist view.

## Discovery sources

- `built-in` — shipped aliases
- `discovered-local` — local materialized models found under `--models-dir`
- `discovered-provider` — entries returned by provider discovery

## Availability terms

For local entries:
- `materialized`
- `not-materialized`

For provider entries:
- `available`
- `unavailable`

This keeps on-disk presence separate from provider reachability.

## Installed filter

```bash
ollm models list --installed
```

`--installed` filters to **materialized local entries only**. Provider references are never treated as installed.

## Examples

```bash
ollm models list --json
ollm models list --discover-provider ollama --json
ollm models list --discover-provider openai-compatible --provider-endpoint http://127.0.0.1:1234/v1 --json
```

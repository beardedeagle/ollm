# Model Discovery

`ollm models list` is a discovery view, not a fixed allowlist view.

## Discovery sources

- `built-in` — shipped aliases
- `discovered-local` — local materialized models found under `--models-dir`

## Availability terms

- `materialized`
- `not-materialized`

## Installed filter

```bash
ollm models list --installed
```

`--installed` filters to materialized local entries only.

## Examples

```bash
ollm models list --json
ollm models list --installed
```

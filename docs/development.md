# Development

## Environment setup

```bash
uv sync
uv sync --extra adapters --extra export
uv sync --extra audio
uv sync --group dev
uv sync --group docs
```

## Verification commands

```bash
uv run python -m compileall src tests
uv run pytest
uv build
uv run python -m pip_audit
```

For runtime-heavy changes, run one targeted script or CLI smoke for the path you touched.

## Documentation site

The project documentation is built with MkDocs Material.

Build locally:

```bash
uv run --group docs mkdocs build --strict
```

Preview locally:

```bash
uv run --group docs mkdocs serve
```

## Examples and manual scripts

- end-user smoke examples live in `examples/`
- export and manual validation scripts live in `scripts/`
- large text fixtures live in `samples/`

Adjust any hard-coded local model paths in manual scripts before running them.

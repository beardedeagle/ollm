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
uv run ruff format src tests examples scripts
uv run ruff check src tests examples scripts
uv run python scripts/check_python_standards.py
uv run python -m compileall src tests
uv run ty check src tests
uv run pytest -q
uv build
uv run python -m pip_audit
uv run --group docs mkdocs build --strict
```

For runtime-heavy changes, run one targeted script or CLI smoke for the path you touched.

## Engineering standards

The canonical standards baseline lives in [Python Standards](guides/python-standards.md).

This repo is currently being treated as greenfield:

- no legacy preservation
- no backwards-compatibility ballast
- delete bad shapes instead of keeping them alive

The standards checker and audit surfaces are:

- `uv run python scripts/check_python_standards.py`
- [Python Standards Audit](guides/python-standards-audit.md)

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

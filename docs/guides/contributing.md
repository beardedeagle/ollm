# Contributing Guide

## Local development workflow

Recommended setup:

```bash
uv sync --group dev --group docs
```

## Verification expectations

Before sending a documentation or code change, run the relevant gates:

```bash
uv run python -m compileall src tests examples scripts
uv run pytest -q
uv build
uv run python -m pip_audit
uv run --group docs mkdocs build --strict
```

For docs-only changes, still verify the docs build and run at least the tests affected by the changed examples or commands.

## Documentation expectations

This project treats docs as part of the product surface:

- keep README and `docs/` aligned
- keep CLI/help text and docs terminology aligned
- avoid internal implementation-plan language in user-facing docs
- prefer truthful statements over broad claims that cannot be proved on the current host or upstream provider contract

## Testing style

- prefer fakes and tiny real fixtures over mocks
- keep provider tests isolated and explicit
- use targeted CLI or script smokes when touching runtime-heavy flows

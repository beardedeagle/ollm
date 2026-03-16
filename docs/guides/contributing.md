# Contributing Guide

## Local development workflow

Recommended setup:

```bash
uv sync --group dev --group docs
```

## Verification expectations

Before sending a documentation or code change, run the relevant gates:

```bash
uv run ruff format src tests examples scripts
uv run ruff check src tests examples scripts
uv run python scripts/check_python_standards.py
uv run python -m compileall src tests examples scripts
uv run ty check src tests
uv run pytest -q
uv build
uv run python -m pip_audit
uv run --group docs mkdocs build --strict
```

For docs-only changes, still verify the docs build and run at least the tests affected by the changed examples or commands.

## Documentation expectations

This project treats docs as part of the product surface:

- keep README and `docs/` aligned
- keep the canonical standards contract in [Python Standards](python-standards.md)
- keep CLI/help text and docs terminology aligned
- avoid internal implementation-plan language in user-facing docs
- prefer truthful statements over broad claims that cannot be proved on the current host

## Standards expectations

`ollm` is being treated as a greenfield codebase:

- no legacy-only code paths
- no compatibility scaffolding
- no preserving bad structures because they already exist

The mechanically enforced rules live in `scripts/check_python_standards.py`, and the current
repo-wide remediation matrix lives in [Python Standards Audit](python-standards-audit.md).

## Testing style

- prefer fakes and tiny real fixtures over mocks
- use targeted CLI or script smokes when touching runtime-heavy flows

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
uv run ty check src tests scripts
uv run python -m compileall src tests scripts
uv run pytest -q
uv build
uv run python -m pip_audit
uv run --group docs mkdocs build --strict
```

If the change touches runtime planning, loading, generation, prompt/chat flows,
server prompt execution, or a user-facing command recommendation, also run:

```bash
uv run python scripts/runtime_smoke.py \
  --model llama3-1B-chat \
  --output .omx/logs/runtime-smoke.json
```

The manual regression helpers in `scripts/test.py`, `scripts/test_autoinference.py`,
and `scripts/full_tests.py` take explicit CLI arguments for model roots,
cache directories, and prompt/sample inputs.

Use the lane you actually changed whenever possible. If you recommend a
specific command to a user, validate that exact lane before push or say which
stricter equivalent you validated.

For docs-only changes, verify the docs build and run at least the tests affected by the changed examples or commands.

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

The mechanically enforced rules live in `scripts/check_python_standards.py`, and the
repo-wide remediation matrix lives in [Python Standards Audit](python-standards-audit.md).

## Testing style

- prefer fakes and tiny real fixtures over mocks
- use `scripts/runtime_smoke.py` for reusable runtime smokes instead of ad hoc one-off commands
- if you changed interactive shell rendering itself, add a manual `uv run ollm chat ...` check in addition to the reusable smoke

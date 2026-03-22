## Summary

- describe the behavior change
- describe any runtime or user-facing command changes

## Verification

- [ ] `uv run ruff format src tests examples scripts`
- [ ] `uv run ruff check src tests examples scripts`
- [ ] `uv run python scripts/check_python_standards.py`
- [ ] `uv run python -m compileall src tests examples scripts`
- [ ] `uv run ty check src tests scripts`
- [ ] `uv run pytest -q`
- [ ] `uv build`
- [ ] `uv run python -m pip_audit`
- [ ] `uv run --group docs mkdocs build --strict`
- [ ] `git diff --check`

## Runtime Smoke

- [ ] Not runtime-heavy, or no prompt/chat/server execution path changed
- [ ] Runtime-heavy change: local smoke run completed with exact command(s) and artifact path below

Command(s):

Artifact path(s):

## User-Facing Lanes

- [ ] No user-facing runtime command or lane recommendation changed
- [ ] Changed or recommended user-facing runtime lane was validated exactly, or a stricter equivalent is listed below

Exact lane or stricter equivalent:

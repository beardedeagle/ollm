# Python Standards

This project treats Python engineering standards as a prerequisite to further feature work.

These rules are not advisory. For `ollm`, they are the working contract for new code, refactors,
runtime changes, and contributor-facing documentation.

## Process rules

- Provide full 5W1H before any change and wait for approval unless the user explicitly waives it.
- Use tree-sitter and code-context-provider tooling when modifying modules, alongside the other
  available repository tools.
- Prefer `eza` over `ls`, `bat` over `cat` or `nl`, `rg` over `grep` or `find`, and pair
  `ast-grep` with structural code-intel tooling.
- Work at a distinguished engineer / software architect bar.
- Follow the repo validation order exactly:
  1. `uv run ruff format src tests examples scripts`
  2. `uv run ruff check src tests examples scripts`
  3. `uv run python scripts/check_python_standards.py`
  4. `uv run ty check src tests scripts`
  5. `uv run python -m compileall src tests scripts`
  6. `uv run pytest -q`
  7. `uv build`
  8. `uv run python -m pip_audit`
  9. `uv run --group docs mkdocs build --strict`
  10. `git diff --check`
- For runtime-heavy changes, structural gates are not enough. Run a real smoke over
  the shared generation stack:
  `uv run python scripts/runtime_smoke.py --model <model-reference> --output .omx/logs/runtime-smoke.json`
- If you recommend a specific user-facing runtime lane or command, validate that exact lane before push, or call out the stricter equivalent you actually validated.
- CI proves structural health only. It does not prove live model semantics, response quality, or the correctness of a newly recommended runtime lane.

## Code rules

- Apply SOLID principles.
- Write modern, idiomatic Python with self-documenting names and modular structure.
- Keep file size within a 500 line soft limit and a 1000 line hard limit.
- All I/O must be async. Pure logic may remain sync.
- Code must be thread safe.
- Code must be strongly and strictly typed.
- Code must be secure, defensive, correct, performant, and warning-free under the validation
  stack.
- Use Google-style docstrings that are compatible with MkDocs and include concrete parameter and
  return types.
- Use absolute imports only.
- Keep imports at the top of the module.
- Use `|` unions instead of `Optional[...]` or `Union[...]`.

## Docstring format

Public modules, classes, and user-facing functions should use Google-style
docstrings with explicit types inside the sections that MkDocs renders:

```python
def resolve(raw_reference: str, models_dir: Path) -> ResolvedModel:
    """Resolve a user-facing model reference.

    Args:
        raw_reference (str): User-supplied model reference such as an alias,
            Hugging Face ID, or local path.
        models_dir (Path): Local models root used for implicit path resolution.

    Returns:
        ResolvedModel: Normalized model metadata used by planning and loading.

    Raises:
        ValueError: Raised when the input is structurally invalid for this API.
    """
```

Keep docstrings concise, but do include:

- `Args:` with names, types, and purpose.
- `Returns:` with the concrete return type and meaning.
- `Raises:` for user-visible validation or loading failures.
- Omit empty sections rather than adding placeholders.

## Forbidden patterns

- No `Any`.
- No forward references.
- No relative imports.
- No TODO, FIXME, or XXX markers.
- No compatibility code or legacy preservation.
- No mock-driven fake success.
- No low-value tests written only for coverage.

## Project-specific policy

`ollm` is being treated as a greenfield project. That means:

- delete bad shapes instead of preserving them for compatibility
- do not add backwards-compatibility ballast
- do not keep legacy-only code paths alive once a better shape replaces them

When working in existing code, keep diffs tight and surgical, but do not preserve a bad structure
just because it already exists.

## Enforcement and audit

- `uv run python scripts/check_python_standards.py` enforces the mechanically checkable rules.
- `docs/guides/python-standards-audit.md` is the remediation matrix for repo-wide
  violations that require follow-on work.
- `docs/development.md` and `docs/guides/contributing.md` summarize the contributor workflow, but
  this page is the canonical standards source.

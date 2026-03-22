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

For runtime-heavy changes, also run the reusable smoke harness on the lane you
actually changed:

```bash
uv run python scripts/runtime_smoke.py \
  --model llama3-1B-chat \
  --output .omx/logs/runtime-smoke.json
```

If you recommend a specific command or runtime lane to a user, validate that
exact lane before push, or explicitly say which stricter equivalent you used.

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

## Contributor ADRs

Longer-horizon contributor-facing design decisions live under `docs/development/`.

Current ADR pages:

- [Tiered KV Cache ADR](development/tiered-kv-cache-adr.md)
- [Weight Transport ADR](development/weight-transport-adr.md)

## Examples and manual scripts

- end-user smoke examples live in `examples/`
- export and manual validation scripts live in `scripts/`
- large text fixtures live in `samples/`
- `scripts/runtime_smoke.py` is the reusable pre-push runtime validation path for
  prompt and chat-session semantics

Some older manual scripts still contain machine-specific paths and should be
treated as ad hoc local experiments rather than the canonical validation path.

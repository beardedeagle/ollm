# Installation

## Requirements

- Python **3.12+**
- `uv` is the recommended environment manager
- a supported runtime environment for your chosen backend:
  - local Transformers-backed execution
  - optimized-native execution for built-in aliases

## Recommended setup with `uv`

```bash
git clone https://github.com/Mega4alik/ollm.git
cd ollm
uv sync
```

Optional groups and extras:

```bash
uv sync --extra adapters   # AutoInference with PEFT adapters
uv sync --extra audio      # voxtral audio example
uv sync --extra cuda       # flash-attn + triton acceleration
uv sync --extra export     # export scripts
uv sync --group dev        # pytest and contributor tooling
uv sync --group docs       # MkDocs Material docs site + API docs
```

## Editable install with `pip`

```bash
pip install --no-build-isolation -e .
pip install --no-build-isolation -e ".[adapters]"
```

`kvikio` remains a manual install because the package name depends on your CUDA version.

## Verify the install

```bash
uv run python -m compileall src tests
uv run pytest -q
uv run ollm doctor --imports --json --no-color
uv run ollm prompt --help
```

# Terminal Interface

## Core commands

```bash
ollm                         # interactive terminal chat
ollm chat                    # explicit alias for interactive chat
ollm prompt "List planets"   # one-shot prompt
ollm doctor --json           # environment and runtime diagnostics
ollm models list             # built-in and discovered local model references
```

Use `ollm` or `ollm chat` only from an interactive terminal. For scripts, pipes, and automation use `ollm prompt`.

## Runtime controls

- `--backend` forces a valid local backend for the resolved model reference
- `--no-specialization` disables optimized-native specialization and prefers the generic path when available
- `--plan-json` prints the resolved runtime plan and exits without running generation

`ollm prompt`, `ollm chat`, `ollm doctor`, and `ollm models info` all honor these controls.

## Model references

`--model` accepts opaque model references. Supported forms include:

- built-in aliases such as `llama3-1B-chat` and `gemma3-12B`
- Hugging Face repo IDs such as `Qwen/Qwen2.5-7B-Instruct`
- local model directories

Provider-prefixed references are rejected so execution stays inside oLLM's local runtime boundary.

## Support levels

oLLM reports one of three active support levels for a resolved model reference:

- `optimized` — a native specialization provider can run the reference
- `generic` — the Transformers-backed generic runtime can run the reference
- `unsupported` — the reference resolves, but the current runtime cannot execute it

## Discovery and availability terms

`ollm models list` is a discovery view. It combines:

- `built-in` entries shipped by oLLM
- `discovered-local` entries found under `--models-dir`

Availability for local references uses:

- `materialized`
- `not-materialized`

`ollm models list --installed` filters to materialized local entries only.

## Generic and optimized execution

The generic execution path covers compatible local or materialized Transformers-backed:

- causal language models
- encoder-decoder text generation models
- image-text conditional generation models that expose a processor-backed `vision_config`

When the resolved model matches a native family specialization (`llama`, `gemma3`, `qwen3-next`, `gpt-oss`, or `voxtral`), oLLM records and selects an optimized-native specialization provider through the runtime plan instead of hard-coding model-family branches in `Inference.load_model()`.

## Specialization visibility and fallback

Planning-only surfaces such as `ollm doctor` and `ollm models info --json` expose the resolved backend, specialization state, and planned specialization pass ids without loading a runtime.

Execution surfaces follow the finalized runtime plan. Prompt response metadata includes:

- execution backend
- specialization state
- execution device type for optimized-native runs
- specialization device profile for optimized-native runs
- applied specialization pass ids
- any recorded fallback reason

If an optimized specialization cannot satisfy its planned pass contract and a compatible generic path exists, oLLM falls back safely to `transformers-generic` instead of pretending the optimized path succeeded.

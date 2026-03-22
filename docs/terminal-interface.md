# Terminal Interface

## Core commands

```bash
ollm                         # interactive terminal chat
ollm chat                    # explicit alias for interactive chat
ollm prompt "List planets"   # one-shot prompt
ollm doctor --json           # environment and runtime diagnostics
ollm models list             # built-in and discovered local model references
ollm serve                   # local-only REST API server
```

Use `ollm` or `ollm chat` only from an interactive terminal. For scripts, pipes, and automation use `ollm prompt`.

## Runtime controls

- `--backend` forces a valid local backend for the resolved model reference
- `--no-specialization` disables optimized-native specialization and prefers the generic path when available
- `--plan-json` prints the resolved runtime plan and exits without running generation

`ollm prompt`, `ollm chat`, `ollm doctor`, and `ollm models info` all honor these controls.

## Configuration sources

oLLM now resolves runtime, generation, and server defaults through an explicit
layered settings contract:

1. CLI flags
2. `OLLM_*` environment variables
3. TOML config file values
4. built-in defaults

By default, oLLM checks `./ollm.toml` when it is present. You can point to a
different config file with `OLLM_CONFIG_FILE=/path/to/ollm.toml`.

Nested environment variables use a double-underscore separator:

- `OLLM_RUNTIME__MODEL_REFERENCE`
- `OLLM_RUNTIME__MODELS_DIR`
- `OLLM_RUNTIME__DEVICE`
- `OLLM_GENERATION__MAX_NEW_TOKENS`
- `OLLM_GENERATION__STREAM`
- `OLLM_SERVER__PORT`

Example config file:

```toml
[runtime]
model_reference = "llama3-8B-chat"
models_dir = "models"
device = "cpu"

[generation]
max_new_tokens = 256
temperature = 0.0
stream = true

[server]
host = "127.0.0.1"
port = 8000
```

The current settings surface covers runtime defaults, generation defaults, and
future server defaults. Prompt-specific values outside that schema, such as the
system prompt text, still remain explicit CLI options today.

## Local server

Install the optional server transport stack first:

```bash
uv sync --extra server
```

Then start the local-only server:

```bash
ollm serve
```

`ollm serve` resolves `host`, `port`, `reload`, and `log_level` through the
same settings-precedence contract as the rest of the CLI. The default bind is
`127.0.0.1`, and the server also publishes:

- `/openapi.json`
- `/docs`
- `/redoc`

The current REST surface is:

- `GET /v1/health`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses/{response_id}`
- `GET /v1/ollm/models`
- `GET /v1/ollm/models/{model_reference}`
- `POST /v1/plan`
- `POST /v1/prompt`
- `POST /v1/prompt/stream`
- `POST /v1/sessions`
- `GET /v1/sessions/{session_id}`
- `POST /v1/sessions/{session_id}/prompt`
- `POST /v1/sessions/{session_id}/prompt/stream`

The streaming transport is SSE-based and the current server-side sessions are
in-memory only. See [Local Server API](guides/local-server.md) for the complete
HTTP surface and [CLI `ollm serve`](cli/server.md) for command-specific usage.

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

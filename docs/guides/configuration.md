# Configuration

## Precedence

oLLM resolves runtime, generation, and server defaults through the same
explicit precedence chain:

1. CLI flags
2. `OLLM_*` environment variables
3. TOML config file values
4. built-in defaults

By default, oLLM loads `./ollm.toml` when it is present. You can point to a
different config file with `OLLM_CONFIG_FILE=/path/to/ollm.toml`.

An example config file that covers runtime, generation, and server settings
lives at `examples/ollm.toml` in the repository root.

## Runtime configuration

The CLI and the Python library both build on `RuntimeConfig` and `GenerationConfig`.

Key runtime configuration fields:

- `model_reference` — the model to resolve
- `models_dir` — root for local materialized models
- `device` — torch device such as `cpu` or `cuda:0`
- `backend` — optional backend override
- `multimodal` — enable multimodal planning when non-text inputs are expected
- `use_specialization` — whether optimized-native specialization is allowed
- `cache_dir` / `use_cache` — KV cache controls
- `kv_cache_strategy` — explicit KV cache strategy selection (`resident`, `chunked`, `paged`, `streamed-segmented`, `log-structured-journal`, `sliding-window-ring-buffer`, `quantized-cold-tier`, or `tiered-write-back`)
- `kv_cache_window_tokens` — bounded recent-context token budget for `sliding-window-ring-buffer`; the field is invalid for full-history strategies
- `kv_cache_lifecycle` — whether KV artifacts are `runtime-scoped` or explicitly `persistent`; `resident` requires `runtime-scoped`
- `kv_cache_adaptation_mode` — whether adaptation telemetry is `disabled`, `observe-only`, or `automatic` (live switching is still not enabled)
- `offload_cpu_layers` / `offload_gpu_layers` — native offload controls when supported

Generation configuration fields:

- `max_new_tokens`
- `temperature`
- `top_p`
- `top_k`
- `seed`
- `stream`

Server configuration fields:

- `host`
- `port`
- `reload`
- `log_level`

See [API Reference: Runtime Config](../api/config.md).

## Environment variables

Nested configuration keys use a double-underscore separator:

- `OLLM_RUNTIME__MODEL_REFERENCE`
- `OLLM_RUNTIME__MODELS_DIR`
- `OLLM_RUNTIME__DEVICE`
- `OLLM_RUNTIME__KV_CACHE_STRATEGY`
- `OLLM_RUNTIME__KV_CACHE_WINDOW_TOKENS`
- `OLLM_RUNTIME__KV_CACHE_LIFECYCLE`
- `OLLM_RUNTIME__KV_CACHE_ADAPTATION_MODE`
- `OLLM_GENERATION__MAX_NEW_TOKENS`
- `OLLM_GENERATION__STREAM`
- `OLLM_SERVER__HOST`
- `OLLM_SERVER__PORT`
- `OLLM_SERVER__LOG_LEVEL`

## Backend override

`--backend` lets you force one of:

- `optimized-native`
- `transformers-generic`

The override is validated against the resolved model reference. If the backend is incompatible, oLLM fails with a structured error instead of silently switching to something else.

## Specialization toggle

`--no-specialization` disables optimized-native specialization selection and forces the generic path when one exists.

Important constraint:

- `--backend optimized-native` cannot be combined with `--no-specialization`

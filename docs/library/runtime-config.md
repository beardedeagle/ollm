# Runtime Configuration

The library and CLI share the same runtime configuration model.

## `RuntimeConfig`

Use `RuntimeConfig` to describe:

- the model reference
- local materialization root
- device
- backend override
- multimodal enablement
- specialization enablement
- cache strategy, cache lifecycle, adaptation mode, cache root, and offload behavior

The current KV scaffolding now distinguishes:

- `kv_cache_strategy` — the current preset such as `chunked` or `tiered-write-back`
- `kv_cache_lifecycle` — `runtime-scoped` today, with future `persistent` support planned explicitly
- `kv_cache_adaptation_mode` — scaffolded as `disabled`, `observe-only`, or `automatic`

## `GenerationConfig`

Use `GenerationConfig` to describe:

- token limits
- sampling controls
- seeding
- streaming

See:

- [Runtime Config API](../api/config.md)

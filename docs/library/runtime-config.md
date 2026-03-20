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

- `kv_cache_strategy` — the current preset such as `chunked`, `quantized-cold-tier`, or `tiered-write-back`
- `kv_cache_lifecycle` — `runtime-scoped` or explicit `persistent` reuse semantics
- `kv_cache_adaptation_mode` — `disabled`, `observe-only`, or `automatic`; observe-only recommendation rules exist, but live switching is still disabled

## `GenerationConfig`

Use `GenerationConfig` to describe:

- token limits
- sampling controls
- seeding
- streaming

See:

- [Runtime Config API](../api/config.md)

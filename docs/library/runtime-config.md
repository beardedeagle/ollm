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
- sliding-window token budget when bounded-history KV is selected

The current KV scaffolding now distinguishes:

- `kv_cache_strategy` — the current preset such as `resident`, `chunked`, `paged`, `streamed-segmented`, `log-structured-journal`, `sliding-window-ring-buffer`, `quantized-cold-tier`, or `tiered-write-back`
- `kv_cache_window_tokens` — bounded recent-context token budget for `sliding-window-ring-buffer`; omitted for full-history strategies
- `kv_cache_lifecycle` — `runtime-scoped` or explicit `persistent` reuse semantics; `resident` requires `runtime-scoped`
- `kv_cache_adaptation_mode` — `disabled`, `observe-only`, or `automatic`; observe-only recommendation rules exist, but live switching is still disabled
- `offload_cpu_layers` — requested CPU offload layer budget
- `offload_cpu_policy` — CPU offload placement policy (`auto`, `prefix`, `suffix`, or `middle-band`)
- `offload_gpu_layers` — requested GPU offload layer budget

Current offload truth:

- `offload_cpu_layers` requires an accelerator runtime device
- `offload_cpu_layers` and `offload_gpu_layers` cannot be combined in the current implementation

## `GenerationConfig`

Use `GenerationConfig` to describe:

- token limits
- sampling controls
- seeding
- streaming

See:

- [Runtime Config API](../api/config.md)

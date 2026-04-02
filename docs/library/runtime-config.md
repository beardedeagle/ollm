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

The KV scaffolding distinguishes:

- `strategy_selector_profile` — the deterministic selector profile (`balanced`, `latency`, `capacity`, or `bounded-window`)
- `kv_cache_strategy` — optional explicit strategy override; when omitted, the selector chooses a concrete preset
- `kv_cache_window_tokens` — bounded recent-context token budget for `sliding-window-ring-buffer`; omitted for full-history strategies
- `dense_projection_chunk_rows` — optional explicit row budget for dense optimized-native MLP chunking; when omitted, the dense Llama, Gemma3, and Voxtral paths keep the `16384`-row ceiling but derive smaller chunks only when accelerator headroom is tight
- `kv_cache_lifecycle` — `runtime-scoped` or explicit `persistent` reuse semantics; `resident` requires `runtime-scoped`
- `kv_cache_adaptation_mode` — `disabled`, `observe-only`, or `automatic`; observe-only recommendation rules exist, but live switching is disabled
- `offload_cpu_layers` — requested CPU offload layer budget
- `offload_cpu_policy` — CPU offload placement policy (`auto`, `prefix`, `suffix`, or `middle-band`)
- `offload_gpu_layers` — requested GPU offload layer budget

Offload constraints:

- `offload_cpu_layers` requires an accelerator runtime device
- `offload_cpu_layers` and `offload_gpu_layers` cannot be combined in the current implementation

Selector behavior:

- selector-default candidates are `paged`, `resident`, and `quantized-cold-tier`
- `sliding-window-ring-buffer` remains explicit bounded-history opt-in only
- `streamed-segmented`, `log-structured-journal`, and `tiered-write-back` remain explicit override choices, not selector defaults

## `GenerationConfig`

Use `GenerationConfig` to describe:

- token limits
- sampling controls
- seeding
- streaming

See:

- [Runtime Config API](../api/config.md)

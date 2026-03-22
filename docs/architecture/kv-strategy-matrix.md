# KV Strategy Matrix

oLLM's current KV cache presets are still selected as single strings such as
`resident`, `chunked`, `paged`, `streamed-segmented`, `log-structured-journal`,
`sliding-window-ring-buffer`, `quantized-cold-tier`, and
`tiered-write-back`.

That is still the public control surface today, but the system is now being
scaffolded around a more explicit internal matrix so future strategies do not
turn into one giant preset enum with hidden semantics.

## Current matrix axes

The current scaffold separates these concerns:

- persistence format
- residency mode
- window policy
- cold-tier encoding
- cache lifecycle
- adaptation mode

The first four describe the current preset itself. The last two describe how
that preset is owned and how telemetry may influence future runtime choices.

## Presets versus axes

Current presets are now understood as bundles of axis values:

| Preset | Persistence Format | Residency Mode | Window Policy | Cold-Tier Encoding |
| --- | --- | --- | --- | --- |
| `resident` | `resident-only` | `fully-resident` | `full-history` | `full-precision` |
| `chunked` | `chunked-manifest` | `buffered-tail` | `full-history` | `full-precision` |
| `paged` | `paged-manifest` | `buffered-tail` | `full-history` | `full-precision` |
| `streamed-segmented` | `streamed-segmented` | `buffered-tail` | `full-history` | `full-precision` |
| `log-structured-journal` | `log-structured-journal` | `buffered-tail` | `full-history` | `full-precision` |
| `sliding-window-ring-buffer` | `sliding-window-ring-buffer` | `buffered-tail` | `sliding-window` | `full-precision` |
| `quantized-cold-tier` | `log-structured-journal` | `buffered-tail` | `full-history` | `quantized` |
| `tiered-write-back` | `log-structured-journal` | `tiered-write-back` | `full-history` | `full-precision` |

The current `tiered-write-back` preset should not be confused with a future
broader multi-tier GPU/CPU/SSD architecture.

This does not erase the semantic differences between presets. For example,
`sliding-window-ring-buffer` deliberately preserves only a bounded recent
history under a `drop-oldest` eviction policy; it is not a transparent
substitute for the full-history strategies. The matrix still keeps the shape
explicit so future presets and richer bounded window variants can compose
cleanly.

## Cache lifecycle

Cache lifecycle is intentionally treated as a separate axis:

- `runtime-scoped`
- `persistent`

Current default behavior is still `runtime-scoped`, but `persistent` is now an
explicit implemented mode.

That means oLLM can now distinguish between:

- within-runtime reuse
- explicit persistent reuse across later runs under a lifecycle-aware,
  model/backend-scoped cache root

This distinction matters because cross-run persistence adds separate concerns:

- cache identity and invalidation
- schema compatibility
- partial-write recovery
- retention / GC
- multi-process coordination
- benchmark truth for cold versus warm starts

## Adaptation mode

Adaptation mode is also scaffolded explicitly:

- `disabled`
- `observe-only`
- `automatic`

Current behavior now supports observe-only recommendation rules. The runtime can
emit a truthful recommendation from current KV state, but it does not yet
switch KV strategies live.

The intended progression is:

1. report signals
2. make observe-only recommendations
3. prove the recommendation loop against benchmarks and repeated-session runs
4. only then enable real automatic policy changes

## Strategy selector

The repo now also has a deterministic pre-run strategy selector above the
explicit presets.

Current selector profiles are:

- `balanced`
- `latency`
- `capacity`
- `bounded-window`

Current selector-default candidates are intentionally conservative:

- `paged`
- `resident`
- `quantized-cold-tier`

The following presets remain explicit opt-in or pinned overrides:

- `sliding-window-ring-buffer`
- `streamed-segmented`
- `log-structured-journal`
- `tiered-write-back`

This selector is separate from `kv_cache_adaptation_mode`.

- selector: chooses the initial strategy before runtime execution
- adaptation mode: emits observe-only recommendations or future live changes
  from current KV state

## Resident-state observability

The in-process resident KV snapshot is now treated as a first-class observable
part of cache state, not just an implementation detail.

Current reporting now has room to distinguish:

- persisted state
- resident in-process state
- hot/pending tails
- bounded-window limits and eviction totals
- cold-tier representation when the persisted encoding is quantized

That matters because resident reuse can improve request behavior without showing
up as persisted on-disk KV activity.

## Near-term intent

This scaffold is meant to support the next wave of work:

- paged persistence
- quantized cold tiers
- a future page-aware GPU / CPU / SSD tiered architecture
- observe-only adaptation recommendations
- stronger persistent lifecycle reuse and retention policy

The important rule is: add new behavior by filling in these axes, not by
letting one preset label absorb unrelated semantics forever.

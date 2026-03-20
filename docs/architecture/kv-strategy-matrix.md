# KV Strategy Matrix

oLLM's current disk-KV presets are still selected as single strings such as
`chunked`, `streamed-segmented`, `log-structured-journal`, and
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
| `chunked` | `chunked-manifest` | `buffered-tail` | `full-history` | `full-precision` |
| `streamed-segmented` | `streamed-segmented` | `buffered-tail` | `full-history` | `full-precision` |
| `log-structured-journal` | `log-structured-journal` | `buffered-tail` | `full-history` | `full-precision` |
| `tiered-write-back` | `log-structured-journal` | `tiered-write-back` | `full-history` | `full-precision` |

This does not change runtime behavior yet. It only makes the shape explicit so
future presets such as `paged`, `resident`, `quantized cold tier`, or
`sliding-window` can compose cleanly.

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

## Resident-state observability

The in-process resident KV snapshot is now treated as a first-class observable
part of cache state, not just an implementation detail.

Current reporting now has room to distinguish:

- persisted state
- resident in-process state
- hot/pending tails

That matters because resident reuse can improve request behavior without showing
up as persisted on-disk KV activity.

## Near-term intent

This scaffold is meant to support the next wave of work:

- paged persistence
- resident baseline mode
- quantized cold tiers
- sliding-window / ring-buffer mode
- observe-only adaptation recommendations
- later persistent lifecycle support

The important rule is: add new behavior by filling in these axes, not by
letting one preset label absorb unrelated semantics forever.

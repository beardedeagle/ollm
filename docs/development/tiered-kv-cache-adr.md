# Tiered KV Cache ADR

This page is contributor-facing architectural guidance.

It is not customer-facing product documentation and not an implementation
claim. It records the recommended future direction for a GPU/CPU/SSD tiered KV
cache so later implementation work stays grounded in the current oLLM
architecture.

## Status

- audience: implementers and maintainers
- kind: ADR / design decision
- implementation status: not implemented

## Scope

This ADR defines the recommended architecture for a future KV cache that spans:

- the active execution device
- host memory
- SSD-backed persistence

It is intentionally separate from the current `tiered-write-back` preset.
Today, `tiered-write-back` means:

- one runtime-scoped strategy preset
- a bounded hot in-memory tail
- a colder journal-backed spill tier on disk

That is useful, but it is not yet the full GPU/CPU/SSD architecture implied by
"tiered KV cache" in the long-context roadmap.

## Decision summary

oLLM should build the future multi-tier KV architecture around:

- immutable, fixed-token KV pages as the primary movement unit
- one authoritative persisted page store on SSD
- an optional decoded CPU residency tier
- an optional accelerator-resident hot tier
- page-driven promotion and eviction controlled by explicit runtime policy
- truthful benchmark and inspection reporting for each tier

The first implementation slice should not attempt a monolithic rewrite. The
recommended order is:

1. formalize the page contract and persisted layout
2. make SSD pages authoritative
3. add a CPU residency tier and page-promotion metrics
4. add accelerator page staging only when the attention path can consume pages
5. compose quantized cold pages and selector integration on top

## Why the current baseline is not enough

Current oLLM already has several explicit KV presets:

- `resident`
- `chunked`
- `paged`
- `streamed-segmented`
- `log-structured-journal`
- `sliding-window-ring-buffer`
- `quantized-cold-tier`
- `tiered-write-back`

Those presets are useful, but they still assume one primary storage/runtime
shape per request. Even the current `tiered-write-back` preset is still a
single strategy bundle, not a general architecture for multi-tier full-history
KV across GPU, CPU, and SSD.

The current code makes that limitation concrete:

- `src/ollm/kvcache.py`
  still exposes one active cache object per request path and treats the current
  persisted store as a single selected backend.
- `src/ollm/kv_cache_policy.py`
  selects flush and spill thresholds, but not a general promotion pipeline
  across multiple authoritative residency tiers.
- `src/ollm/kv_cache_state.py`
  can report resident, hot, and persisted state, but it does not yet distinguish
  accelerator-tier pages from CPU-tier pages.

## Design goals

- Preserve full-history KV semantics by default.
- Make GPU, CPU, and SSD ownership explicit instead of hidden behind one preset
  string.
- Keep the persisted layout durable, inspectable, and benchmark-observable.
- Allow quantized cold storage to compose with the tiered architecture instead
  of requiring a separate storage rewrite.
- Preserve deterministic selector behavior rather than turning this into an
  online optimizer.

## Non-goals

- Distributed inference or multi-host KV movement.
- Claiming that every backend can consume tiered KV immediately.
- Pretending the future architecture is just the current `tiered-write-back`
  preset with bigger thresholds.
- Hiding bounded-history semantics inside tiering. Sliding-window behavior
  remains a separate window-policy decision.

## Critical truth constraint

For standard full-attention decode, every token step still needs all prior KV.
That means a future GPU/CPU/SSD cache is not just an eviction problem.

If the model path still expects one contiguous per-layer KV tensor on the
accelerator, then demoting old pages to CPU or SSD only moves cost around; it
does not create a real scalable runtime architecture.

Because of that, accelerator tiering should only be claimed once the attention
path can consume page-iterated KV or an equivalent staged block interface.

This is the most important design guardrail in the whole document.

## Recommended data model

### Page unit

Use fixed-token, layer-local KV pages as the primary movement unit.

Each page should represent:

- one layer
- one contiguous token range
- one encoding contract
- one key payload
- one value payload

Recommended page identity fields:

- `schema_version`
- `model_identity`
- `backend_identity`
- `layer_idx`
- `start_token`
- `end_token`
- `dtype_or_encoding`
- `page_checksum`

Why fixed-token pages:

- they match the existing `paged` strategy direction
- they give a deterministic transfer unit for GPU/CPU/SSD movement
- they keep benchmark comparisons aligned to logical units rather than raw file counts
- they avoid the unbounded rewrite behavior of variable chunk ranges

### Authoritative persisted store

The SSD tier should be the authoritative persisted representation.

Recommended persisted structure:

- one root manifest per cache identity
- one per-layer page table
- immutable page blobs once sealed
- manifest updates written only after page blobs are durable

That structure should preserve the current good properties already emerging in
oLLM:

- explicit manifests instead of opaque `.pt` blobs
- durable format/version checks
- cache-root identity scoped by model/backend/lifecycle

## Tier model

### Tier 0: accelerator hot tier

Purpose:

- hold the current working set required by the active decode/prefill step
- prioritize append-biased recent pages plus any explicitly prefetched next pages

This tier is not authoritative.

### Tier 1: CPU warm tier

Purpose:

- hold decoded full-precision pages ready for accelerator promotion
- absorb repeated page faults without rereading SSD immediately

This tier is also not authoritative.

### Tier 2: SSD cold tier

Purpose:

- hold the authoritative long-lived page store
- optionally hold quantized cold pages in later phases

This is the durable source of truth for persistent lifecycle mode.

## Promotion and eviction semantics

### Promotion

Promotion should be page-driven, not tensor-global.

Recommended promotion path:

1. page requested by attention/runtime iterator
2. serve from accelerator tier if present
3. else serve from CPU tier and promote to accelerator
4. else read from SSD into CPU tier, then optionally promote to accelerator

Prefetch should follow the expected page scan order, not a generic recency
heuristic.

### Eviction

Eviction should differ by tier:

- accelerator tier: evict pages outside the active decode/prefetch window
- CPU tier: evict least-recently-served pages subject to configured budget
- SSD tier: do not "evict" as part of normal runtime pressure; use lifecycle GC

The design should not describe SSD page removal as normal cache eviction. That
is retention / GC, not working-set management.

## Write path

The write path should not force every token append to rewrite existing pages.

Recommended approach:

- keep one append-oriented mutable tail per layer in memory
- seal that tail into immutable SSD pages when it crosses page capacity
- only then publish the updated page table

This allows the current journal-oriented work to remain useful as an ingest
pattern without making the journal the final authoritative long-lived layout.

Recommended relationship:

- journal: write-optimized ingress
- paged store: authoritative long-lived layout

## Interaction with existing presets

### `resident`

`resident` remains the no-disk baseline.

In the future design, it is effectively the degenerate case where:

- all pages remain in memory
- no SSD tier is used
- no CPU/accelerator demotion occurs

### `paged`

`paged` should become the persistence-format foundation for the future tiered
architecture, not a competing direction.

### `quantized-cold-tier`

Quantization should apply to SSD-resident cold pages only, not to the active
accelerator hot tier by default.

That means quantization is orthogonal to tiering:

- tiering defines where pages live
- cold-tier encoding defines how SSD pages are represented

### `sliding-window-ring-buffer`

Sliding-window remains a separate window-policy axis.

Do not merge bounded-history semantics into the first tiered design. Otherwise
benchmarks will conflate:

- full-history tiering
- bounded-history eviction

### `tiered-write-back`

Treat the current `tiered-write-back` preset as the closest precursor, not as
the final architecture.

The design should explicitly say:

- current `tiered-write-back` proves hot/cold split reporting and batching
- it does not yet prove true GPU/CPU/SSD page-aware tiering

## Runtime planning and selector integration

The runtime selector should not choose tiered KV as one opaque winner.

Instead, planning should eventually surface at least:

- whether tiering is enabled
- page size
- cold-tier encoding
- CPU tier budget
- accelerator hot-tier budget
- expected fallback when tier-aware execution is unavailable

Relevant current surfaces:

- `src/ollm/runtime/plan.py`
- `src/ollm/runtime/inspection.py`

This should remain deterministic and table-driven. The selector may use
observed budgets and platform/model family, but this ADR rejects a vague online
optimizer framing.

## Benchmark and inspection truth

The current benchmark/reporting surfaces already understand:

- strategy id
- persistence format
- residency mode
- persisted tokens / artifacts
- resident bytes
- hot bytes
- spill counts
- compaction counts

Relevant current surfaces:

- `src/ollm/runtime/benchmark_details.py`
- `src/ollm/runtime/benchmark_history_summary_support.py`
- `docs/benchmarking.md`

The future tiered architecture should add explicit tier-aware observability:

- accelerator-tier resident pages / bytes
- CPU-tier resident pages / bytes
- SSD authoritative pages / bytes
- promotion counts by source and destination
- page-fault counts
- prefetched pages / bytes
- useful-prefetch hit rate
- demotion counts
- quantized cold-page bytes vs decoded warm-page bytes

Without those fields, oLLM would risk claiming tiered wins while hiding where
the bytes and latency actually moved.

## Failure, recovery, and fallback

The ADR makes these rules explicit:

- page blobs are immutable once published
- root/page-table manifests are versioned
- cache identity must include model, backend, encoding, page size, and schema
- partial writes must never become authoritative
- invalid manifests or missing page blobs invalidate the persistent cache root
- multi-process access requires explicit coordination; it is not implied
- unsupported backends fall back to non-tiered strategies rather than
  pretending to run tiered mode

Recommended persistent write rule:

1. write page blob
2. fsync/replace durable temp path
3. update page table
4. update root manifest last

## Recommended phased implementation order

### Phase 1

Make fixed-token pages the authoritative persisted layout for one existing
full-history strategy.

Recommended first slice:

- extend `paged` into the canonical persisted page-table format
- keep runtime behavior otherwise single-tier
- add tier-capable metadata fields without claiming tiering yet

### Phase 2

Add a CPU warm-page tier above the authoritative SSD layout.

Goals:

- avoid repeated SSD reads
- measure promotion and page-fault behavior
- keep accelerator behavior unchanged if page-aware attention is not ready

### Phase 3

Add page-aware accelerator staging for one optimized-native family and one
device class.

Only this phase should start making real GPU/CPU/SSD tiering claims.

### Phase 4

Compose quantized cold pages with the authoritative SSD store and benchmark the
quality/capacity tradeoff on top of the tiered layout.

### Phase 5

Integrate selector policy and observe-only tier recommendations based on page
faults, promotion pressure, CPU residency pressure, and accelerator working-set
fit.

## Recommendation

Proceed with the future tiered KV architecture, but do it as a paged,
page-aware, benchmark-truthful design.

Do not describe it as current `tiered-write-back` generalized and do not claim
real GPU/CPU/SSD scaling before page-aware execution exists.

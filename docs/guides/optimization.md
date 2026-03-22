# Optimization Guide

This guide explains how oLLM's optimized-native path differs from the generic runtime and how to interpret specialization behavior.

## Runtime tiers

### Optimized-native
Used when a built-in alias or compatible native-family local model matches a specialization provider.

Current native families:

- `llama`
- `gemma3`
- `qwen3-next`
- `gpt-oss`
- `voxtral`

### Transformers-generic
Used for compatible local or materialized models that can run through the generic Transformers-backed path.

## Specialization passes

Optimized-native planning can record reusable passes such as:

- `disk-cache`
- `cpu-offload`
- `gpu-offload`
- `mlp-chunking`
- `moe-routing`
- `attention-replacement`
- `multimodal-shell`
- `gds-export-weights`

These passes are validated against the assembled optimized runtime before execution proceeds.

## Fallback behavior

If an optimized specialization cannot satisfy its planned pass contract and a compatible generic Transformers path exists, oLLM falls back safely to `transformers-generic`.

This is intentional. The runtime does **not** pretend the optimized path succeeded.

## Offload and cache controls

When supported by the selected backend, oLLM can expose:

- disk KV cache
- CPU layer offload
- mixed GPU / CPU layer placement

These controls are backend-dependent. The generic path does not expose the same
low-level layer-placement controls as optimized-native runtimes.

Current CPU offload policies are:

- `auto`
- `prefix`
- `suffix`
- `middle-band`

`auto` currently resolves to `middle-band`. Simultaneous CPU and GPU offload is
intentionally rejected in this slice because the mixed-placement path is still
prefix-shaped and would be misleading if reported as policy-driven.

The optimized-native KV cache surface now exposes eight explicit presets:

- `chunked`
- `paged`
- `streamed-segmented`
- `log-structured-journal`
- `sliding-window-ring-buffer`
- `quantized-cold-tier`
- `tiered-write-back`
- `resident`

`chunked` persists a manifest-backed chunk store under
`cache_dir/kv_cache_chunked`. `paged` persists a fixed-capacity page table
under `cache_dir/kv_cache_paged`, so movement and reconstruction stay aligned
to deterministic page boundaries instead of variable chunk sizes.
`streamed-segmented` persists a sequential
segment-backed store under `cache_dir/kv_cache_streamed_segmented`.
`log-structured-journal` persists a single append-oriented journal per layer
under `cache_dir/kv_cache_log_structured_journal` and compacts entry metadata
deterministically once the journal crosses its configured entry threshold.
`sliding-window-ring-buffer` persists only the bounded recent tail under
`cache_dir/kv_cache_sliding_window_ring_buffer`; once the configured
`kv_cache_window_tokens` limit is exceeded, the oldest cached tokens are
evicted under a `drop-oldest` policy. This mode changes runtime semantics and
should be used only when a bounded recent context is the intended contract.
Current local proof keeps it as an explicit opt-in mode rather than a general
default strategy.
`quantized-cold-tier` persists the same full-history journal shape under
`cache_dir/kv_cache_quantized_cold_tier`, but stores colder entries in the
explicit `int8-symmetric-per-tensor` representation and dequantizes back to
the runtime dtype on load.
`resident` keeps full-history KV entirely in memory and does not initialize any
disk root. It is the truthful no-spill baseline, not a persistence strategy.
`tiered-write-back` persists only the colder KV prefix under
`cache_dir/kv_cache_tiered_write_back` while keeping a bounded hot region in
memory; its cold tier now uses a journal-backed append store. All seven
disk-backed presets use typed raw tensor payloads plus explicit metadata, and
none uses opaque
pickle-backed `.pt` cache blobs. The runtime also applies a
platform/resource-aware buffering or spill policy on top of the selected
strategy so the cache can trade write amplification against memory headroom
instead of flushing every delta identically on every machine.
That preset is still not the full future GPU/CPU/SSD tiered architecture.
`resident` does not initialize any disk-KV root at all; it keeps full-history
KV entirely in memory and exists as the explicit low-overhead baseline when the
active runtime can afford that footprint. It is intentionally not aligned with
oLLM's large-model spill/offload goal.

## Runtime strategy selector

The runtime now has a deterministic pre-run selector above the explicit KV
presets.

Current selector profiles are:

- `balanced`
- `latency`
- `capacity`
- `bounded-window`

Current selector-default candidates are intentionally narrow:

- `paged`
- `resident`
- `quantized-cold-tier`

The other presets stay explicit opt-in or pinned overrides for now:

- `sliding-window-ring-buffer`
- `streamed-segmented`
- `log-structured-journal`
- `tiered-write-back`

The selector is not the same thing as the live KV adaptation surface.
`kv_cache_adaptation_mode` still describes post-start observe-only or automatic
adaptation behavior, while the selector chooses the initial strategy before the
runtime starts.

Within one loaded runtime, the cache layer now also keeps a resident in-process
snapshot of the reconstructed per-layer KV state so repeated updates do not
need to reread and rebuild the same on-disk history every token. For the
`streamed-segmented` store specifically, readback now coalesces extents by
segment file instead of replaying a separate file-range read for every extent.

## GPT-OSS `gds_export` requirement

The optimized GPT-OSS path is intentionally strict:

- a validated `gds_export/` tree must be present beside the model
- the export manifest must remain inside that export directory
- unsafe torch-serialized or pickle-backed artifacts are rejected

## Hardware expectations

The benchmark harness is designed to stay truthful on limited-RAM hosts:

- planner overhead and no-specialization cost do not require large model loads
- runtime comparisons only load weights when the target actually exists locally
- the requested primary target now reports cold-start and warm-runtime behavior separately, including TTFT, inter-token latency, prompt/output throughput, peak memory, cache footprint, and supported utilization / allocator-gap metrics
- optimized-native benchmark requests can also expose native loader and KV IO timing summaries plus the storage paths used by the request
- prompt-length scaling, output-length scaling, and repeated-turn session growth are measured only for the requested primary target, not for every built-in family alias
- unavailable optimized comparisons are reported as unavailable instead of being fabricated

When present, the native runtime profile is the most truthful place to inspect
whether an optimized request actually used GDS, standard safetensor IO,
CPU-offloaded artifacts, or disk KV cache IO.

For disk KV specifically, `kvload` and `kvsave` now represent reads and writes
against the selected explicit disk-KV store rather than whole-layer torch
artifacts. Benchmark/runtime output also reports `kv_cache_strategy` so the
active backend is visible during A/B runs, and `cache_state` exposes the
hot/cold split, persisted artifact count, compaction count, and cold-store
format for tier-aware strategies.
The reported disk-cache footprint reflects the persisted chunk store only. A
selected KV policy may keep a bounded tail in memory until its spill or flush
threshold is reached.
When a request is satisfied from the resident in-process KV snapshot rather
than from disk, `kvload` can legitimately be absent for that step even though
disk KV remains the selected strategy.

Those native event totals are operation-level timings. On runtimes that submit
multiple storage reads before waiting for completion, the summed native IO
event totals can exceed the enclosing request wall-clock time because the reads
can overlap.

# Benchmarking

oLLM ships a dedicated runtime benchmark harness:

```bash
uv run python scripts/benchmark_runtime.py --device cpu --output .omx/runtime-benchmark.json
```

The CLI now defaults to the `quick` benchmark profile for development use. That
profile limits the run to the requested primary target, skips the family-wide
matrix and deeper scaling sweeps, and applies tighter per-lane budgets so a
strategy check cannot silently turn into an hour-long run.

Every benchmark run now also records its full raw payload plus a normalized
summary under `.omx/logs/benchmark-history/`. If a prior run with the same
comparison key exists, the CLI appends a comparison summary and emits any
obvious potential regressions on `stderr` without changing the JSON written to
`stdout`.

History matching now includes an explicit `codebase_label` so fork and upstream
baseline runs do not collide accidentally. By default, that label is derived
from the normalized `origin` remote URL. Override it only when you need a
different stable grouping:

```bash
uv run python scripts/benchmark_runtime.py \
  --device cpu \
  --history-codebase-label current-fork \
  --output .omx/runtime-benchmark.json
```

Use `--profile full` only when you explicitly want the heavier matrix:

```bash
uv run python scripts/benchmark_runtime.py \
  --profile full \
  --device cpu \
  --output .omx/runtime-benchmark-full.json
```

Use `--no-record-history` only when you explicitly want to skip that ledger.
The comparison key is based on the actual run shape, including model, device,
backend, strategy, codebase label, and prompt/profile controls, so
incomparable runs do not get mashed together.

For KV strategy A/B work, select the preset explicitly:

```bash
uv run python scripts/benchmark_runtime.py \
  --device cpu \
  --kv-cache-strategy streamed-segmented \
  --output .omx/runtime-benchmark-streamed.json
```

The paged strategy uses the same switch:

```bash
uv run python scripts/benchmark_runtime.py \
  --device cpu \
  --kv-cache-strategy paged \
  --output .omx/runtime-benchmark-paged.json
```

Tiered write-back uses the same switch:

```bash
uv run python scripts/benchmark_runtime.py \
  --device cpu \
  --kv-cache-strategy tiered-write-back \
  --output .omx/runtime-benchmark-tiered.json
```

The log-structured journal strategy is selected the same way:

```bash
uv run python scripts/benchmark_runtime.py \
  --device cpu \
  --kv-cache-strategy log-structured-journal \
  --output .omx/runtime-benchmark-journal.json
```

Resident mode uses the same switch, but it reports `cache_mode="resident-kv"`
and leaves `cache_dir_size_mb` empty because no on-disk KV root is created:

```bash
uv run python scripts/benchmark_runtime.py \
  --device cpu \
  --kv-cache-strategy resident \
  --output .omx/runtime-benchmark-resident.json
```

The bounded sliding-window mode requires an explicit window size:

```bash
uv run python scripts/benchmark_runtime.py \
  --device cpu \
  --kv-cache-strategy sliding-window-ring-buffer \
  --kv-cache-window-tokens 256 \
  --output .omx/runtime-benchmark-sliding-window.json
```

The harness is designed to stay truthful on hardware-constrained machines:

- it always measures specialization planner overhead without loading model weights
- it measures the extra planning cost when no specialization applies by creating a tiny local T5 fixture on the fly
- it reports a bounded family-wide comparison matrix for the current optimized families using any locally materialized built-in aliases it finds
- the requested `--model-reference` now gets deeper benchmark sections:
  - cold-start vs warm-runtime comparisons
  - TTFT and inter-token latency
  - prompt-token and output-token throughput
  - current and peak process RSS
  - accelerator memory, cache footprint, process CPU, best-effort accelerator utilization, and allocator-gap metrics when the host/runtime can measure them truthfully
  - optimized-native loader and KV IO timing summaries when stats are available
  - prompt-length scaling
  - output-length scaling
  - repeated-turn session growth
  - reopened-runtime session growth for persistent KV validation
- it marks unsupported metrics and non-executable optimized paths as unavailable instead of fabricating numbers

Only the runtime comparison loads requested model weights. The planning and no-specialization fallback measurements are intentionally lightweight.

## Sweep controls

The deeper primary-target sweeps can be tuned from the CLI:

```bash
uv run python scripts/benchmark_runtime.py \
  --profile full \
  --prompt-scale-tokens 32,128,512 \
  --output-scale-tokens 16,64,128 \
  --session-turns 4 \
  --session-max-new-tokens 4
```

Interpretation notes:

- family-wide results stay limited to cold-start and warm-runtime comparisons so the benchmark remains bounded
- session-growth uses its own per-turn output cap instead of inheriting the output-scaling sweep, because repeated-turn probes are intended to measure retained-session growth rather than long-form generation throughput
- prompt throughput is prompt tokens divided by TTFT/prefill latency
- output throughput is generated output tokens divided by total generation latency
- peak RSS includes a source label; long-lived warm/scaling/session probes use stage-local sampled peaks instead of process-lifetime peaks
- allocator-gap metrics are reported as reserved-minus-allocated style slack when the backend exposes the required counters; unsupported backends serialize them as `null`

On loader-streamed families such as optimized Gemma3 on CPU, a long per-turn
session-growth response can become dominated by repeated safetensor layer reads
instead of disk-KV behavior. The bounded `--session-max-new-tokens` default is
there to keep this probe representative and practical on development machines.

## Native loader and KV IO profile

For optimized-native runs, the request metrics can now include a
`native_runtime_profile` payload. This captures event timing summaries emitted
by the runtime itself, rather than numbers inferred from high-level wall clock
latency.

Typical event names include:

- `layer_load`
- `experts_load`
- `kvload`
- `kvsave`
- `gds_read`
- `safetensor_read`
- `safetensor_pread`
- `offloaded_cpu_to_cuda`

The same profile also reports storage-path labels such as:

- `gds`
- `safetensor-io`
- `cpu-offloaded-artifacts`
- `disk-kv-cache`
- `torch-artifact-io`

These fields are only present when the selected runtime actually emits native
stats. Generic Transformers-backed runs may report `null` here.

For disk KV requests, `disk-kv-cache` now refers to the explicit strategy root
under `cache_dir/kv_cache_chunked`,
`cache_dir/kv_cache_paged`,
`cache_dir/kv_cache_streamed_segmented`, or
`cache_dir/kv_cache_tiered_write_back`, or
`cache_dir/kv_cache_sliding_window_ring_buffer`, not to pickle-backed `.pt`
layer artifacts. The request metrics report both `kv_cache_strategy` and
`cache_state`, so benchmark comparisons can distinguish the selected backend
and, for tier-aware strategies, the current hot/cold split. `cache_dir_size_mb`
describes only the persisted on-disk portion of the cache, while `cache_state`
surfaces persisted artifact counts, compaction counts, cold-store format,
cold-tier representation, hot in-memory tokens, spill counts, and, for bounded
window strategies, the active window size plus eviction totals.
Within one loaded runtime, the cache layer can satisfy repeated requests from
an in-process resident KV snapshot instead of rereading the same persisted
history from disk. In those cases, `kvload` may legitimately disappear even
though disk KV is still the active strategy. The `streamed-segmented` store now
also coalesces extents by segment file on readback, so fewer file-range reads
can show up under the same `disk-kv-cache` storage path label.
For `tiered-write-back`, the persisted cold tier now uses a journal-backed
append store under `cache_dir/kv_cache_tiered_write_back/cold`, so the tiered
spill path no longer depends on the chunked cold-store substrate.
That current preset is still a bounded hot-tail plus cold journal strategy, not
the future GPU/CPU/SSD multi-tier architecture.
For `log-structured-journal`, compaction is visible both through
`cache_state.compaction_count` and, when it occurs during a request, native
runtime profile timing under `kvcompact`. `kvsave` now measures append-path
write cost without double-counting compaction rewrite time; when compaction
occurs, `kvcompact` reports that rewrite separately.
For `paged`, benchmark output surfaces the explicit `paged-manifest`
persistence format and `ollm-kv-paged` cold-store format. The persisted
artifact count for this strategy means page-table entries, not raw blob files,
so comparisons stay aligned to logical fixed-size movement units.
For `quantized-cold-tier`, the persisted cold journal uses the explicit
`int8-symmetric-per-tensor` representation and benchmark output surfaces that
representation through `cache_state.cold_tier_representation`.
For `sliding-window-ring-buffer`, benchmark output also surfaces
`cache_state.window_max_tokens`, `cache_state.eviction_policy`,
`cache_state.eviction_count`, and `cache_state.evicted_tokens` so bounded
history runs are never confused with full-history runs. When you benchmark this
mode, pass an explicit `--kv-cache-window-tokens` value so benchmark history
does not compare unlike window sizes. Current local CPU and MPS proof keeps
this mode in the explicit opt-in bucket rather than the selector-default
bucket; it materially bounds persisted KV, but it is not a general perf win.
For `resident`, benchmark output surfaces `cache_mode="resident-kv"` plus the
resident layer/token/byte counters while leaving `cache_dir_size_mb` empty
because there is no persisted KV root.
Cold, warm, prompt-scaling, output-scaling, and session-growth probes all use
the same persistent benchmark-history ledger, so bounded proof runs remain
recorded and comparable instead of becoming ad hoc local artifacts.
For persistent-KV proof work, `--probe-mode reopen-session-growth` reloads the
runtime every turn under `kv_cache_lifecycle="persistent"` so the resulting
turns exercise persisted cold-KV reuse instead of same-process resident reuse.

When the optimized loader uses async submission plus later completion, the
native event totals represent per-operation storage latency, not a partition of
request wall time. That means summed `gds_read`, `safetensor_*`, or other
native event totals can exceed the request's top-level wall-clock latency when
multiple reads overlap.

## Hardware-specific workflow

For a lightweight non-CUDA baseline:

```bash
uv run python scripts/benchmark_runtime.py --device cpu --output .omx/runtime-benchmark-cpu.json
```

On a CUDA host with a materialized optimized-native target, run the same
workflow against `cuda:0` to capture native loader and storage-path behavior:

```bash
uv run python scripts/benchmark_runtime.py --device cuda:0 --output .omx/runtime-benchmark-cuda.json
```

## Adjacent upstream baseline workflow

When you want a concept-level baseline against upstream oLLM proper, keep that
clone outside this repo tree and let it maintain its own benchmark history.
One safe pattern is:

```bash
git clone https://github.com/Mega4alik/ollm.git ../ollm-upstream-baseline
cd ../ollm-upstream-baseline
python3 -m venv .venv
source .venv/bin/activate
pip install --no-build-isolation -e .
```

Upstream does not ship this fork's `scripts/benchmark_runtime.py` harness, so
do not pretend the same command surface exists there. Instead, keep any
compatibility probe or wrapper local-only and outside this repo, and label its
artifacts explicitly as upstream-baseline. That keeps upstream sources,
artifacts, and history out of this fork's tracked tree while still making the
baseline attributable and reproducible.

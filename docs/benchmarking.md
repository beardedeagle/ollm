# Benchmarking

oLLM ships a dedicated runtime benchmark harness:

```bash
uv run python scripts/benchmark_runtime.py --device cpu --output .omx/runtime-benchmark.json
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
- it marks unsupported metrics and non-executable optimized paths as unavailable instead of fabricating numbers

Only the runtime comparison loads requested model weights. The planning and no-specialization fallback measurements are intentionally lightweight.

## Sweep controls

The deeper primary-target sweeps can be tuned from the CLI:

```bash
uv run python scripts/benchmark_runtime.py \
  --prompt-scale-tokens 32,128,512 \
  --output-scale-tokens 16,64,128 \
  --session-turns 4
```

Interpretation notes:

- family-wide results stay limited to cold-start and warm-runtime comparisons so the benchmark remains bounded
- prompt throughput is prompt tokens divided by TTFT/prefill latency
- output throughput is generated output tokens divided by total generation latency
- peak RSS includes a source label; long-lived warm/scaling/session probes use stage-local sampled peaks instead of process-lifetime peaks
- allocator-gap metrics are reported as reserved-minus-allocated style slack when the backend exposes the required counters; unsupported backends serialize them as `null`

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

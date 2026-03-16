# Benchmarking

oLLM ships a dedicated runtime benchmark harness:

```bash
uv run python scripts/benchmark_runtime.py --device cpu --output .omx/runtime-benchmark.json
```

The harness is designed to stay truthful on hardware-constrained machines:

- it always measures specialization planner overhead without loading model weights
- it measures the extra planning cost when no specialization applies by creating a tiny local T5 fixture on the fly
- it reports a runtime-comparison matrix for the current optimized families using any locally materialized built-in aliases it finds
- it marks missing or non-executable optimized paths as unavailable instead of fabricating numbers
- it also benchmarks the requested `--model-reference` directly and reports `comparison_available: false` when the optimized path cannot execute on the current host

Only the runtime comparison loads requested model weights. The planning and no-specialization fallback measurements are intentionally lightweight.

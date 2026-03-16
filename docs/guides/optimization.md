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

### Provider-backed
Used when the execution boundary is a provider endpoint.

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

These controls are backend-dependent. Provider-backed runtimes do not expose the same low-level layer-placement controls as optimized-native runtimes.

## GPT-OSS `gds_export` requirement

The optimized GPT-OSS path is intentionally strict:

- a validated `gds_export/` tree must be present beside the model
- the export manifest must remain inside that export directory
- unsafe torch-serialized or pickle-backed artifacts are rejected

## Hardware expectations

The benchmark harness is designed to stay truthful on limited-RAM hosts:

- planner overhead and no-specialization cost do not require large model loads
- runtime comparisons only load weights when the target actually exists locally
- unavailable optimized comparisons are reported as unavailable instead of being fabricated

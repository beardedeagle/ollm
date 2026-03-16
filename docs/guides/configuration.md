# Configuration

## Runtime configuration

The CLI and the Python library both build on `RuntimeConfig` and `GenerationConfig`.

Key runtime configuration fields:

- `model_reference` — the model to resolve
- `models_dir` — root for local materialized models
- `device` — torch device such as `cpu` or `cuda:0`
- `backend` — optional backend override
- `provider_endpoint` — provider API root URL when required
- `multimodal` — enable multimodal planning when non-text inputs are expected
- `use_specialization` — whether optimized-native specialization is allowed
- `cache_dir` / `use_cache` — disk KV cache controls
- `offload_cpu_layers` / `offload_gpu_layers` — native offload controls when supported

Generation configuration fields:

- `max_new_tokens`
- `temperature`
- `top_p`
- `top_k`
- `seed`
- `stream`

See [API Reference: Runtime Config](../api/config.md).

## Backend override

`--backend` lets you force one of:

- `optimized-native`
- `transformers-generic`
- `ollama`
- `openai-compatible`

The override is validated against the resolved model reference. If the backend is incompatible, oLLM fails with a structured error instead of silently switching to something else.

## Specialization toggle

`--no-specialization` disables optimized-native specialization selection and forces the generic path when one exists.

Important constraint:

- `--backend optimized-native` cannot be combined with `--no-specialization`

## Provider endpoints

Provider endpoints must be:

- absolute `http` or `https` URLs
- credential-free in the URL itself

Examples:

```bash
# good
--provider-endpoint http://127.0.0.1:1234/v1

# rejected
--provider-endpoint https://user:pass@example.test/v1
```

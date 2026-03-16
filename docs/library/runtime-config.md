# Runtime Configuration

The library and CLI share the same runtime configuration model.

## `RuntimeConfig`

Use `RuntimeConfig` to describe:

- the model reference
- local materialization root
- device
- backend override
- provider endpoint
- multimodal enablement
- specialization enablement
- cache and offload behavior

## `GenerationConfig`

Use `GenerationConfig` to describe:

- token limits
- sampling controls
- seeding
- streaming

See:

- [Runtime Config API](../api/config.md)

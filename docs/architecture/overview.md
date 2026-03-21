# Architecture Overview

oLLM is organized around a resolver-and-plan model instead of a fixed allowlist gate.

## High-level flow

1. parse a model reference
2. resolve it into a normalized `ResolvedModel`
3. discover capabilities
4. select a backend and support level
5. refine the runtime plan through backend-specific probes
6. route CLI and future server transports through the internal `ApplicationService`
7. load the runtime or expose the plan as inspection output
8. execute through a local runtime

## Main subsystems

- `ModelReference` parsing
- `ModelResolver`
- capability discovery
- `BackendSelector`
- `RuntimePlan`
- `ApplicationService`
- `RuntimeLoader`
- `RuntimeExecutor`
- specialization registry / matching / application
- KV strategy matrix and future tiered KV design

See also:

- [KV Strategy Matrix](kv-strategy-matrix.md)
- [Tiered KV Cache Design](tiered-kv-cache.md)

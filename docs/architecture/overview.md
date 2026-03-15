# Architecture Overview

oLLM is organized around a resolver-and-plan model instead of a fixed allowlist gate.

## High-level flow

1. parse a model reference
2. resolve it into a normalized `ResolvedModel`
3. discover capabilities
4. select a backend and support level
5. refine the runtime plan through backend-specific probes
6. load the runtime or expose the plan as inspection output
7. execute through either a local runtime or a provider transport

## Main subsystems

- `ModelReference` parsing
- `ModelResolver`
- capability discovery
- `BackendSelector`
- `RuntimePlan`
- `RuntimeLoader`
- `RuntimeExecutor`
- specialization registry / matching / application
- provider backends

# Architecture Overview

oLLM is organized around a resolver-and-plan model instead of a fixed allowlist
gate. The same internal flow powers the CLI, the Python library, and the local
server surface.

## High-level flow

<div class="mermaid">
flowchart LR
    A[User input<br/>CLI / Python / Server] --> B[ModelReference.parse]
    B --> C[ModelResolver]
    C --> D[Capability discovery]
    D --> E[BackendSelector]
    E --> F[RuntimeLoader.plan]
    F --> G[Strategy selector]
    G --> H{Inspect only?}
    H -->|Yes| I[RuntimePlan / JSON inspection]
    H -->|No| J[RuntimeLoader.load]
    J --> K[ApplicationService / RuntimeClient]
    K --> L[RuntimeExecutor]
</div>

1. Parse a user-facing model reference.
2. Resolve it into a normalized `ResolvedModel`.
3. Discover capability and support metadata.
4. Select a backend and support level.
5. Refine the runtime plan through backend-specific probes.
6. Apply deterministic pre-run strategy selection for KV behavior.
7. Either expose the plan as inspection output or load the runtime.
8. Execute through the shared application/runtime stack.

## Main subsystems

| Subsystem | Responsibility |
| --- | --- |
| `ModelReference` | Parses opaque model input such as aliases, Hugging Face IDs, and local paths. |
| `ModelResolver` | Normalizes a reference into a `ResolvedModel` with capability and source metadata. |
| Capability discovery | Inspects local model artifacts when built-in metadata is not enough. |
| `BackendSelector` | Chooses the most truthful backend and support level for the current runtime config. |
| `RuntimePlan` | Carries the resolved backend choice, specialization state, and inspection details. |
| `RuntimeLoader` | Materializes models when needed, plans execution, and loads the backend runtime. |
| `ApplicationService` | Provides the shared control-plane surface used by CLI and server transports. |
| `RuntimeExecutor` | Executes prompt requests once a runtime has been loaded. |
| Specialization registry | Matches and applies optimized-native providers and passes. |

## Why this shape exists

This layering keeps model resolution, planning, loading, and execution separate.
That lets oLLM answer "what would run and why?" before it loads weights, which
is why `ollm prompt --plan-json`, `ollm models info`, and the server planning
surfaces can stay truthful without forcing a full runtime load.

See also:

- [KV Strategy Matrix](kv-strategy-matrix.md)
- [Backend Selection](backend-selection.md)
- [Specialization](specialization.md)

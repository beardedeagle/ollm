# Backend Selection

`BackendSelector` turns a resolved model plus a runtime configuration into a `RuntimePlan`.

## Selection precedence

1. explicit backend override when valid
2. optimized-native specialization when valid and enabled
3. provider-backed backend for provider refs whose contract is satisfied
4. generic Transformers backend when compatible
5. unsupported with a structured reason

## Why it matters

This makes backend selection:

- deterministic
- inspectable
- testable
- explainable in both CLI and library surfaces

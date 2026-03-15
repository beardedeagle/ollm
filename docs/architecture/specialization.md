# Specialization

Specialization is an optimization layer, not the only execution path.

## Pieces

- specialization providers
- pass matching
- pass planning
- applied-specialization validation
- safe fallback to `transformers-generic`

## Key contract

A specialization may be skipped safely, and when planned specialization cannot be applied, the runtime can fall back safely when a generic path exists.

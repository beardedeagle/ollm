# Runtime Planning and Inspection

The resolver and planner are intentionally inspectable. oLLM does not treat backend selection as opaque magic.

## Plan-only surfaces

These commands can print the runtime plan without loading a backend:

```bash
ollm prompt --plan-json --model llama3-8B-chat
ollm chat --plan-json --model llama3-8B-chat
ollm doctor --plan-json --model llama3-8B-chat
ollm models info llama3-8B-chat --plan-json
```

## What a runtime plan contains

A plan includes:

- resolved backend id
- support level
- generic model kind when applicable
- disk/offload support flags
- specialization enablement and state
- specialization provider id
- planned specialization pass ids
- fallback reason when applicable

## Specialization states

- `not-planned`
- `planned`
- `applied`
- `fallback`

Planning-only surfaces report the **planned** state. Actual prompt response metadata reports the finalized execution state.

## Why this matters

This makes it possible to distinguish:

- what oLLM resolved
- why it picked that backend
- whether specialization was only planned or actually applied
- whether execution had to fall back to the generic path

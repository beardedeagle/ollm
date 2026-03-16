# Model Resolution

Model resolution converts user input into a structured `ResolvedModel`.

## Inputs

- built-in aliases
- Hugging Face repo IDs
- local paths

## Output

A `ResolvedModel` records:

- normalized name
- source kind
- repo id / revision when relevant
- local path when relevant
- capabilities
- native family when recognized
- architecture/model type when inspected from local artifacts

## Important design point

The catalog is metadata for built-in aliases, not an admission gate for the whole runtime.

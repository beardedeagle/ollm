# CLI Overview

The CLI has five main surfaces:

- `ollm prompt` — one-shot scripted prompting
- `ollm chat` — interactive chat
- `ollm doctor` — diagnostics and runtime inspection
- `ollm models` — discovery, inspection, download, and path reporting
- `ollm serve` — local-only REST API server

## Shared runtime controls

These commands share the same runtime-planning vocabulary:

- `--model`
- `--models-dir`
- `--device`
- `--backend`
- `--multimodal`
- `--no-specialization`
- `--plan-json`

See the per-command pages for the full behavior of each surface.

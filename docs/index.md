# oLLM

oLLM is a Python library and terminal interface for local LLM inference. It combines:

- optimized-native runtimes for built-in aliases when a specialization matches
- a generic Transformers-backed path for compatible local or materialized models
- runtime inspection so you can see which backend will run, why it was selected, and what the current support level is

## Audience

- operators and end users who want to run prompts and inspect local models
- Python developers who want to embed oLLM through `RuntimeClient` or the low-level optimized-native helpers
- contributors who need architecture, verification, and docs-build guidance

## Documentation map

### Getting Started
- [Installation](guides/installation.md)
- [Configuration](guides/configuration.md)
- [Quickstart](guides/quickstart.md)

### User Guide
- [Terminal Interface](terminal-interface.md)
- [Local Server API](guides/local-server.md)
- [Model References](guides/model-references.md)
- [Multimodal Workflows](guides/multimodal.md)
- [Runtime Planning and Inspection](guides/runtime-planning.md)
- [Model Discovery](guides/models-discovery.md)
- [Optimization Guide](guides/optimization.md)
- [Troubleshooting](guides/troubleshooting.md)
- [Benchmarking](benchmarking.md)

### CLI Reference
- [CLI Overview](cli.md)
- [`ollm serve`](cli/server.md)
- [`ollm prompt`](cli/prompt.md)
- [`ollm chat`](cli/chat.md)
- [`ollm doctor`](cli/doctor.md)
- [`ollm models`](cli/models.md)

### Library and API
- [High-level Client](library/runtime-client.md)
- [Runtime Configuration](library/runtime-config.md)
- [Optimized-native Helpers](library/inference.md)
- [API Reference](api/client.md)
- [Local Server API Reference](api/server.md)

### Architecture and Contributing
- [Architecture Overview](architecture/overview.md)
- [Contributing Guide](guides/contributing.md)
- [Development](development.md)

## Core concepts

### Model references

`--model` accepts opaque model references, not just a fixed built-in list. Supported forms include:

- a built-in alias such as `llama3-1B-chat`
- a Hugging Face repository ID such as `Qwen/Qwen2.5-7B-Instruct`
- a local model directory

### Support levels

oLLM reports one of three active support levels:

- `optimized`
- `generic`
- `unsupported`

### Safety model

oLLM intentionally stays conservative in several places:

- the generic runtime only loads local or materialized weights from `safetensors`
- unsupported references fail with an explicit reason instead of silently leaving the local runtime boundary
- planning and execution report specialization and fallback state explicitly

## Quick examples

```bash
ollm prompt --model llama3-8B-chat "Summarize this file"
ollm prompt --model gemma3-12B --multimodal --image ./diagram.png "Describe this image"
ollm doctor --json
ollm models list
```

## Examples

- `examples/example.py`
- `examples/example_image.py`
- `examples/example_audio.py`

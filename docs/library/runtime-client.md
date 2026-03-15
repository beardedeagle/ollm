# High-level Client

`RuntimeClient` is the recommended high-level Python entry point.

Use it when you want:

- model resolution without loading
- planning and inspection output
- one-shot prompting
- reusable chat sessions
- the same runtime semantics as the CLI

```python
from pathlib import Path

from ollm import GenerationConfig, RuntimeClient, RuntimeConfig

client = RuntimeClient()
runtime_config = RuntimeConfig(
    model_reference="Qwen/Qwen2.5-7B-Instruct",
    models_dir=Path("models"),
    device="cpu",
    backend="transformers-generic",
    use_specialization=False,
)

plan = client.describe_plan(runtime_config)
response = client.prompt(
    "List planets",
    runtime_config=runtime_config,
    generation_config=GenerationConfig(stream=False, max_new_tokens=64),
)
```

See the generated API docs for the full symbol reference:

- [Runtime Client API](../api/client.md)

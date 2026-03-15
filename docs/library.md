# Library

## High-level runtime API

`RuntimeClient` is the high-level entry point built directly on the same resolver, planner, loader, and executor stack used by the CLI.

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
print(plan["runtime_plan"]["backend_id"])

response = client.prompt(
    "List planets",
    runtime_config=runtime_config,
    generation_config=GenerationConfig(stream=False, max_new_tokens=64),
)
print(response.text)
```

Useful high-level operations:

- `RuntimeClient.resolve(...)`
- `RuntimeClient.plan(...)`
- `RuntimeClient.describe_plan(...)`
- `RuntimeClient.load(...)`
- `RuntimeClient.prompt(...)`
- `RuntimeClient.session(...)`

## Low-level optimized-native helpers

The direct optimized-native helpers still exist when you want explicit control over the native specialization path.

```python
from ollm import Inference, TextStreamer

o = Inference("llama3-1B-chat", device="cuda:0", logging=True)
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=2)
past_key_values = o.DiskCache(cache_dir="./kv_cache/")
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)
```

For compatible local Llama or Gemma3 directories, `AutoInference` remains the direct optimized-native helper:

```python
from ollm import AutoInference

o = AutoInference(
    "./models/gemma3-12B",
    adapter_dir="./myadapter/checkpoint-20",
    device="cuda:0",
    multimodality=False,
    logging=True,
)
```

Use `RuntimeClient` for new high-level code. Keep `Inference` and `AutoInference` for direct low-level optimized-native control.

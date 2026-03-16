# Optimized-native Helpers

`Inference` and `AutoInference` remain available for direct optimized-native control.

## When to use them

Use these helpers when you specifically want:

- direct control over the optimized-native path
- explicit CPU/GPU offload operations through the specialization provider
- direct access to the native model/tokenizer/processor objects

## When not to use them

For new high-level application code, prefer `RuntimeClient`, because it uses the same resolver, backend selection, provider support, and plan inspection model as the CLI.

## Typical `Inference` example

```python
from ollm import Inference, TextStreamer

o = Inference("llama3-1B-chat", device="cuda:0", logging=True)
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=2)
past_key_values = o.DiskCache(cache_dir="./kv_cache/")
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)
```

## Typical `AutoInference` example

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

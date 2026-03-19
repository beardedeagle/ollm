# Optimized-native Helpers

`Inference` and `AutoInference` remain available for direct optimized-native control.

## When to use them

Use these helpers when you specifically want:

- direct control over the optimized-native path
- explicit CPU/GPU offload operations through the specialization provider
- direct access to the native model/tokenizer/processor objects

## When not to use them

For new high-level application code, prefer `RuntimeClient`, because it uses the same resolver, backend selection, and plan inspection model as the CLI.

## Typical `Inference` example

```python
from ollm import Inference, TextStreamer

o = Inference("llama3-1B-chat", device="cuda:0", logging=True)
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=2)
past_key_values = o.DiskCache(
    cache_dir="./kv_cache/",
    cache_strategy="streamed-segmented",
)
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)
```

That disk cache path now uses an explicit disk-KV store under
`cache_dir/kv_cache_chunked` by default, with explicit
dtype/shape/sequence metadata and raw payloads instead of pickle-backed torch
artifacts. When the selected runtime uses
`kv_cache_strategy="streamed-segmented"`, it writes to
`cache_dir/kv_cache_streamed_segmented` instead. When it uses
`kv_cache_strategy="tiered-write-back"`, it writes the cold tier to
`cache_dir/kv_cache_tiered_write_back` through a journal-backed append store
while keeping a bounded hot tail in memory. The active runtime then applies a
platform/resource-aware buffering or spill policy on top of the selected
store. `Inference.DiskCache()` accepts the same switch through
`cache_strategy=...`.

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

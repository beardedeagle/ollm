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
    kv_cache_strategy="chunked",
    kv_cache_lifecycle="runtime-scoped",
    kv_cache_adaptation_mode="observe-only",
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

The direct optimized-native helpers exist when you want explicit control over the native specialization path.

```python
from ollm import Inference, TextStreamer

o = Inference("llama3-1B-chat", device="cuda:0", logging=True)
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=2, policy="middle-band")
past_key_values = o.DiskCache(
    cache_dir="./kv_cache/",
    cache_strategy="streamed-segmented",
)
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)
```

`RuntimeConfig.kv_cache_strategy` selects the KV cache baseline explicitly.

- `resident` keeps full-history KV in memory with no disk root.
- `chunked` uses `cache_dir/kv_cache_chunked`.
- `paged` uses `cache_dir/kv_cache_paged` and preserves full history through an
  explicit page table.
- `streamed-segmented` uses `cache_dir/kv_cache_streamed_segmented`.
- `log-structured-journal` uses `cache_dir/kv_cache_log_structured_journal`.
- `quantized-cold-tier` uses `cache_dir/kv_cache_quantized_cold_tier`.
- `sliding-window-ring-buffer` uses
  `cache_dir/kv_cache_sliding_window_ring_buffer` and retains only the most
  recent `kv_cache_window_tokens` cached tokens.
- `tiered-write-back` uses `cache_dir/kv_cache_tiered_write_back` while keeping
  a bounded hot region in memory and a journal-backed cold tier on disk.

The bounded `sliding-window-ring-buffer` mode is a semantic recent-context mode,
not a transparent replacement for the full-history strategies. All seven
disk-backed presets use typed raw payloads plus explicit metadata instead of
opaque torch cache blobs. The active runtime then applies a
platform/resource-aware buffering or spill policy on top of the selected store.

The low-level optimized-native helper exposes the same cache switch directly
through `Inference.DiskCache(cache_strategy=...)`.

Supported CPU offload policies are `auto`, `prefix`, `suffix`, and
`middle-band`. `auto` resolves to `middle-band`, and combined
CPU+GPU offload is intentionally unsupported in this slice.

For compatible local Llama or Gemma3 directories, `AutoInference` is the direct optimized-native helper:

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

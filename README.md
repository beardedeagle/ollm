<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://ollm.s3.us-east-1.amazonaws.com/files/logo2.png">
    <img alt="oLLM" src="https://ollm.s3.us-east-1.amazonaws.com/files/logo2.png" width=52%>
  </picture>
</p>

<h3 align="center">
LLM Inference for Large-Context Offline Workloads
</h3>

oLLM is a lightweight Python library for large-context LLM inference, built on top of Huggingface Transformers and PyTorch. It enables running models like [gpt-oss-20B](https://huggingface.co/openai/gpt-oss-20b), [qwen3-next-80B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) or [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on 100k context using ~$200 consumer GPU with 8GB VRAM.  No quantization is used—only fp16/bf16 precision. 

<p dir="auto"><em><a href="https://github.com/Mega4alik/ollm/wiki/Releases">Latest updates</a> (1.0.3)</em> 🔥</p>
<ul dir="auto">
<li><code>AutoInference</code> for compatible local Llama and Gemma3 model directories with optional <a href="https://github.com/huggingface/peft">PEFT</a> adapter support</li>
<li><code>kvikio</code> and <code>flash-attn</code> are optional now, meaning no hardware restrictions beyond HF transformers</li>
<li>Multimodal <b>voxtral-small-24B</b> (audio+text) added. <a href="https://github.com/Mega4alik/ollm/blob/main/example_audio.py">[sample with audio]</a> </li>
<li>Multimodal <b>gemma3-12B</b> (image+text) added. <a href="https://github.com/Mega4alik/ollm/blob/main/example_image.py">[sample with image]</a> </li>
<li><b>qwen3-next-80B</b> (160GB model) added with <span style="color:blue">⚡️1tok/2s</span> throughput (our fastest model so far)</li>
<li>gpt-oss-20B flash-attention-like implementation added to reduce VRAM usage </li>
<li>gpt-oss-20B chunked MLP added to reduce VRAM usage </li>
</ul>

---
###  8GB Nvidia 3060 Ti Inference memory usage:

| Model   | Weights | Context length | KV cache |  Baseline VRAM (no offload) | oLLM GPU VRAM | oLLM Disk (SSD) |
| ------- | ------- | -------- | ------------- | ------------ | ---------------- | --------------- |
| [qwen3-next-80B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | 160 GB (bf16) | 50k | 20 GB | ~190 GB   | ~7.5 GB | 180 GB  |
| [gpt-oss-20B](https://huggingface.co/openai/gpt-oss-20b) | 13 GB (packed bf16) | 10k | 1.4 GB | ~40 GB   | ~7.3GB | 15 GB  |
| [gemma3-12B](https://huggingface.co/google/gemma-3-12b-it)  | 25 GB (bf16) | 50k   | 18.5 GB          | ~45 GB   | ~6.7 GB       | 43 GB  |
| [llama3-1B-chat](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)  | 2 GB (bf16) | 100k   | 12.6 GB          | ~16 GB   | ~5 GB       | 15 GB  |
| [llama3-3B-chat](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)  | 7 GB (bf16) | 100k  | 34.1 GB | ~42 GB   | ~5.3 GB     | 42 GB |
| [llama3-8B-chat](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)  | 16 GB (bf16) | 100k  | 52.4 GB | ~71 GB   | ~6.6 GB     | 69 GB  |

<small>By "Baseline" we mean typical inference without any offloading</small>

How do we achieve this:

- Loading layer weights from SSD directly to GPU one by one
- Offloading KV cache to SSD and loading back directly to GPU, no quantization or PagedAttention
- Offloading layer weights to CPU if needed
- FlashAttention-2 with online softmax. Full attention matrix is never materialized. 
- Chunked MLP. Intermediate upper projection layers may get large, so we chunk MLP as well 
---
Typical use cases include:
- Analyze contracts, regulations, and compliance reports in one pass
- Summarize or extract insights from massive patient histories or medical literature
- Process very large log files or threat reports locally
- Analyze historical chats to extract the most common issues/questions users have
---
**Supported GPUs**: NVIDIA (with additional performance benefits from `kvikio` and `flash-attn`), AMD, and Apple Silicon (MacBook).



## Getting Started

oLLM now targets Python 3.12 and newer.

Use `uv` to create and manage the local environment:
```bash
git clone https://github.com/Mega4alik/ollm.git
cd ollm
uv sync

# optional features
uv sync --extra adapters   # AutoInference with PEFT adapters
uv sync --extra audio      # voxtral audio example
uv sync --extra cuda       # flash-attn + triton acceleration
uv sync --extra export     # export scripts
uv sync --group dev        # pytest and contributor tooling
```

If you prefer `pip`, install oLLM from source with editable mode:

```bash
pip install --no-build-isolation -e .
pip install --no-build-isolation -e ".[adapters]" # optional extras

# for Nvidia GPUs with cuda (optional): 
pip install kvikio-cu{cuda_version} Ex, kvikio-cu12 #speeds up the inference
```
> 💡 **Note**  
> `kvikio` remains a manual install because the package name depends on your CUDA version. For `voxtral-small-24B`, use `uv sync --extra audio` or install `mistral-common[audio]` and `librosa`.

Check out the [Troubleshooting](https://github.com/Mega4alik/ollm/wiki/Troubleshooting) in case of any installation issues 

## Terminal Interface

oLLM now ships with a first-party terminal interface in addition to the importable library.

```bash
ollm                         # interactive terminal chat
ollm chat                    # explicit alias for interactive chat
ollm prompt "List planets"   # one-shot prompt
ollm doctor --json           # environment and runtime diagnostics
ollm models list             # known, local, and provider-discovered model references
```

Use `ollm` or `ollm chat` only from an interactive terminal. For scripts, pipes, and automation use `ollm prompt`:

```bash
ollm prompt --model llama3-8B-chat "Summarize this file"
cat notes.txt | ollm prompt --stdin --model llama3-8B-chat
ollm prompt --multimodal --model gemma3-12B --image ./diagram.png "Describe this image"
```

`ollm doctor` reports missing optional extras, runtime availability, path issues, and model readiness. `ollm models` provides `list`, `info`, `download`, and `path` subcommands for both built-in aliases and arbitrary model references. `ollm models list` now acts as a discovery view: it combines built-in aliases, local materialized models, and provider-discovered entries from Ollama and LM Studio by default, and it can probe generic OpenAI-compatible providers when you pass `--discover-provider openai-compatible --provider-endpoint <url>`.

`--model` now accepts opaque model references. Today that means:
- built-in aliases such as `llama3-1B-chat` and `gemma3-12B` load through registered optimized specialization providers
- Hugging Face repo IDs such as `Qwen/Qwen2.5-7B-Instruct` resolve and materialize locally
- local model directories resolve directly
- Ollama provider refs such as `ollama:llama3.2` execute through the local Ollama API
- LM Studio provider refs such as `lmstudio:qwen2.5` execute through the local OpenAI-compatible API
- generic OpenAI-compatible refs such as `openai-compatible:local-model` execute when `--provider-endpoint` is supplied

The current generic execution path now covers compatible local or materialized Transformers-backed:
- causal language models such as Qwen2-family checkpoints
- encoder-decoder text generation models such as T5-family checkpoints
- image-text conditional generation models that expose a processor-backed `vision_config`

When the resolved model matches a native family specialization (`llama`, `gemma3`, `qwen3-next`, `gpt-oss`, or `voxtral`), `ollm` now records and selects the matching optimized specialization provider through the runtime plan instead of hard-coding model-family branches inside `Inference.load_model()`. Built-in aliases still prefer the optimized native backend, and compatible local native-family directories can now do the same when a specialization provider matches while preserving the original local-path reference internally for optimized local loads. Provider-backed execution is now live for `ollama:<model>`, `lmstudio:<model>`, and `openai-compatible:<model>` references, while Msty-specific execution and audio-focused generic conditional generation remain deferred.

Optimized-native planning now also records reusable specialization passes such as `disk-cache`, `cpu-offload`, `gpu-offload`, `mlp-chunking`, `moe-routing`, `attention-replacement`, `multimodal-shell`, and `gds-export-weights`. Those passes are now validated against the assembled optimized runtime before execution proceeds. If an optimized specialization cannot satisfy its planned pass contract and a compatible generic Transformers path exists, `ollm` falls back safely to `transformers-generic` instead of silently pretending the optimized path succeeded.

Planning-only surfaces such as `ollm doctor` and `ollm models info --json` now expose an explicit `specialization_state` plus the planned specialization provider/pass ids. They do **not** execute a backend load just to answer an inspection request, so they should be read as runtime-planning output.

Actual execution surfaces follow the finalized runtime plan instead. In particular, prompt response metadata includes the execution backend, specialization state, applied specialization pass ids, and any recorded fallback reason.

For provider-backed execution, `ollm` currently supports:
- the local Ollama API on the default `http://127.0.0.1:11434` endpoint
- LM Studio on the default `http://127.0.0.1:1234/v1` OpenAI-compatible endpoint
- arbitrary OpenAI-compatible servers when `--provider-endpoint` points at the provider API root

`ollm doctor --model <provider-ref>` and `ollm models info <provider-ref>` probe the configured endpoint and report executability truthfully. If an Ollama model advertises `vision` capability, `ollm prompt --multimodal --model ollama:<model>` can send local-file or data-URL image inputs through the Ollama chat API. The OpenAI-compatible backend is intentionally narrower right now: it is text-only, rejects `--top-k`, does not support PEFT adapters or custom layer offload, and requires `--provider-endpoint` for `openai-compatible:<model>` references. Audio provider execution remains unsupported.

The optimized GPT-OSS provider is intentionally stricter than before: it only matches when a validated `gds_export/` tree is present beside the model, and that export manifest must stay inside the export directory and avoid torch-serialized or pickle-backed artifacts.

For safety, the generic runtime only loads local or materialized model weights from safetensors artifacts. Arbitrary `.bin` or pickle-backed checkpoints are intentionally rejected on that path.

Interactive prompt history is in-memory by default. Use `--history-file` only when you explicitly want persistent local history.

Chat also supports queued multimodal attachments when the resolved model capabilities include the requested modality:

```bash
ollm chat --model gemma3-12B --multimodal
/image ./diagram.png
/send Describe this image

ollm chat --model voxtral-small-24B --multimodal
/audio ./sample.wav
/send What can you tell me about this audio?
```

## Library Example

Code snippet sample 

```python
from ollm import Inference, file_get_contents, TextStreamer
o = Inference("llama3-1B-chat", device="cuda:0", logging=True) # built-in optimized aliases such as llama3-1B-chat or qwen3-next-80B
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=2) #(optional) offload some layers to CPU for speed boost
past_key_values = o.DiskCache(cache_dir="./kv_cache/") #set None if context is small
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

messages = [{"role":"system", "content":"You are helpful AI assistant"}, {"role":"user", "content":"List planets"}]
input_ids = o.tokenizer.apply_chat_template(messages, reasoning_effort="minimal", tokenize=True, add_generation_prompt=True, return_tensors="pt").to(o.device)
outputs = o.model.generate(input_ids=input_ids,  past_key_values=past_key_values, max_new_tokens=500, streamer=text_streamer).cpu()
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
```
or run the sample script as `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python example.py` 

```python
# with AutoInference, you can run compatible local Llama or Gemma3 model directories
# uv sync --extra adapters
from ollm import AutoInference
o = AutoInference("./models/gemma3-12B", # compatible local Llama or Gemma3 model
  adapter_dir="./myadapter/checkpoint-20", # PEFT adapter checkpoint if available
  device="cuda:0", multimodality=False, logging=True)
...
```
**More samples**
- [gemma3-12B image+text](https://github.com/Mega4alik/ollm/blob/main/example_image.py)
- [voxtral-small-24B audio+text](https://github.com/Mega4alik/ollm/blob/main/example_audio.py)
- [AutoInference + SFT](https://github.com/Mega4alik/peftee?tab=readme-ov-file#usage)

## Development

The project now builds with Hatchling and uses `uv` for environment and lock management.

Run the automated test suite with:

```bash
uv run pytest
```

Fast syntax-only verification remains:

```bash
uv run python -m compileall src tests
```


## Knowledge base
- [Documentation](https://github.com/Mega4alik/ollm/wiki/Documentation)
- [Community](https://github.com/Mega4alik/ollm/wiki/Community) articles, video, blogs
- [Troubleshooting](https://github.com/Mega4alik/ollm/wiki/Troubleshooting)


## Roadmap
*For visibility of what's coming next (subject to change)*
- Qwen3-Next quantized version
- Qwen3-VL or alternative vision model
- Qwen3-Next MultiTokenPrediction in R&D


## Contact us
If there’s a model family you’d like to see optimized natively, feel free to suggest it in the [discussion](https://github.com/Mega4alik/ollm/discussions/4) — I’ll do my best to make it happen.

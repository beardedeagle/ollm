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
<li>Multimodal <b>voxtral-small-24B</b> (audio+text) added. <a href="examples/example_audio.py">[sample with audio]</a> </li>
<li>Multimodal <b>gemma3-12B</b> (image+text) added. <a href="examples/example_image.py">[sample with image]</a> </li>
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

The primary project documentation now lives under `docs/` and is built with MkDocs Material:

- [Overview](docs/index.md)
- [Getting Started / Installation](docs/guides/installation.md)
- [User Guide: Terminal Interface](docs/terminal-interface.md)
- [User Guide: Model References](docs/guides/model-references.md)
- [User Guide: Providers](docs/guides/providers.md)
- [Optimization Guide](docs/guides/optimization.md)
- [CLI Reference](docs/cli.md)
- [Architecture](docs/architecture/overview.md)
- [API Reference](docs/api/client.md)
- [Development](docs/development.md)

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
ollm prompt --model llama3-8B-chat --backend transformers-generic --no-specialization "Summarize this file"
ollm prompt --model llama3-8B-chat --plan-json
```

`ollm doctor` reports missing optional extras, runtime availability, path issues, and model readiness. `ollm models` provides `list`, `info`, `download`, and `path` subcommands for both built-in aliases and arbitrary model references. `ollm models list` acts as a discovery view: it combines built-in aliases, local materialized models, and provider-discovered entries from Ollama and LM Studio by default, and it can also probe `openai-compatible` or `msty` providers when you pass `--discover-provider <name> --provider-endpoint <url>`.

Runtime selection and inspection controls:
- `--backend` lets you force one of `optimized-native`, `transformers-generic`, `ollama`, or `openai-compatible` when that backend is valid for the resolved model reference
- `--no-specialization` disables optimized native specialization selection and forces the generic path when one exists
- `--plan-json` prints the resolved runtime plan as JSON and exits without running generation
- `--provider-endpoint` must be an absolute `http` or `https` URL and must not embed credentials

`ollm prompt`, `ollm chat`, `ollm doctor`, and `ollm models info` now all honor `--backend` and `--no-specialization`. `ollm prompt`, `ollm chat`, `ollm doctor`, and `ollm models info` also support `--plan-json` for script-friendly inspection of the resolver/backend decision.

Runtime vocabulary:
- support levels:
  - `optimized`: a native specialization provider can run the reference
  - `generic`: the Transformers-backed generic runtime can run the reference
  - `provider-backed`: execution goes through a provider API such as Ollama or an OpenAI-compatible endpoint
  - `unsupported`: the reference resolves, but the current runtime cannot execute it
- discovery sources:
  - `built-in`: a shipped oLLM alias
  - `discovered-local`: a materialized local model directory found under `--models-dir`
  - `discovered-provider`: a model reference returned by provider discovery
- availability terms:
  - `materialized` / `not-materialized` describe whether local weights are present on disk
  - `available` / `unavailable` describe whether a provider reference is currently reachable and executable
  - `ollm models list --installed` filters to materialized local entries only; provider references are never treated as installed

`--model` now accepts opaque model references. Today that means:
- built-in aliases such as `llama3-1B-chat` and `gemma3-12B` load through registered optimized specialization providers
- Hugging Face repo IDs such as `Qwen/Qwen2.5-7B-Instruct` resolve and materialize locally
- local model directories resolve directly
- Ollama provider refs such as `ollama:llama3.2` execute through the local Ollama API
- Msty provider refs such as `msty:llama3.2` execute through the Msty Local AI service when `--provider-endpoint` is supplied
- LM Studio provider refs such as `lmstudio:qwen2.5` execute through the local OpenAI-compatible API
- generic OpenAI-compatible refs such as `openai-compatible:local-model` execute when `--provider-endpoint` is supplied

The current generic execution path now covers compatible local or materialized Transformers-backed:
- causal language models such as Qwen2-family checkpoints
- encoder-decoder text generation models such as T5-family checkpoints
- image-text conditional generation models that expose a processor-backed `vision_config`

When the resolved model matches a native family specialization (`llama`, `gemma3`, `qwen3-next`, `gpt-oss`, or `voxtral`), `ollm` now records and selects the matching optimized specialization provider through the runtime plan instead of hard-coding model-family branches inside `Inference.load_model()`. Built-in aliases still prefer the optimized native backend, and compatible local native-family directories can now do the same when a specialization provider matches while preserving the original local-path reference internally for optimized local loads. Provider-backed execution is now live for `ollama:<model>`, `msty:<model>`, `lmstudio:<model>`, and `openai-compatible:<model>` references, while audio-focused generic conditional generation remains deferred.

Optimized-native planning now also records reusable specialization passes such as `disk-cache`, `cpu-offload`, `gpu-offload`, `mlp-chunking`, `moe-routing`, `attention-replacement`, `multimodal-shell`, and `gds-export-weights`. Those passes are now validated against the assembled optimized runtime before execution proceeds. If an optimized specialization cannot satisfy its planned pass contract and a compatible generic Transformers path exists, `ollm` falls back safely to `transformers-generic` instead of silently pretending the optimized path succeeded.

Planning-only surfaces such as `ollm doctor` and `ollm models info --json` now expose an explicit `specialization_state` plus the planned specialization provider/pass ids. They do **not** execute a backend load just to answer an inspection request, so they should be read as runtime-planning output.

Actual execution surfaces follow the finalized runtime plan instead. In particular, prompt response metadata includes the execution backend, specialization state, applied specialization pass ids, and any recorded fallback reason.

For provider-backed execution, `ollm` currently supports:
- the local Ollama API on the default `http://127.0.0.1:11434` endpoint
- Msty Local AI through the existing Ollama transport when `--provider-endpoint` points at the configured Msty Local AI service endpoint
- LM Studio on the default `http://127.0.0.1:1234/v1` OpenAI-compatible endpoint
- arbitrary OpenAI-compatible servers when `--provider-endpoint` points at the provider API root

Example Ollama vision prompt with a remote image URL:

```bash
ollm prompt --multimodal --model ollama:llava --image https://example.com/diagram.png "Describe this image"
```

`ollm doctor --model <provider-ref>` and `ollm models info <provider-ref>` probe the configured endpoint and report executability truthfully. If an Ollama-family provider model advertises `vision` capability, `ollm prompt --multimodal --model ollama:<model>` can now send local-file paths, base64 data URLs, and remote `http`/`https` image URLs through the Ollama chat API. Remote image URLs are fetched client-side with bounded downloads and content-type validation before being base64-forwarded to Ollama. `msty:<model>` now reuses the same Ollama transport but intentionally requires an explicit `--provider-endpoint`, because the Msty Local AI service endpoint is user-configurable and `ollm` does not guess it. The OpenAI-compatible backend now supports audio **request execution** for generic `openai-compatible:<model>` references when `--provider-endpoint` is supplied: `--audio` inputs may be local `.wav`/`.mp3` files, base64 data URLs, or remote `http`/`https` WAV/MP3 URLs, and `ollm` fetches/validates remote audio client-side before forwarding it as OpenAI-compatible `input_audio`. Because the generic OpenAI-compatible `/models` probe does not expose per-model audio capabilities, `ollm models info` and `ollm doctor` keep the model modalities conservative and instead report `audio_input_support: request-capable` in the runtime-plan details for `openai-compatible:` refs. `lmstudio:<model>` remains text-only in `ollm`, because the current LM Studio OpenAI-compatible docs do not give this repo a verified audio input contract. Provider-backed execution still rejects `--top-k`, PEFT adapters, and custom layer offload controls.

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

The new high-level library entry path is `RuntimeClient`, which is built directly on the resolver/backend/runtime stack used by the CLI.

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

`RuntimeClient.session(...)` returns a reusable `ChatSession` built on the same runtime stack when you want conversational state in Python code.

### Low-level optimized-native API

The older low-level optimized-native helpers still exist for direct control of the native specialization path:

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

You can still run the original sample script as `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python examples/example.py`.
**More samples**
- [gemma3-12B image+text](examples/example_image.py)
- [voxtral-small-24B audio+text](examples/example_audio.py)
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

Build the documentation site with:

```bash
uv run --group docs mkdocs build --strict
```

### Benchmark and perf proof

oLLM now ships a dedicated runtime benchmark harness:

```bash
uv run python scripts/benchmark_runtime.py --device cpu --output .omx/runtime-benchmark.json
```

The harness is designed to stay truthful on hardware-constrained machines:
- it always measures specialization planner overhead without loading model weights
- it measures the extra planning cost when no specialization applies by using a tiny local T5 fixture created on the fly
- it reports a runtime-comparison matrix for the current optimized families, using any locally materialized built-in aliases it finds and marking missing families as unavailable instead of fabricating results
- it also benchmarks the requested `--model-reference` directly and reports `comparison_available: false` when the optimized path cannot execute on the current host

Only the runtime comparison loads requested model weights. The planning and no-specialization fallback measurements are low-RAM and safe to run on development laptops.


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

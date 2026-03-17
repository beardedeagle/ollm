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
ollm models list             # known and locally discovered model references
```

Use `ollm` or `ollm chat` only from an interactive terminal. For scripts, pipes, and automation use `ollm prompt`:

```bash
ollm prompt --model llama3-8B-chat "Summarize this file"
cat notes.txt | ollm prompt --stdin --model llama3-8B-chat
ollm prompt --multimodal --model gemma3-12B --image ./diagram.png "Describe this image"
ollm prompt --model llama3-8B-chat --backend transformers-generic --no-specialization "Summarize this file"
ollm prompt --model llama3-8B-chat --plan-json
```

`ollm doctor` reports missing optional extras, runtime availability, path issues, and model readiness. `ollm models` provides `list`, `info`, `download`, and `path` subcommands for both built-in aliases and arbitrary model references. `ollm models download` materializes only runtime-critical local artifacts rather than a full Hugging Face repository snapshot, and `ollm models list` acts as a discovery view over built-in aliases and local materialized models under `--models-dir`.

Runtime selection and inspection controls:
- `--backend` lets you force one of `optimized-native` or `transformers-generic` when that backend is valid for the resolved model reference
- `--no-specialization` disables optimized native specialization selection and forces the generic path when one exists
- `--plan-json` prints the resolved runtime plan as JSON and exits without running generation

`ollm prompt`, `ollm chat`, `ollm doctor`, and `ollm models info` now all honor `--backend` and `--no-specialization`. `ollm prompt`, `ollm chat`, `ollm doctor`, and `ollm models info` also support `--plan-json` for script-friendly inspection of the resolver/backend decision.

Configuration layering is now first-class:

- built-in defaults remain the lowest-precedence layer
- `./ollm.toml` is loaded automatically when present
- `OLLM_*` environment variables override config-file values
- explicit CLI flags override both env and config-file defaults

You can also point to a different config file with `OLLM_CONFIG_FILE=/path/to/ollm.toml`.

Example `ollm.toml`:

```toml
[runtime]
model_reference = "llama3-8B-chat"
models_dir = "models"
device = "mps"
backend = "optimized-native"
cache_dir = "kv_cache"
use_cache = true

[generation]
max_new_tokens = 256
temperature = 0.0
stream = true

[server]
host = "127.0.0.1"
port = 8000
reload = false
```

Example environment overrides:

```bash
export OLLM_RUNTIME__MODEL_REFERENCE=Qwen/Qwen2.5-7B-Instruct
export OLLM_RUNTIME__DEVICE=cpu
export OLLM_GENERATION__MAX_NEW_TOKENS=128
```

This slice configures runtime, generation, and future server defaults. Request-specific values that are not part of that schema, such as the prompt/chat system message, still remain explicit CLI arguments today.

Runtime vocabulary:
- support levels:
  - `optimized`: a native specialization provider can run the reference
  - `generic`: the Transformers-backed generic runtime can run the reference
  - `unsupported`: the reference resolves, but the current runtime cannot execute it
- discovery sources:
  - `built-in`: a shipped oLLM alias
  - `discovered-local`: a materialized local model directory found under `--models-dir`
- availability terms:
  - `materialized` / `not-materialized` describe whether local weights are present on disk
  - `ollm models list --installed` filters to materialized local entries only

`--model` now accepts opaque model references. Today that means:
- built-in aliases such as `llama3-1B-chat` and `gemma3-12B` load through registered optimized specialization providers
- Hugging Face repo IDs such as `Qwen/Qwen2.5-7B-Instruct` resolve and materialize locally
- local model directories resolve directly

The current generic execution path now covers compatible local or materialized Transformers-backed:
- causal language models such as Qwen2-family checkpoints
- encoder-decoder text generation models such as T5-family checkpoints
- image-text conditional generation models that expose a processor-backed `vision_config`

When the resolved model matches a native family specialization (`llama`, `gemma3`, `qwen3-next`, `gpt-oss`, or `voxtral`), `ollm` now records and selects the matching optimized specialization provider through the runtime plan instead of hard-coding model-family branches inside `Inference.load_model()`. Built-in aliases still prefer the optimized native backend, and compatible local native-family directories can now do the same when a specialization provider matches while preserving the original local-path reference internally for optimized local loads.

Optimized-native planning now also records reusable specialization passes such as `disk-cache`, `cpu-offload`, `gpu-offload`, `mlp-chunking`, `moe-routing`, `attention-replacement`, `multimodal-shell`, and `gds-export-weights`. Those passes are now validated against the assembled optimized runtime before execution proceeds. If an optimized specialization cannot satisfy its planned pass contract and a compatible generic Transformers path exists, `ollm` falls back safely to `transformers-generic` instead of silently pretending the optimized path succeeded.

Planning-only surfaces such as `ollm doctor` and `ollm models info --json` now expose an explicit `specialization_state` plus the planned specialization provider/pass ids. They do **not** execute a backend load just to answer an inspection request, so they should be read as runtime-planning output.

Actual execution surfaces follow the finalized runtime plan instead. In particular, prompt response metadata includes the execution backend, specialization state, execution-device profile details for optimized-native runs, applied specialization pass ids, and any recorded fallback reason.

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
uv run ruff format src tests examples scripts
uv run ruff check src tests examples scripts
uv run python scripts/check_python_standards.py
uv run python -m compileall src tests
uv run ty check src tests
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
- the requested `--model-reference` now gets deeper analysis:
  - cold-start vs warm-runtime comparisons
  - TTFT and inter-token latency
  - prompt-token and output-token throughput
  - current and peak process RSS
  - accelerator memory, cache footprint, process CPU, best-effort accelerator utilization, and allocator-gap metrics when the host/runtime can measure them truthfully
  - prompt-length scaling, output-length scaling, and repeated-turn session-growth sweeps
- family-wide comparisons stay bounded to cold-start and warm-runtime generic-vs-optimized results so the harness remains practical on development machines
- unsupported metrics and non-executable optimized paths remain explicitly unavailable instead of being fabricated
- peak RSS reports now carry source labels so warm/scaling/session sections can use stage-local sampled peaks instead of misleading process-lifetime maxima

Useful knobs for the primary-target sweeps:

```bash
uv run python scripts/benchmark_runtime.py \
  --prompt-scale-tokens 32,128,512 \
  --output-scale-tokens 16,64,128 \
  --session-turns 4
```

Only the runtime comparison loads requested model weights. The planning and no-specialization fallback measurements are low-RAM and safe to run on development laptops.

### Engineering standards

The canonical contributor standards live in [docs/guides/python-standards.md](docs/guides/python-standards.md).

This repo is currently being treated as greenfield:

- no legacy-only code paths
- no compatibility scaffolding
- delete bad shapes instead of preserving them

The repo-local standards checker is:

```bash
uv run python scripts/check_python_standards.py
```

The current repo-wide remediation matrix lives in
[docs/guides/python-standards-audit.md](docs/guides/python-standards-audit.md).


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

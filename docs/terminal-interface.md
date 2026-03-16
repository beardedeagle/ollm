# Terminal Interface

## Core commands

```bash
ollm                         # interactive terminal chat
ollm chat                    # explicit alias for interactive chat
ollm prompt "List planets"   # one-shot prompt
ollm doctor --json           # environment and runtime diagnostics
ollm models list             # built-in, local, and provider-discovered model references
```

Use `ollm` or `ollm chat` only from an interactive terminal. For scripts, pipes, and automation use `ollm prompt`.

## Runtime controls

- `--backend` forces a valid backend for the resolved model reference
- `--no-specialization` disables optimized-native specialization and prefers the generic path when available
- `--plan-json` prints the resolved runtime plan and exits without running generation
- `--provider-endpoint` sets the provider API root URL and must be an absolute `http` or `https` URL without embedded credentials

`ollm prompt`, `ollm chat`, `ollm doctor`, and `ollm models info` all honor these controls.

## Model references

`--model` accepts opaque model references. Supported forms include:

- built-in aliases such as `llama3-1B-chat` and `gemma3-12B`
- Hugging Face repo IDs such as `Qwen/Qwen2.5-7B-Instruct`
- local model directories
- provider references such as:
  - `ollama:llama3.2`
  - `msty:llama3.2`
  - `lmstudio:qwen2.5`
  - `openai-compatible:local-model`

## Support levels

oLLM reports one of four support levels for a resolved model reference:

- `optimized` — a native specialization provider can run the reference
- `generic` — the Transformers-backed generic runtime can run the reference
- `provider-backed` — execution goes through a provider API
- `unsupported` — the reference resolves, but the current runtime cannot execute it

## Discovery and availability terms

`ollm models list` is a discovery view. It combines:

- `built-in` entries shipped by oLLM
- `discovered-local` entries found under `--models-dir`
- `discovered-provider` entries returned by provider discovery

Availability terms are intentionally different for local and provider references:

- `materialized` / `not-materialized` describe whether local weights are present on disk
- `available` / `unavailable` describe whether a provider reference is currently reachable and executable

`ollm models list --installed` filters to materialized local entries only. Provider references are never treated as installed.

## Generic, optimized, and provider-backed execution

The generic execution path covers compatible local or materialized Transformers-backed:

- causal language models
- encoder-decoder text generation models
- image-text conditional generation models that expose a processor-backed `vision_config`

When the resolved model matches a native family specialization (`llama`, `gemma3`, `qwen3-next`, `gpt-oss`, or `voxtral`), oLLM records and selects an optimized-native specialization provider through the runtime plan instead of hard-coding model-family branches in `Inference.load_model()`.

Provider-backed execution currently supports:

- Ollama on `http://127.0.0.1:11434`
- Msty Local AI through the existing Ollama transport when `--provider-endpoint` is supplied
- LM Studio on `http://127.0.0.1:1234/v1`
- arbitrary OpenAI-compatible endpoints when `--provider-endpoint` is supplied

## Specialization visibility and fallback

Planning-only surfaces such as `ollm doctor` and `ollm models info --json` expose the resolved backend, specialization state, and planned specialization pass ids without loading a runtime.

Execution surfaces follow the finalized runtime plan. Prompt response metadata includes:

- execution backend
- specialization state
- applied specialization pass ids
- any recorded fallback reason

If an optimized specialization cannot satisfy its planned pass contract and a compatible generic path exists, oLLM falls back safely to `transformers-generic` instead of pretending the optimized path succeeded.

## Provider-specific notes

- `ollama:<model>` supports vision requests with local files, base64 data URLs, and remote `http` / `https` image URLs
- `msty:<model>` reuses the Ollama transport and requires an explicit `--provider-endpoint`
- `openai-compatible:<model>` supports audio request inputs from local `.wav` / `.mp3` files, base64 data URLs, and remote WAV/MP3 URLs when `--provider-endpoint` is supplied
- `lmstudio:<model>` remains text-only in oLLM because the current LM Studio OpenAI-compatible docs do not provide a verified audio input contract for this repo

Provider-backed execution rejects `--top-k`, PEFT adapters, and custom layer offload controls.

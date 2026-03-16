# `ollm prompt`

`ollm prompt` is the script-friendly execution surface.

## Use cases

- one-shot prompts
- stdin-driven automation
- structured JSON output
- multimodal requests
- runtime-plan inspection with `--plan-json`

## Key options

- `--model`
- `--backend`
- `--multimodal`
- `--image`
- `--audio`
- `--no-specialization`
- `--stream / --no-stream`
- `--format text|json`
- `--plan-json`

## Examples

```bash
ollm prompt --model llama3-8B-chat "Summarize this file"
cat notes.txt | ollm prompt --stdin --model llama3-8B-chat
ollm prompt --model gemma3-12B --multimodal --image ./diagram.png "Describe this image"
ollm prompt --model voxtral-small-24B --multimodal --audio ./sample.wav "Describe this clip"
ollm prompt --model llama3-8B-chat --plan-json
```

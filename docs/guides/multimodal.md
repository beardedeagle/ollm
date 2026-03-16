# Multimodal Workflows

## Images

### Generic/local image-text models
Use `--multimodal` plus `--image` with a compatible model such as `gemma3-12B`.

```bash
ollm prompt --model gemma3-12B --multimodal --image ./diagram.png "Describe this image"
```

### Ollama vision models
For `ollama:<model>` refs that advertise vision capability, oLLM supports:

- local image files
- base64 image data URLs
- remote `http` / `https` image URLs

Remote image URLs are fetched client-side with bounded downloads and content-type validation before being base64-forwarded to the Ollama chat API.

## Audio

### Optimized-native local audio
`voxtral-small-24B` supports audio through the optimized-native path.

### OpenAI-compatible provider audio requests
For generic `openai-compatible:<model>` refs, `--audio` may contain:

- local `.wav` or `.mp3` files
- base64 audio data URLs
- remote `http` / `https` WAV/MP3 URLs

Remote audio is fetched and validated client-side before it is forwarded as OpenAI-compatible `input_audio`.

### Current exclusions

- `ollama:` does **not** support audio inputs in oLLM
- `msty:` does **not** support audio inputs in oLLM
- `lmstudio:` remains text-only in oLLM

## Interactive chat attachments

```bash
ollm chat --model gemma3-12B --multimodal
/image ./diagram.png
/send Describe this image
```

```bash
ollm chat --model voxtral-small-24B --multimodal
/audio ./sample.wav
/send What can you tell me about this audio?
```

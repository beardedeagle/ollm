# Multimodal Workflows

## Images

### Generic/local image-text models
Use `--multimodal` plus `--image` with a compatible model such as `gemma3-12B`.

```bash
ollm prompt --model gemma3-12B --multimodal --image ./diagram.png "Describe this image"
```

## Audio

### Optimized-native local audio
`voxtral-small-24B` supports audio through the optimized-native path.

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

# Quickstart

## Text prompt

```bash
ollm prompt --model llama3-8B-chat "Summarize this file"
```

## Read from stdin

```bash
cat notes.txt | ollm prompt --stdin --model llama3-8B-chat
```

## Inspect the runtime plan

```bash
ollm prompt --model llama3-8B-chat --plan-json
```

## Force the generic path

```bash
ollm prompt --model llama3-8B-chat --backend transformers-generic --no-specialization "Summarize this file"
```

## Provider-backed prompt

```bash
ollm prompt --model openai-compatible:local-model \
  --provider-endpoint http://127.0.0.1:1234/v1 \
  "Say hello"
```

## Discover available model references

```bash
ollm models list
ollm models list --installed
ollm models list --discover-provider openai-compatible --provider-endpoint http://127.0.0.1:1234/v1
```

## Run diagnostics

```bash
ollm doctor --json
ollm doctor --model ollama:llava --json
```

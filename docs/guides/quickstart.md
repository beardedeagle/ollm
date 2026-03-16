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

## Discover available model references

```bash
ollm models list
ollm models list --installed
```

## Run diagnostics

```bash
ollm doctor --json
```

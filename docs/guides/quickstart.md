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

## Start the local API server

```bash
uv sync --extra server
ollm serve
```

Once the server is running locally, inspect the schema at `/openapi.json` or
open the interactive docs at `/docs`.

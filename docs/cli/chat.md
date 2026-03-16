# `ollm chat`

`ollm chat` is the interactive terminal interface.

## Use cases

- conversational sessions
- transcript resume/autosave workflows
- queued multimodal attachments
- interactive runtime selection while keeping the same planning model as `ollm prompt`

## Key options

- `--model`
- `--backend`
- `--provider-endpoint`
- `--multimodal`
- `--resume`
- `--save`
- `--history-file`
- `--plain`
- `--plan-json`

## Interactive attachment flow

```text
/image ./diagram.png
/audio ./sample.wav
/send Describe the attachment
```

Use `ollm prompt` rather than `ollm chat` for scripts, pipes, and automation.

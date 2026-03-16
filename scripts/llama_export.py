import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ollm.async_io import path_mkdir, path_write_text

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUT_DIR = "gds_export"
path_mkdir(Path(OUT_DIR), exist_ok=True)

# Load weights on CPU (normal HF path), then export raw
# If your model is sharded across multiple .safetensors, iterate them.
state_dict = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=None,
    torch_dtype=torch.float16,  # "auto" same as original torch.bfloat16,
    low_cpu_mem_usage=True,
).state_dict()

manifest = {}
for name, tensor in state_dict.items():
    # Only export layer weights to keep it small; adjust filter as needed
    if not name.startswith(("model.layers", "transformer.h")):
        continue

    t = tensor.to("cpu").contiguous()  # ensure contiguous for .tofile
    filename = f"{name.replace('.', '__')}.bin"
    path = Path(OUT_DIR) / filename
    path_mkdir(path.parent, parents=True, exist_ok=True)
    t.numpy().tofile(path)  # raw bytes

    manifest[name] = {
        "path": filename,
        "dtype": str(t.dtype).replace("torch.", ""),  # e.g., "float16"
        "shape": list(t.shape),
    }

path_write_text(Path(OUT_DIR) / "manifest.json", json.dumps(manifest, indent=2))

print(f"Exported {len(manifest)} tensors to {OUT_DIR}")

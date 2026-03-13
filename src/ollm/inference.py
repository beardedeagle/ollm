from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from ollm.gds_loader import DenseWeightsLoader, GDSWeights, MoEWeightsLoader, SingleDenseWeightsLoader
from ollm.kvcache import KVCache
from ollm.runtime.catalog import find_model_catalog_entry
from ollm.utils import Stats


def get_attn_implementation() -> str | None:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        print("Warning: flash_attention_2 is not imported. The context length will be limited")
        return None


def download_hf_snapshot(
    repo_id: str,
    model_dir: str,
    force_download: bool = False,
    revision: str | None = None,
) -> None:
    print(f"Downloading {repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        force_download=force_download,
        revision=revision,
    )


class Inference:
    def __init__(self, model_id: str, device: str = "cuda:0", logging: bool = True, multimodality: bool = False):
        self.model_id = model_id
        self.device = torch.device(device)
        self.multimodality = multimodality
        self.stats = Stats() if logging else None

    def hf_download(self, model_dir: str, force_download: bool = False) -> None:
        entry = find_model_catalog_entry(self.model_id)
        if entry is None:
            raise ValueError(
                f"Inference only supports built-in optimized aliases. Received {self.model_id!r}."
            )
        download_hf_snapshot(entry.repo_id, model_dir, force_download=force_download)

    def ini_model(self, models_dir: str = "./models/", force_download: bool = False) -> None:
        entry = find_model_catalog_entry(self.model_id)
        if entry is None:
            raise ValueError(
                f"Inference only supports built-in optimized aliases. Received {self.model_id!r}."
            )

        model_dir = Path(models_dir).expanduser().resolve() / entry.model_id
        if force_download or not model_dir.exists():
            self.hf_download(str(model_dir), force_download=force_download)

        self.load_model(str(model_dir))

    def load_model(self, model_dir: str) -> None:
        print("loading model from", model_dir)
        if self.model_id == "qwen3-next-80B":
            from ollm import qwen3_next

            qwen3_next.loader = MoEWeightsLoader(model_dir, device=self.device)
            qwen3_next.stats = self.stats
            self.model = qwen3_next.MyQwen3NextForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                attn_implementation=get_attn_implementation(),
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )
        elif self.model_id == "gemma3-12B":
            from ollm import gemma3

            gemma3.loader = DenseWeightsLoader(model_dir, device=self.device)
            gemma3.stats = self.stats
            automodel = gemma3.MyGemma3ForConditionalGeneration if self.multimodality else gemma3.MyGemma3ForCausalLM
            self.model = automodel.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                attn_implementation=get_attn_implementation(),
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_dir)
        elif self.model_id == "voxtral-small-24B":
            from ollm import voxtral

            voxtral.loader = DenseWeightsLoader(model_dir, device=self.device)
            voxtral.stats = self.stats
            self.model = voxtral.MyVoxtralForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype="auto",
                device_map="cpu",
                attn_implementation=get_attn_implementation(),
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_dir)
            self.tokenizer = self.processor.tokenizer
        elif self.model_id == "gpt-oss-20B":
            from ollm import gpt_oss

            gpt_oss.loader = GDSWeights(Path(model_dir) / "gds_export", device=self.device)
            gpt_oss.stats = self.stats
            self.model = gpt_oss.MyGptOssForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )
        else:
            from ollm import llama

            if self.model_id == "llama3-1B-chat":
                llama.loader = SingleDenseWeightsLoader(model_dir, device=self.device)
            else:
                llama.loader = DenseWeightsLoader(model_dir, device=self.device)
            llama.stats = self.stats
            self.model = llama.MyLlamaForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                attn_implementation=get_attn_implementation(),
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )

        self.model.eval()
        self.model.to(self.device)
        if not hasattr(self, "tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def offload_layers_to_cpu(self, **args) -> None:
        self.model.offload_layers_to_cpu(**args)

    def offload_layers_to_gpu_cpu(self, **args) -> None:
        self.model.offload_layers_to_gpu_cpu(**args)

    def DiskCache(self, cache_dir: str = "./kvcache"):
        if self.model_id == "gpt-oss-20B":
            print(f"{self.model_id} DiskCache is not supported at the moment. Using default DynamicCache instead")
            return None
        if self.model_id == "qwen3-next-80B":
            from ollm.qwen3_next import Qwen3NextDiskCache

            return Qwen3NextDiskCache(self.model.config, cache_dir=cache_dir, device=self.device, stats=self.stats)
        return KVCache(cache_dir=cache_dir, device=self.device, stats=self.stats)


class AutoInference(Inference):
    def __init__(
        self,
        model_dir: str,
        adapter_dir: str | None = None,
        device: str = "cuda:0",
        logging: bool = True,
        multimodality: bool = False,
    ):
        self.device = torch.device(device)
        self.stats = Stats() if logging else None
        self.multimodality = multimodality

        config = AutoConfig.from_pretrained(model_dir)
        architectures = getattr(config, "architectures", None) or ()
        architecture = architectures[0] if architectures else None
        if architecture == "LlamaForCausalLM":
            self.model_id = "llama3-8B-chat" if self.is_sharded(model_dir) else "llama3-1B-chat"
        elif architecture in {"Gemma3ForConditionalGeneration", "Gemma3ForCausalLM"}:
            self.model_id = "gemma3-12B"
        else:
            raise ValueError(
                f"The current generic path cannot run architecture {architecture!r}. "
                "Use a built-in optimized alias or a compatible local Llama/Gemma3 model directory."
            )

        self.load_model(model_dir)
        if adapter_dir:
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig.from_pretrained(adapter_dir)
            self.model = get_peft_model(self.model, peft_config)
            self.model.load_adapter(adapter_dir, adapter_name="default")

    def is_sharded(self, model_dir: str) -> bool:
        return any("index.json" in file.name for file in Path(model_dir).expanduser().resolve().iterdir())

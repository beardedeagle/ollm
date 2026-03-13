import os

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoProcessor, AutoConfig

from .utils import Stats
from .gds_loader import GDSWeights, DenseWeightsLoader, MoEWeightsLoader, SingleDenseWeightsLoader
from .kvcache import KVCache
from .runtime.catalog import get_model_catalog_entry, supported_model_ids


def get_attn_implementation():
	try:
		import flash_attn
		return "flash_attention_2"
	except ImportError:
		print("Warning: flash_attention_2 is not imported. The context length will be limited")
		return None


def download_model_snapshot(model_id, model_dir, force_download=False):
	entry = get_model_catalog_entry(model_id)
	print(f"Downloading {entry.repo_id} ...")
	snapshot_download(
		repo_id=entry.repo_id,
		local_dir=model_dir,
		local_dir_use_symlinks=False,
		force_download=force_download,
	)


class Inference:
	def __init__(self, model_id, device="cuda:0", logging=True, multimodality=False):
		self.model_id = model_id
		self.device = torch.device(device)
		self.multimodality = multimodality
		self.stats = Stats() if logging else None

	def hf_download(self, model_dir):
		download_model_snapshot(self.model_id, model_dir)

	
	def ini_model(self, models_dir="./models/", force_download=False):
		models_list = supported_model_ids()
		if self.model_id not in models_list:
			raise ValueError("Incorrect model id. It must be one of", models_list)
		
		model_dir = os.path.join(models_dir, self.model_id)
		if os.path.exists(model_dir)==False or force_download==True:
			self.hf_download(model_dir)

		self.load_model(model_dir)

	
	def load_model(self, model_dir):
		print("loading model from", model_dir)
		if self.model_id=="qwen3-next-80B":
			from . import qwen3_next
			qwen3_next.loader = MoEWeightsLoader(model_dir, device=self.device)
			qwen3_next.stats = self.stats
			self.model = qwen3_next.MyQwen3NextForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
		elif self.model_id=="gemma3-12B":
			from . import gemma3
			gemma3.loader = DenseWeightsLoader(model_dir, device=self.device)
			gemma3.stats = self.stats
			automodel = gemma3.MyGemma3ForConditionalGeneration if self.multimodality else gemma3.MyGemma3ForCausalLM
			self.model = automodel.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			self.processor = AutoProcessor.from_pretrained(model_dir)
		elif self.model_id=="voxtral-small-24B":
			from . import voxtral
			voxtral.loader = DenseWeightsLoader(model_dir, device=self.device)
			voxtral.stats = self.stats
			self.model = voxtral.MyVoxtralForConditionalGeneration.from_pretrained(model_dir, torch_dtype="auto", device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			self.processor = AutoProcessor.from_pretrained(model_dir)
			self.tokenizer = self.processor.tokenizer
		elif self.model_id=="gpt-oss-20B":
			from . import gpt_oss
			gpt_oss.loader = GDSWeights(os.path.join(model_dir, "gds_export"), device=self.device)
			gpt_oss.stats = self.stats
			self.model = gpt_oss.MyGptOssForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, ignore_mismatched_sizes=True)		
		else:
			from . import llama
			llama.loader = SingleDenseWeightsLoader(model_dir, device=self.device) if self.model_id in ["llama3-1B-chat"] else DenseWeightsLoader(model_dir, device=self.device)
			llama.stats = self.stats
			self.model = llama.MyLlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation=get_attn_implementation(), low_cpu_mem_usage=True, ignore_mismatched_sizes=True)

		self.model.eval()
		self.model.to(self.device)
		if not hasattr(self, "tokenizer"): self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

	
	def offload_layers_to_cpu(self, **args):
		self.model.offload_layers_to_cpu(**args)
	
	def offload_layers_to_gpu_cpu(self, **args):
		self.model.offload_layers_to_gpu_cpu(**args)
	
	def DiskCache(self, cache_dir="./kvcache"):
		if self.model_id in ["gpt-oss-20B"]:
			print(f"{self.model_id} DiskCache is not supported at the moment. Using default DynamicCache instead")
			return None
		elif self.model_id=="qwen3-next-80B":
			from .qwen3_next import Qwen3NextDiskCache
			return Qwen3NextDiskCache(self.model.config, cache_dir=cache_dir, device=self.device, stats=self.stats)
		else:
			return KVCache(cache_dir=cache_dir, device=self.device, stats=self.stats)



class AutoInference(Inference):
	def __init__(self, model_dir, adapter_dir=None, device="cuda:0", logging=True, multimodality=False):
		self.device = torch.device(device)
		self.stats = Stats() if logging else None
		config = AutoConfig.from_pretrained(model_dir)
		arc = config.architectures[0]
		sharded = self.is_sharded(model_dir)
		if arc == "LlamaForCausalLM":
			self.model_id = "llama3-8B-chat" if sharded else "llama3-1B-chat"
		elif arc == "Gemma3ForConditionalGeneration":
			self.model_id = "gemma3-12B"
			#multimodality = True
		elif arc == "Gemma3ForCausalLM": self.model_id = "gemma3-12B"
		else:
			raise ValueError("This model type is not supported")
			
		self.multimodality = multimodality
		self.load_model(model_dir) #peft_config.base_model_name_or_path
		if adapter_dir:
			from peft import LoraConfig, get_peft_model
			peft_config = LoraConfig.from_pretrained(adapter_dir)
			self.model = get_peft_model(self.model, peft_config)   # this creates LoRA modules with grad enabled
			self.model.load_adapter(adapter_dir, adapter_name="default")
			#self.model = self.model.model #?

	def is_sharded(self, model_dir):
		files = os.listdir(model_dir)
		is_sharded = any("index.json" in f for f in files)
		return is_sharded

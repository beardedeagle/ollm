from datetime import datetime
from typing import Protocol, cast

import torch
from transformers import AutoProcessor, AutoTokenizer, TextStreamer

from ollm import gds_loader, gemma3, qwen3_next, voxtral
from ollm.gds_loader import DenseWeightsLoader, MoEWeightsLoader


class _GenerativeModelProtocol(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...
    def offload_layers_to_cpu(self, layers_num: int = 2) -> None: ...
    def eval(self) -> object: ...
    def to(self, device: torch.device) -> object: ...
    @property
    def config(self) -> object: ...


class _ChatTokenizerProtocol(Protocol):
    def apply_chat_template(self, *args: object, **kwargs: object) -> torch.Tensor: ...
    def decode(self, *args: object, **kwargs: object) -> str: ...


def ini_model(model_id):
    if model_id == "qwen3-next-80B":
        o, CausalLM = qwen3_next, qwen3_next.MyQwen3NextForCausalLM
        setattr(o, "loader", MoEWeightsLoader(model_dir, device=str(device)))
    elif model_id == "gemma3-12B":
        o, CausalLM = gemma3, gemma3.MyGemma3ForCausalLM
        setattr(o, "loader", DenseWeightsLoader(model_dir, device=str(device)))
    elif model_id == "voxtral-small-24B":
        o, CausalLM = voxtral, voxtral.MyVoxtralForConditionalGeneration
        setattr(o, "loader", DenseWeightsLoader(model_dir, device=str(device)))

    stats = None  # Stats()
    setattr(o, "stats", stats)
    gds_loader.stats = stats
    return (o, CausalLM)


def inference_chat():
    sm, um, max_new_tokens = "您是有用的人工智能助手", "列出从水星开始的行星", 100
    # sm, um, max_new_tokens = "You are helpful AI assistant", "List planets starting from Mercury", 100
    messages = [{"role": "system", "content": sm}, {"role": "user", "content": um}]
    tokenizer = cast(_ChatTokenizerProtocol, AutoTokenizer.from_pretrained(model_dir))
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        reasoning_effort="minimal",
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    ).to(device)
    text_streamer = TextStreamer(
        cast(AutoTokenizer, AutoTokenizer.from_pretrained(model_dir)),
        skip_prompt=True,
        skip_special_tokens=False,
    )
    typed_model = model
    with torch.no_grad():
        # past_key_values = KVCache(cache_dir="/media/mega4alik/ssd/kv_cache/", stats=o.stats, device=device) #DynamicCache(offloading=True)
        past_key_values = qwen3_next.Qwen3NextDiskCache(
            model.config,
            cache_dir="/media/mega4alik/ssd/kv_cache/",
            device=device,
            stats=o.stats,
        )
        print(
            "\n\nGenerate started.",
            datetime.now().strftime("%H:%M:%S"),
            "input_ids.shape:",
            input_ids.shape,
        )
        outputs = (
            typed_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                past_key_values=past_key_values,
                use_cache=True,
                streamer=text_streamer,
            )
            .detach()
            .cpu()
        )
        answer = tokenizer.decode(
            outputs[0][input_ids.shape[-1] :], skip_special_tokens=False
        )
        print(answer)


def inference_audio():
    processor = AutoProcessor.from_pretrained(model_dir)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
                },
                {"type": "text", "text": "What can you tell me about this audio?"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(messages, return_tensors="pt").to(device)
    text_streamer = TextStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=False
    )
    with torch.inference_mode():
        print("\n\nAudio Generate started.", datetime.now().strftime("%H:%M:%S"))
        outputs = (
            model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                past_key_values=None,
                use_cache=True,
                streamer=text_streamer,
            )
            .detach()
            .cpu()
        )
        answer = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=False
        )
        print(answer)


# =======================================================
if __name__ == "__main__":
    device = torch.device("cuda:0")
    model_id = "qwen3-next-80B"  # "gemma3-12B" # #"voxtral-small-24B"
    model_dir = f"/media/mega4alik/ssd/models/{model_id}/"
    print("loading", model_dir)
    o, CausalLM = ini_model(model_id)
    model = cast(
        _GenerativeModelProtocol,
        CausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            attn_implementation="flash_attention_2",
        ),
    )
    # model.clean_layers_weights()
    # model.offload_layers_to_gpu_cpu(gpu_layers_num=48, cpu_layers_num=0)
    model.offload_layers_to_cpu(layers_num=1)
    model.eval()
    model.to(device)
    inference_chat()

"""Fixture helpers for runtime benchmark planner scenarios."""

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


def create_tiny_t5_fixture(root: Path) -> Path:
    """Create a tiny local T5 fixture for planner-only benchmark scenarios."""

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from transformers.models.t5 import T5Config, T5ForConditionalGeneration

    model_dir = root / "tiny-t5"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "<pad>": 0,
        "</s>": 1,
        "<unk>": 2,
        "hello": 3,
        "world": 4,
        "benchmark": 5,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast_tokenizer.save_pretrained(model_dir)
    config = T5Config(
        vocab_size=len(vocab),
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        decoder_start_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        T5ForConditionalGeneration(config).save_pretrained(
            model_dir, safe_serialization=True
        )
    return model_dir

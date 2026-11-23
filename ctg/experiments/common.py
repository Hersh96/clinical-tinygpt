"""Helpers shared by the experiment scripts."""
from __future__ import annotations

import torch

from ctg.model.gpt import GPT
from ctg.train.loop import build_model_config

# Fixed prompts used for every sample shown on the website.
PROMPTS = ["The patient is a 67-year-old man with", "She was started on metformin",
           "Physical examination revealed", "The patient denies", "On discharge, the patient was"]


def load_checkpoint(path: str, device) -> GPT:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = GPT(build_model_config(ck["model_config"])).to(device).eval()
    model.load_state_dict(ck["model"])
    return model


def gpt2_tokenizer():
    """GPT-2's tokenizer from the Hugging Face Hub: (encode, decode_one, eot_id)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = 10**9  # we only count tokens; silence the 1024-token warning
    return (lambda s: tok.encode(s, add_special_tokens=False)), (lambda i: tok.decode([i])), tok.eos_token_id

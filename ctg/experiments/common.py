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

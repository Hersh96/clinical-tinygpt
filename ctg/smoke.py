"""Smoke test of the whole Part 1 pipeline on the tiny test fixtures (CPU, seconds).

fixtures -> BPE (vocab 300) -> token files -> train a tiny model -> generate.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ctg.data.tokens import encode_split
from ctg.model.generate import generate
from ctg.model.gpt import GPT
from ctg.tokenizer.bpe import BPETokenizer
from ctg.train.loop import build_model_config, load_config, train


def main() -> None:
    out = Path("build/smoke")
    out.mkdir(parents=True, exist_ok=True)
    notes = [json.loads(l)["note"] for l in open("tests/fixtures/notes.jsonl")]
    tok = BPETokenizer()
    tok.train(notes, 300)
    jsonl = out / "all.jsonl"
    jsonl.write_text("".join(json.dumps({"text": n}) + "\n" for n in notes * 5))
    for split in ("train", "val"):
        encode_split(jsonl, tok, out / f"{split}.bin")
    cfg = load_config("configs/smoke.yaml")
    res = train(cfg)
    print("train:", res)
    ck = torch.load(Path(cfg["ckpt_dir"]) / cfg["run_name"] / "last.pt", weights_only=False)
    model = GPT(build_model_config(ck["model_config"])).eval()
    model.load_state_dict(ck["model"])
    prompt = torch.tensor([tok.encode("Patient 3 is a")])
    print("sample:", tok.decode(generate(model, prompt, 30, temperature=0)[0].tolist()))


if __name__ == "__main__":
    main()

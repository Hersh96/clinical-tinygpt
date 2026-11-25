# Clinical TinyGPT

A small GPT-style language model built from scratch on clinical text, to learn how LLMs work inside. Every core piece is hand-written in PyTorch and every concept has a measured result:

| Concept | What's built | What's measured |
| --- | --- | --- |
| Tokenization | Byte-level BPE trainer and encoder (`ctg/tokenizer/bpe.py`) | Tokens per word at 4k / 8k / 16k vocab vs GPT-2 |
| Transformer | Decoder-only GPT with RoPE, RMSNorm, SwiGLU, tied embeddings (`ctg/model/`) | Attention patterns on a real sentence |
| Training | AdamW, warmup + cosine LR, gradient accumulation, bf16, checkpoint/resume (`ctg/train/`) | Loss curves for 4 model sizes |
| Scaling | Sweep over ~5M, 15M, 40M, 85M parameters (`ctg/experiments/scaling.py`) | Power-law fit of loss vs size |
| KV cache | Hand-written cache and generation loop (`ctg/model/kv_cache.py`, `generate.py`) | Tokens/s with and without the cache |
| Evaluation | Bits per byte, comparable across tokenizers (`ctg/experiments/eval_bpb.py`) | Ours vs GPT-2 small on held-out notes |
| MLOps | YAML configs, `make` targets, MLflow tracking, CI | Every run reproducible from its config |

The results are presented as an interactive guide at [hershshukla.com/projects/clinical-tinygpt](https://hershshukla.com/projects/clinical-tinygpt/index.html) (source in `site/`).

> **Not for clinical use.** This is a learning project trained on public, de-identified case reports. Its outputs are synthetic text.

## Data

- [AGBonnet/augmented-clinical-notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes): about 30k de-identified clinical case notes (MIT license).
- PubMed abstracts from [ccdv/pubmed-summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization), capped at about 80M tokens. The dataset card states no license, so it's used for training only and no text is redistributed.

Documents are split 90/5/5 into train, validation and test by a hash of their ID (`ctg/data/prepare.py`), so the split is identical on every machine. The licenses used are written to `data/processed/LICENSES.md`.

## Setup (Windows laptop, WSL2, RTX 4050 6 GB)

1. On Windows, install a current NVIDIA driver. WSL2 uses the Windows driver, so don't install a Linux one.
2. Install WSL2 with Ubuntu (24.04 recommended): `wsl --install -d Ubuntu-24.04`. Python 3.12+ is required; the setup script installs it on older Ubuntu.
3. In Ubuntu, clone this repo and run:

```bash
bash scripts/setup_wsl2.sh      # creates .venv and prints "cuda available: True"
source .venv/bin/activate
```

## Run everything, in order

```bash
make test               # unit tests (CPU, seconds)
make smoke              # whole pipeline on tiny fixtures (CPU, seconds)
make data               # download + clean + split (~500 MB of text)
make tokenizer          # train BPE at 4k, 8k, 16k
make tokenizer-compare  # tokens per word vs GPT-2
make tokens             # encode the corpus with the 8k tokenizer
make sweep              # train 4 sizes (hours; resumable: re-run to continue)
make bench              # KV cache benchmark
make eval               # bits per byte vs GPT-2 + text samples
make export             # write the website's JSON from the results
```

Browse every run with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

If the laptop sleeps or shuts down during `make sweep`, run it again. Each model resumes from `ckpt/<run>/last.pt` at the same step, learning rate and data position.

## Preview the guide page locally

```bash
make export                      # refresh site data from your runs
python -m http.server -d site/public 8000
# open http://localhost:8000/projects/clinical-tinygpt/index.html
```

The committed data comes from the full runs on the laptop GPU (Nov 2026). Re-running `make export` after new runs refreshes it.

## Learn the code

Read in this order. Each file opens with a plain-English explanation of its concept.

1. `ctg/tokenizer/bpe.py`: text to tokens
2. `ctg/model/layers.py`: RMSNorm, RoPE, attention, SwiGLU
3. `ctg/model/gpt.py`: the full model
4. `ctg/model/kv_cache.py` and `ctg/model/generate.py`: fast generation
5. `ctg/train/schedule.py` and `ctg/train/loop.py`: training
6. `ctg/experiments/`: the measurements behind every chart

## Repository layout

```
ctg/            the Python package (data, tokenizer, model, train, experiments, export)
configs/        one YAML per experiment
tests/          pytest suite (CPU)
site/           the interactive guide page and its tests (site/public/projects/ml-guide.css is a copy of the website's stylesheet, for local testing only)
docs/superpowers/  design spec and implementation plan
```

## Part 2 (next)

Part 2 fine-tunes SmolLM2 (360M with LoRA, 1.7B with 4-bit QLoRA) to summarize the same notes. It compares against the zero-shot baseline, checks for made-up facts, and measures quantized speed. Its results get added to the same guide page.

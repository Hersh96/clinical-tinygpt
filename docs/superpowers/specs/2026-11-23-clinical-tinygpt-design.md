# Clinical TinyGPT: LLM internals and fine-tuning design

## Intent and why
Hersh wants one project that teaches him how LLMs work inside by building them: tokenization, the transformer, the KV cache, scaling, fine-tuning and MLOps. Training runs on his own laptop. The finished work goes on hershshukla.com as a results page with the repo link and charts of what was measured.
Two parts in one repo, built in order. Part 1 trains a small GPT from scratch on clinical text. Part 2 fine-tunes SmolLM2 on the same text. Clinical text ties it to his existing ClinicalBERT, adherence and readmission portfolio work.

## Goals, non-goals, anti-goals
Goals: Hersh writes core pieces himself (BPE tokenizer, attention, KV cache, training loop); every concept yields a measured result that becomes a chart; everything trains on an RTX 4050 laptop (6 GB VRAM) in hours; Part 1 ships to the site before Part 2 starts; repo shows real engineering (tracked runs, tests, configs, reproducible scripts).
Non-goals: live chatbot/demo on the website; beating published clinical LLMs; multi-GPU/cloud training.
Anti-goals: copying nanochat or a course notebook wholesale; charts that compare nothing; credentialed data like MIMIC or implying clinical fitness.

## Constraints
- RTX 4050 laptop 6 GB: Part 1 models <= ~85M params, 512 ctx, bf16. Part 2: SmolLM2-360M LoRA; SmolLM2-1.7B only 4-bit QLoRA.
- Laptop thermals: single run < ~3h, resumable from checkpoint.
- Public data only, licenses allow publishing results.
- Site is static results page following existing guides at /projects/<slug>/index.html with ml-guide.css and PNG charts (React/TS portfolio, entries in src/data/portfolio.ts). No server-side model.
- One person learning: small steps; Hersh writes core code; libraries only where the concept isn't the point.

## Approaches
Picked: 3 = both (from-scratch GPT nanochat-style + LoRA/QLoRA SmolLM2 fine-tune), Part 1 first. Rejected alone: 1 (no real-model fine-tuning), 2 (transformer black box). Raschka LLMs-from-scratch = reference only.

## Design
Repo `clinical-tinygpt`, part1_scratch/ and part2_finetune/ sharing one data step and one MLflow store. Each concept = one module + one experiment + one chart.

Data: AGBonnet/augmented-clinical-notes (~30k notes with summaries) plus PubMed abstracts subset to reach ~100M tokens for pretraining; licenses checked. Fixed 90/5/5 split; summarization test set unseen in both parts.

Part 1:
1. Hand-written byte-level BPE, trained at 4k/8k/16k vocab; compare to GPT-2 tokenizer on tokens/word on clinical text and drug-name/abbreviation splits.
2. Decoder-only transformer in PyTorch: causal self-attention, RoPE, RMSNorm, SwiGLU; unit tests for shape and causality.
3. Training loop: AdamW, warmup+cosine, grad accumulation, bf16, checkpoint/resume, MLflow logging.
4. Scaling sweep: 4 sizes (~5M, 15M, 40M, 85M) at fixed token budget; power-law fit of val loss vs params.
5. Generation with and without hand-written KV cache: tokens/s and memory vs output length.
6. Eval: perplexity on held-out clinical text vs GPT-2 small; a few samples shown as-is.

Part 2:
1. Baseline zero-shot SmolLM2-360M-Instruct on note summarization.
2. LoRA SFT (TRL + PEFT) note->summary, rank sweep 4/8/16/32.
3. QLoRA SmolLM2-1.7B 4-bit vs 360M on quality, VRAM, train time.
4. Eval: ROUGE-L, BERTScore on test; 50-example hand check for hallucinated meds/conditions.
5. Merge adapter, quantize to GGUF Q8/Q4, measure tokens/s and quality loss with llama.cpp.
6. Publish adapter + model card on HF Hub, not-for-clinical-use note.

MLOps: YAML config per experiment, make targets (make tokenizer, make sweep, make finetune); MLflow for every run; site charts generated from MLflow exports by one script; pinned deps, seeds; GitHub Actions CPU unit tests + tiny smoke train on push.

Charts: tokens/word own BPE x3 vs GPT-2; train/val loss for 4 sizes; log-log val loss vs params with fit; tokens/s vs output length with/without cache; ROUGE-L baseline vs LoRA ranks vs QLoRA 1.7B; VRAM & train time per method; GGUF speed vs quality.

Website: Part 1 ships first as /projects/clinical-tinygpt/index.html in existing guide style + new portfolio.ts entry; Part 2 charts added to same page later. Page: repo link, short explanation per concept with its chart, results summary at top.

## Calls made (Claude's, unconfirmed)
Summarization as Part 2 task; datasets above pending license check; model/vocab sizes and 100M token budget; MLflow over W&B; repo name.
Left to builder: exact hyperparameters, chart styling within ml-guide.css, folder layout inside each part, unit tests beyond shape/causality.

## Updates after Hersh's answers (2026-11-23, approved)
- Claude writes all the code (Hersh has never done this). Core pieces are hand-written, not imported, and heavily commented for learning.
- Laptop runs Windows; everything runs under WSL2 (Ubuntu) with CUDA.
- Fallback dataset if augmented-clinical-notes license fails: ccdv/pubmed-summarization.
- Website: an interactive, novice-friendly guide in the hershshukla.com theme (ml-guide.css, light/dark), static HTML+JS at /projects/clinical-tinygpt/index.html plus a portfolio.ts entry.
  - Start here: summary, headline numbers, clickable pipeline map (text > tokens > model > training > generation > fine-tuning > deployment).
  - One chapter per concept: plain explanation, chart of the real result, "what this means", link to the code file.
  - Progress rail sidebar with read state and glossary on hover.
  - Interactive pieces: in-browser tokenizer playground (4k/8k/16k vs GPT-2, from exported merges); transformer step-through (embeddings, real attention heatmap, next-token probabilities); toggleable loss curves; scaling slider on the fitted curve; KV cache animation + measured speed; fine-tuning side-by-side summaries with hallucinations highlighted (precomputed); filterable efficiency charts.
  - Part 1 chapters ship first; Part 2 chapters added later. All chart data is JSON from the MLflow export script.

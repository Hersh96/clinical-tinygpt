# Clinical TinyGPT, Part 1 (from scratch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small GPT from scratch on clinical text (own BPE tokenizer, transformer, KV cache, training loop, scaling sweep), track it in MLflow, and ship an interactive novice-friendly guide page for it on hershshukla.com.

**Architecture:** One Python package `ctg` in a new repo `clinical-tinygpt`. Each concept is one module plus one experiment script; every experiment logs to a local MLflow store, and one export script turns MLflow runs into JSON that a static HTML/JS guide page reads. Part 2 (SmolLM2 fine-tuning) gets its own plan after Part 1 ships; it reuses `ctg/data`, MLflow and the export script.

**Tech Stack:** Python 3.11, PyTorch 2.x (CUDA 12, WSL2 Ubuntu), `datasets`, `regex`, `numpy`, `scipy`, `mlflow`, `pyyaml`, `tiktoken` (GPT-2 baseline only), `transformers` (GPT-2 small baseline only), `pytest`. Site: vanilla HTML/CSS/JS, Chart.js 4.4.1 vendored, Node 20 for JS tests, Playwright for page checks.

**Spec:** `docs/superpowers/specs/2026-11-23-clinical-tinygpt-design.md` (live copy: Claude Doc "Clinical TinyGPT: LLM internals and fine-tuning design").

## Global Constraints

- Hardware target: RTX 4050 laptop, 6 GB VRAM. Every Part 1 run fits in 6 GB with bf16 and 512-token context.
- Any single training run finishes in under ~3 hours and resumes from a checkpoint.
- Model sizes: about 5M, 15M, 40M, 85M parameters (non-embedding counts within ±15%).
- Tokenizer vocab sizes: 4,096, 8,192, 16,384. Pretraining token budget: ~100M tokens.
- Data: public Hugging Face datasets only. Primary `AGBonnet/augmented-clinical-notes` + PubMed abstracts subset; no MIMIC.
- Split: 90/5/5 train/val/test, deterministic by document id; test set is never trained on.
- Core pieces (BPE, attention, RoPE, RMSNorm, SwiGLU, KV cache, training loop, LR schedule) are hand-written in `ctg/`, never imported from `transformers`/`tokenizers`. Every core file opens with a plain-English docstring explaining the concept.
- Every experiment is a YAML config + `make` target; every run logs to MLflow (`./mlruns`); every config has a `seed`.
- Dependencies pinned in `requirements.lock`.
- Model card / page copy states the model is not for clinical use.
- Site page: static, same theme as the site (`/projects/ml-guide.css`, light and dark), path `/projects/clinical-tinygpt/index.html`, works at phone width with no horizontal scroll, no server-side model.

## Review Focus

1. **Text with characters outside ASCII** (µg, °, en dashes, emoji pasted into the playground): BPE must round-trip any UTF-8 string byte-exactly, in Python and in the browser. Test in Task 2 and Task 10.
2. **A training run killed mid-way** (laptop sleep, thermal shutdown): resume must continue from the same step, LR and data position, not restart. Test in Task 5.
3. **Generation longer than the context window** (512): the cached and uncached paths must both crop/slide the same way, not crash or silently diverge. Test in Task 4.
4. **Comparing perplexity across different tokenizers:** per-token perplexity is not comparable between our BPE and GPT-2; use bits per byte. Test in Task 7.
5. **Page opened with no JavaScript data loaded yet or a missing JSON file:** chapters show a readable "data not available" message, not a blank chart. Test in Task 10.

---

## File Structure

```
clinical-tinygpt/
  pyproject.toml, requirements.lock, Makefile, README.md, .github/workflows/ci.yml
  scripts/setup_wsl2.sh                 # CUDA/WSL2 environment setup with checks
  configs/data.yaml, tokenizer.yaml, model_{5m,15m,40m,85m}.yaml, smoke.yaml
  ctg/
    data/prepare.py                     # download, clean, split -> jsonl
    data/tokens.py                      # encode corpus -> uint16 .bin, batch sampler
    tokenizer/bpe.py                    # byte-level BPE (train/encode/decode/save/load)
    model/config.py                     # GPTConfig + size presets
    model/layers.py                     # RMSNorm, RoPE, CausalSelfAttention, SwiGLU, Block
    model/gpt.py                        # GPT model
    model/kv_cache.py                   # KVCache
    model/generate.py                   # generate(), with/without cache
    train/schedule.py                   # lr_at()
    train/loop.py                       # train(cfg) with checkpoint/resume + MLflow
    experiments/tokenizer_compare.py    # tokens/word vs GPT-2
    experiments/scaling.py              # sweep driver + fit_power_law()
    experiments/bench_kv.py             # tokens/s + memory vs length
    experiments/eval_bpb.py             # bits per byte vs GPT-2 small
    export/site_data.py                 # MLflow + artifacts -> site JSON
  tests/ (one test file per module above)
site/  (copied into the portfolio repo by Task 10)
  public/projects/clinical-tinygpt/index.html, guide.css, guide.js, bpe.js,
    vendor/chart.umd.min.js, data/*.json
  tests/bpe.test.mjs, tests/page.spec.mjs
```

---

### Task 1: Repo scaffold, environment and data split

**Files:**
- Create: `pyproject.toml`, `requirements.lock`, `Makefile`, `scripts/setup_wsl2.sh`, `configs/data.yaml`, `ctg/data/prepare.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Produces: `prepare_corpus(cfg: dict, out_dir: Path) -> dict[str, Path]` returning `{"train","val","test"}` jsonl paths; each line `{"id": str, "source": str, "text": str, "summary": str|None}`. `split_for(doc_id: str, seed: int) -> Literal["train","val","test"]`.

- [ ] **Step 1: Write failing tests**

```python
def test_split_is_deterministic_and_proportional():
    ids = [f"doc{i}" for i in range(20000)]
    a = [split_for(i, seed=0) for i in ids]; b = [split_for(i, seed=0) for i in ids]
    assert a == b
    assert abs(a.count("train")/len(a) - 0.90) < 0.01
    assert abs(a.count("val")/len(a) - 0.05) < 0.01

def test_prepare_has_no_overlap(tmp_path, tiny_fixture_cfg):
    paths = prepare_corpus(tiny_fixture_cfg, tmp_path)
    ids = {k: {json.loads(l)["id"] for l in open(p)} for k, p in paths.items()}
    assert not (ids["train"] & ids["test"]) and not (ids["train"] & ids["val"])

def test_clean_normalizes_whitespace_keeps_unicode():
    assert clean_text("BP  120/80\r\n\r\n\r\nµg dose") == "BP 120/80\n\nµg dose"
```
`tiny_fixture_cfg` points at `tests/fixtures/notes.jsonl` (20 hand-written synthetic notes) instead of the Hub.

- [ ] **Step 2: Run** `pytest tests/test_prepare.py -v` — Expected: FAIL (module not found).
- [ ] **Step 3: Implement** `split_for` (sha256 of `f"{seed}:{doc_id}"` mod 100: <90 train, <95 val, else test), `clean_text(s: str) -> str`, `prepare_corpus`. `configs/data.yaml` lists `AGBonnet/augmented-clinical-notes` (fields `note`, `summary`) and a PubMed abstracts source capped so total ≈100M GPT-2 tokens; record each dataset's license string into `data/LICENSES.md` and stop with a clear error if a license field is missing. `setup_wsl2.sh` checks `nvidia-smi`, installs Python 3.11 venv and `requirements.lock`, and prints `torch.cuda.is_available()`.
- [ ] **Step 4: Run** `pytest tests/test_prepare.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat: scaffold repo and deterministic data split`.

### Task 2: Hand-written byte-level BPE tokenizer

**Files:** Create `ctg/tokenizer/bpe.py`, `configs/tokenizer.yaml`; Test `tests/test_bpe.py`

**Interfaces:**
- Produces: `class BPETokenizer` with `train(texts: Iterable[str], vocab_size: int) -> None`, `encode(text: str) -> list[int]`, `decode(ids: list[int]) -> str`, `save(path: Path) -> None` (writes `merges.json`: `{"vocab_size": int, "pattern": str, "merges": [[int,int],...], "special": {"<|endoftext|>": int}}`), `@classmethod load(path) -> BPETokenizer`. Base vocab = 256 bytes; merge i creates id 256+i; `<|endoftext|>` = last id.

- [ ] **Step 1: Write failing tests**

```python
CASES = ["", "a", "Pt denies chest pain.", "Metformin 500 mg PO BID", "µg °C – naïve 🙂", "x" * 5000]
def test_roundtrip_any_utf8(trained_tok):
    for s in CASES: assert trained_tok.decode(trained_tok.encode(s)) == s
def test_vocab_size_exact(corpus):
    t = BPETokenizer(); t.train(corpus, vocab_size=300); assert len(t.vocab) == 300
def test_training_is_deterministic(corpus):
    a, b = BPETokenizer(), BPETokenizer(); a.train(corpus, 300); b.train(corpus, 300)
    assert a.merges == b.merges
def test_first_merge_is_most_frequent_pair():
    t = BPETokenizer(); t.train(["aaab aaab aaab"], 257); assert t.merges[0] == (97, 97)
def test_save_load_identical(tmp_path, trained_tok):
    trained_tok.save(tmp_path/"m.json"); t2 = BPETokenizer.load(tmp_path/"m.json")
    assert t2.encode("Lisinopril 10 mg daily") == trained_tok.encode("Lisinopril 10 mg daily")
def test_endoftext_is_single_token(trained_tok):
    assert trained_tok.encode("<|endoftext|>", allow_special=True) == [trained_tok.special["<|endoftext|>"]]
```

- [ ] **Step 2: Run** `pytest tests/test_bpe.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement.** Pre-split with the GPT-2 regex (`regex` module) so merges never cross word boundaries; count pairs per unique chunk weighted by chunk frequency (fast enough for ~10M chars of sample text); ties broken by smallest pair. Train on a 50 MB sample of train split. `encode` applies merges by lowest merge rank first.
- [ ] **Step 4: Run** tests — Expected: PASS. Then `make tokenizer` trains 4k/8k/16k and writes `artifacts/tokenizer/{4096,8192,16384}/merges.json`.
- [ ] **Step 5: Commit** `feat: hand-written byte-level BPE`.

### Task 3: Tokenizer comparison + token dataset

**Files:** Create `ctg/experiments/tokenizer_compare.py`, `ctg/data/tokens.py`; Test `tests/test_tokens.py`, `tests/test_tokenizer_compare.py`

**Interfaces:**
- Consumes: `BPETokenizer.load`, jsonl from Task 1.
- Produces: `compare(tokenizers: dict[str, Callable[[str], list[int]]], texts: list[str]) -> dict[str, dict]` with keys `tokens_per_word`, `chars_per_token`, and `examples` (10 fixed clinical terms → token strings). `encode_split(jsonl: Path, tok: BPETokenizer, out: Path) -> int` writes uint16 `.bin` with `<|endoftext|>` between docs and returns the token count. `class TokenStream(path: Path, block_size: int, seed: int)` with `batch(batch_size: int, device) -> tuple[Tensor, Tensor]` and `state_dict()/load_state_dict()` (RNG state, for resume).

- [ ] **Step 1: Tests:** `compare` on `["a b c"]` with a char tokenizer gives `tokens_per_word == 5/3`; `encode_split` round-trips doc text through `decode` with separators; `TokenStream.batch` returns `y == x` shifted by one and dtype `int64`; two streams with the same seed and loaded state give identical batches.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement**; the example terms are fixed in code: `["metformin", "hydrochlorothiazide", "acetaminophen", "bid", "prn", "dyspnea", "tachycardia", "hba1c", "mg/dL", "status post"]`. Results log to MLflow run `tokenizer_compare` as `tokenizer_compare.json`.
- [ ] **Step 4: Run** — PASS. **Step 5: Commit** `feat: tokenizer comparison and token stream`.

### Task 4: Transformer model, KV cache and generation

**Files:** Create `ctg/model/{config,layers,gpt,kv_cache,generate}.py`, `configs/model_{5m,15m,40m,85m}.yaml`; Test `tests/test_model.py`, `tests/test_generate.py`

**Interfaces:**
- Produces: `@dataclass GPTConfig(vocab_size: int, n_layer: int, n_head: int, n_embd: int, block_size: int = 512, dropout: float = 0.0)`; `PRESETS: dict[str, GPTConfig]` for `"5m","15m","40m","85m"` (vocab 8192). `class GPT(nn.Module)`: `forward(idx: LongTensor[B,T], targets: LongTensor|None = None, cache: KVCache|None = None, start_pos: int = 0) -> tuple[Tensor, Tensor|None]`, `num_params(non_embedding=True) -> int`, `forward(..., return_attn=True)` additionally returns per-layer attention weights (used only by the site export). Weight tying between token embedding and LM head. `class KVCache(n_layer, batch, n_head, head_dim, max_len, device, dtype)`; `generate(model, idx, max_new_tokens: int, temperature=1.0, top_k: int|None = None, use_cache=True, seed: int|None = None) -> LongTensor`.

- [ ] **Step 1: Tests**

```python
def test_logits_shape(): m = GPT(PRESETS["5m"]); x = torch.randint(0, 8192, (2, 16)); assert m(x)[0].shape == (2, 16, 8192)
def test_causal_future_does_not_change_past():
    m = GPT(small_cfg).eval(); x = torch.randint(0, 100, (1, 10)); y = x.clone(); y[0, 7:] = 3
    assert torch.allclose(m(x)[0][:, :7], m(y)[0][:, :7], atol=1e-5)
@pytest.mark.parametrize("name,target", [("5m",5e6),("15m",15e6),("40m",40e6),("85m",85e6)])
def test_preset_sizes(name, target): assert abs(GPT(PRESETS[name]).num_params() - target) / target < 0.15
def test_loss_at_init_near_log_vocab(): _, loss = GPT(small_cfg)(x, x); assert abs(loss.item() - math.log(small_cfg.vocab_size)) < 0.5
def test_cached_greedy_equals_uncached():
    a = generate(m, prompt, 40, temperature=0, use_cache=True); b = generate(m, prompt, 40, temperature=0, use_cache=False)
    assert torch.equal(a, b)
def test_generation_past_block_size(): out = generate(m_block32, prompt_len30, 50, temperature=0, use_cache=True); assert out.shape[1] == 80
def test_cached_matches_uncached_past_block_size(): # both crop to the last block_size tokens the same way
    assert torch.equal(generate(m_block32, p, 50, temperature=0, use_cache=True), generate(m_block32, p, 50, temperature=0, use_cache=False))
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.** Pre-norm blocks: RMSNorm → attention (RoPE on q,k; use `F.scaled_dot_product_attention(is_causal=True)` for training and an explicit softmax path when `return_attn=True`) → RMSNorm → SwiGLU (hidden = 8/3·n_embd rounded to 64). When the sequence would exceed `block_size` with a cache, clear the cache and re-prefill the last `block_size // 2` tokens; the uncached path crops to that same window, so both agree.
- [ ] **Step 4: Run** — PASS. **Step 5: Commit** `feat: GPT with RoPE, RMSNorm, SwiGLU and KV cache`.

### Task 5: Training loop with LR schedule, checkpoint/resume and MLflow

**Files:** Create `ctg/train/schedule.py`, `ctg/train/loop.py`, `configs/smoke.yaml`; Test `tests/test_schedule.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `GPT`, `PRESETS`, `TokenStream`.
- Produces: `lr_at(step: int, max_lr: float, warmup: int, total: int, min_lr_ratio: float = 0.1) -> float`. `train(cfg: dict) -> dict` returning `{"run_id", "final_val_loss", "tokens_seen", "params"}`. Checkpoints at `ckpt/<run_name>/last.pt` with model, optimizer, step, stream state, RNG states. MLflow metrics: `train_loss`, `val_loss`, `lr`, `tokens_per_sec`, `gpu_mem_gb`; params: whole config.

- [ ] **Step 1: Tests:** `lr_at(0,…)==0`, `lr_at(warmup,…)==max_lr`, `lr_at(total,…)==max_lr*0.1`, monotone decreasing after warmup; smoke config (tiny model, 200 steps, CPU) reduces train loss on a repeated batch by >50%; **resume test:** run 100 steps, kill at step 60 (simulated via `max_steps=60` then a second call with `resume=True, max_steps=100`), and the final loss and weights equal an uninterrupted 100-step run (`atol=1e-6`, CPU, deterministic).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** AdamW (betas 0.9/0.95, weight decay 0.1 on 2-D params only), grad clip 1.0, grad accumulation to reach ~0.25M tokens/step, bf16 autocast on CUDA, val every N steps on a fixed 50-batch val slice, checkpoint every 10 minutes and at exit (including SIGINT/SIGTERM).
- [ ] **Step 4: Run** — PASS. `make smoke` runs `configs/smoke.yaml` end to end in <2 min on CPU.
- [ ] **Step 5: Commit** `feat: training loop with resume and MLflow`.

### Task 6: Scaling sweep and power-law fit

**Files:** Create `ctg/experiments/scaling.py`; Test `tests/test_scaling.py`

**Interfaces:**
- Consumes: `train`, MLflow runs tagged `sweep=scaling_v1`.
- Produces: `fit_power_law(n_params: list[float], losses: list[float]) -> dict` with `{"a","alpha","c","r2"}` for `L(N) = a·N^(-alpha) + c`; `predict(fit: dict, n: float) -> float`. Logs `scaling_fit.json` to a parent MLflow run.

- [ ] **Step 1: Tests:** fitting synthetic points from `a=400, alpha=0.3, c=1.8` with 1% noise recovers alpha within ±0.05 and r2 > 0.98; `predict` is monotone decreasing in n.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** with `scipy.optimize.curve_fit` (bounds keep alpha in (0,1), c ≥ 0). `make sweep` trains the four presets in sequence at the same token budget (~100M tokens; reduce for 85M only if the 3-hour cap would be broken, and record the actual tokens in MLflow), each resumable.
- [ ] **Step 4: Run** — PASS. **Step 5: Commit** `feat: scaling sweep and power-law fit`.

### Task 7: KV cache benchmark and bits-per-byte evaluation

**Files:** Create `ctg/experiments/bench_kv.py`, `ctg/experiments/eval_bpb.py`; Test `tests/test_eval.py`

**Interfaces:**
- Consumes: trained checkpoints, `generate`, GPT-2 small via `transformers` (baseline only).
- Produces: `bench(model, lengths: list[int] = [64,128,256,512,1024], repeats=3) -> list[dict]` rows `{"length","cache","tokens_per_sec","peak_mem_mb"}`. `bits_per_byte(nll_sum_nats: float, n_bytes: int) -> float = nll_sum_nats / (ln 2 · n_bytes)`; `eval_bpb(model_fn, texts) -> float`. Also writes 5 fixed-prompt samples per model (temperature 0.8, seed 0) to `samples.json`.

- [ ] **Step 1: Tests:** `bits_per_byte(math.log(2)*800, 100) == 8.0`; a uniform model over 256 bytes scores 8.0 bpb on any text; `bench` on the tiny model returns one row per (length, cache) pair and cache=True is not slower than cache=False at the longest length.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement**; both models score the same test-split texts, so the number is comparable across tokenizers. **Step 4: Run** — PASS. **Step 5: Commit** `feat: KV cache benchmark and bpb eval`.

### Task 8: Run the experiments on the laptop

Runs on Hersh's laptop (WSL2, RTX 4050), via Remote Control or by Hersh pasting one command. No new code.

- [ ] **Step 1:** `bash scripts/setup_wsl2.sh` — Expected: prints `cuda available: True` and the GPU name.
- [ ] **Step 2:** `make data tokenizer tokenizer-compare` — Expected: three `merges.json`, `tokenizer_compare.json`, `.bin` files with token counts logged.
- [ ] **Step 3:** `make sweep` (several hours, resumable; re-running the same command continues) — Expected: four finished MLflow runs and `scaling_fit.json` with r2 > 0.9.
- [ ] **Step 4:** `make bench eval` — Expected: `bench_kv.json`, `bpb.json`, `samples.json` in MLflow.
- [ ] **Step 5:** Commit config changes and the `mlruns` export summary (not checkpoints) — `chore: record Part 1 results`.

### Task 9: Export site data

**Files:** Create `ctg/export/site_data.py`; Test `tests/test_export.py`

**Interfaces:**
- Consumes: MLflow runs from Tasks 3–7, `artifacts/tokenizer/*/merges.json`, the 40M checkpoint for attention.
- Produces: `export(out_dir: Path) -> list[Path]` writing to `site/public/projects/clinical-tinygpt/data/`: `summary.json` (headline numbers), `tokenizer_compare.json`, `merges_{4096,8192,16384}.json`, `loss_curves.json` (`{size: [[step, train, val], ...]}`, downsampled to ≤300 points each), `scaling.json` (points + fit), `kv_bench.json`, `bpb.json`, `samples.json`, `attention_example.json` (one fixed sentence: tokens, per-layer/head weights rounded to 3 decimals, top-10 next-token probabilities). Each file carries `"schema_version": 1` and `"generated_at"`.

- [ ] **Step 1: Tests:** export against a fixture MLflow dir writes every file listed; each file is < 1 MB; `loss_curves` has ≤300 points per size; attention rows sum to 1 ± 0.01.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS. `make export` regenerates everything. **Step 5: Commit** `feat: export site data from MLflow`.

### Task 10: Interactive guide page (Part 1 chapters)

**Files:**
- Create: `site/public/projects/clinical-tinygpt/{index.html, guide.css, guide.js, bpe.js, vendor/chart.umd.min.js}`, `site/tests/bpe.test.mjs`, `site/tests/page.spec.mjs`
- Modify (portfolio repo; working copy at `/mnt/project-files/portfolio/site-update/`): `src/data/portfolio.ts` (new first entry in `projects`), `public/sitemap.xml`

**Interfaces:**
- Consumes: JSON files from Task 9.
- Produces: `bpe.js` exports `loadMerges(json) -> Tokenizer` with `encode(str) -> number[]` and `tokens(str) -> string[]` (byte-exact with Python).

Page structure (same header/footer markup as `hospital-readmission/index.html`, links `/projects/ml-guide.css` then `guide.css` for the extra pieces only, using the CSS variables in `ml-guide.css` so light/dark follow the site):

1. **Start here:** eyebrow, h1, lede in plain English, `.stats` with 4 headline numbers from `summary.json`, a clickable pipeline map (text → tokens → model → training → generation → fine-tuning → deployment; fine-tuning and deployment shown as "coming in Part 2").
2. **Chapters** (each: plain explanation → interactive chart → "What this means" box → "Read the code" link to the exact GitHub file): Tokenization (playground: textarea, vocab selector 4k/8k/16k, colored token chips, token count next to GPT-2's from precomputed examples; tokens/word bar chart), Transformer (step-through of one sentence: tokens → embeddings → attention heatmap with layer/head selector → next-token probability bars), Training (loss chart with size toggles and hover), Scaling (log-log chart with fitted line and a size slider showing predicted loss), KV cache (an animation of tokens filling a cache grid vs recomputing, plus tokens/s vs length chart), Results (bpb vs GPT-2 small, generated samples shown as-is with a "synthetic, not for clinical use" note).
3. **Progress rail:** sticky sidebar on desktop (top bar on phones) with chapter links and a check when a chapter has been scrolled through (stored in `localStorage` in try/catch; works without it); glossary terms are `<abbr>`-style buttons with a popover definition.

- [ ] **Step 1: Write failing tests.** `bpe.test.mjs`: for every string in a fixtures file produced by the Python tokenizer (`tests/fixtures/bpe_cases.json`, including `"µg °C – naïve 🙂"` and empty string), `encode` matches the Python ids for all 3 vocabs. `page.spec.mjs` (Playwright, static server): every chapter heading renders; typing "Metformin 500 mg" in the playground shows ≥1 token chip; no horizontal scroll at 375 px width; page renders in `prefers-color-scheme: dark`; with `data/scaling.json` returning 404 the Scaling chapter shows "Data not available yet" and the other charts still render.
- [ ] **Step 2: Run** `node --test site/tests/bpe.test.mjs && npx playwright test site/tests --trace off` — Expected: FAIL.
- [ ] **Step 3: Implement** the page, `bpe.js` (UTF-8 bytes via `TextEncoder`, same regex pre-split using JS Unicode property escapes), and charts with Chart.js 4.4.1 vendored (no CDN at runtime). Add the `portfolio.ts` entry: `file: "clinical_tinygpt.py"`, `name: "Clinical TinyGPT: an LLM built from scratch"`, `category: "LLM internals / Tokenization / Scaling"`, tags `["PyTorch", "BPE", "Transformers", "KV cache", "MLflow"]`, `metric` from `summary.json` (e.g. bpb vs GPT-2), `guideUrl: "/projects/clinical-tinygpt/index.html"`, `url: "https://github.com/Hersh96/clinical-tinygpt"`.
- [ ] **Step 4: Run** tests — Expected: PASS. Take desktop light, desktop dark and phone screenshots for Hersh to check.
- [ ] **Step 5: Commit** `feat: interactive guide page for Part 1`, then copy the site files into `/mnt/project-files/portfolio/site-update/` for handover like earlier guides.

### Task 11: CI, README and handover

**Files:** Create `.github/workflows/ci.yml`, `README.md`

- [ ] **Step 1:** CI on push: install CPU torch from `requirements.lock`, `pytest -q`, `make smoke`, `node --test site/tests/bpe.test.mjs`. Expected: green on the PR.
- [ ] **Step 2:** README: what it is, the not-for-clinical-use note, a results table from `summary.json`, the WSL2 setup steps, the `make` targets in order, and a "learn the code" reading order (bpe.py → layers.py → gpt.py → kv_cache.py → loop.py).
- [ ] **Step 3:** Hersh creates the empty GitHub repo `clinical-tinygpt`; push the branch and open a PR. **Commit** `docs: README and CI`.

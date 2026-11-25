import math

import pytest
import torch

from ctg.experiments.bench_kv import bench
from ctg.experiments.eval_bpb import bits_per_byte, eval_bpb, nll_sum
from ctg.model.config import GPTConfig
from ctg.model.gpt import GPT


def test_bits_per_byte_formula():
    assert bits_per_byte(math.log(2) * 800, 100) == pytest.approx(8.0)


def uniform_logits(ids: torch.Tensor) -> torch.Tensor:
    return torch.zeros(ids.shape[0], ids.shape[1], 256)


def test_uniform_byte_model_scores_eight_bits():
    def model_fn(text):
        ids = [0] + list(text.encode("utf-8"))  # 0 acts as the start token
        return nll_sum(uniform_logits, ids, block_size=16)
    assert eval_bpb(model_fn, ["Pt denies chest pain.", "µg °C"]) == pytest.approx(8.0)


def test_nll_sum_counts_every_token_once():
    calls = []

    def fn(ids):
        calls.append(ids.shape[1])
        return torch.zeros(1, ids.shape[1], 10)
    total = nll_sum(fn, [i % 10 for i in range(35)], block_size=8)  # 35 ids -> 34 predictions
    assert total == pytest.approx(34 * math.log(10))
    assert max(calls) <= 8


def test_model_nll_is_finite():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=50, n_layer=1, n_head=1, n_embd=16, block_size=8)).eval()
    assert math.isfinite(nll_sum(lambda x: m(x)[0], list(range(20)), block_size=8))


def test_bench_rows_and_cache_speedup():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=100, n_layer=4, n_head=4, n_embd=128, block_size=256)).eval()
    rows = bench(m, lengths=[16, 192], repeats=1, device="cpu")
    assert {(r["length"], r["cache"]) for r in rows} == {(16, True), (16, False), (192, True), (192, False)}
    by = {(r["length"], r["cache"]): r for r in rows}
    assert by[(192, True)]["tokens_per_sec"] >= by[(192, False)]["tokens_per_sec"]
    assert all(r["tokens_per_sec"] > 0 for r in rows)

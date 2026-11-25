import math

import pytest
import torch

from ctg.model.config import PRESETS, GPTConfig
from ctg.model.gpt import GPT
from ctg.model.kv_cache import KVCache

small_cfg = GPTConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=32)


def test_logits_shape():
    m = GPT(PRESETS["5m"])
    x = torch.randint(0, 8192, (2, 16))
    assert m(x)[0].shape == (2, 16, 8192)


def test_causal_future_does_not_change_past():
    torch.manual_seed(0)
    m = GPT(small_cfg).eval()
    x = torch.randint(0, 100, (1, 10))
    y = x.clone()
    y[0, 7:] = 3
    assert torch.allclose(m(x)[0][:, :7], m(y)[0][:, :7], atol=1e-5)


@pytest.mark.parametrize("name,target", [("5m", 5e6), ("15m", 15e6), ("40m", 40e6), ("85m", 85e6)])
def test_preset_sizes(name, target):
    assert abs(GPT(PRESETS[name]).num_params() - target) / target < 0.15


def test_presets_use_8k_vocab_and_512_context():
    assert all(c.vocab_size == 8192 and c.block_size == 512 for c in PRESETS.values())


def test_loss_at_init_near_log_vocab():
    torch.manual_seed(0)
    x = torch.randint(0, 100, (4, 32))
    y = torch.randint(0, 100, (4, 32))  # random targets: an untrained model should be ~uniform
    _, loss = GPT(small_cfg)(x, y)
    assert abs(loss.item() - math.log(small_cfg.vocab_size)) < 0.5


def test_weights_are_tied():
    m = GPT(small_cfg)
    assert m.lm_head.weight is m.tok_emb.weight


def test_too_long_input_is_an_error():
    with pytest.raises(ValueError):
        GPT(small_cfg)(torch.zeros(1, 33, dtype=torch.long))


def test_return_attn_rows_sum_to_one():
    m = GPT(small_cfg).eval()
    logits, _, attn = m(torch.randint(0, 100, (1, 9)), return_attn=True)
    assert len(attn) == 2 and attn[0].shape == (1, 2, 9, 9)
    assert torch.allclose(attn[1].sum(-1), torch.ones(1, 2, 9), atol=1e-5)
    assert torch.all(attn[0].triu(1) == 0)  # nothing attends to the future


def test_incremental_forward_with_cache_matches_full_forward():
    torch.manual_seed(0)
    m = GPT(small_cfg).eval()
    x = torch.randint(0, 100, (2, 12))
    full = m(x)[0]
    cache = KVCache(2, 2, 2, 16, 32, "cpu", torch.float32)
    a = m(x[:, :5], cache=cache, start_pos=0)[0]
    b = m(x[:, 5:], cache=cache, start_pos=5)[0]
    assert torch.allclose(torch.cat([a, b], 1), full, atol=1e-5)

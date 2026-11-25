import pytest
import torch

from ctg.model.config import GPTConfig
from ctg.model.generate import generate
from ctg.model.gpt import GPT


@pytest.fixture(scope="module")
def m_block32():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=32)).eval()


def test_cached_greedy_equals_uncached(m_block32):
    p = torch.randint(0, 100, (1, 5))
    a = generate(m_block32, p, 20, temperature=0, use_cache=True)
    b = generate(m_block32, p, 20, temperature=0, use_cache=False)
    assert a.shape == (1, 25) and torch.equal(a, b)


def test_prompt_is_kept(m_block32):
    p = torch.randint(0, 100, (2, 5))
    assert torch.equal(generate(m_block32, p, 3, temperature=0)[:, :5], p)


def test_generation_past_block_size(m_block32):
    p = torch.randint(0, 100, (1, 30))
    assert generate(m_block32, p, 50, temperature=0, use_cache=True).shape[1] == 80


def test_cached_matches_uncached_past_block_size(m_block32):
    p = torch.randint(0, 100, (1, 30))
    a = generate(m_block32, p, 50, temperature=0, use_cache=True)
    b = generate(m_block32, p, 50, temperature=0, use_cache=False)
    assert torch.equal(a, b)


def test_prompt_longer_than_block_size(m_block32):
    p = torch.randint(0, 100, (1, 70))
    a = generate(m_block32, p, 10, temperature=0, use_cache=True)
    b = generate(m_block32, p, 10, temperature=0, use_cache=False)
    assert a.shape[1] == 80 and torch.equal(a, b)


def test_sampling_is_reproducible_with_seed(m_block32):
    p = torch.randint(0, 100, (1, 4))
    a = generate(m_block32, p, 15, temperature=0.8, top_k=10, seed=1)
    b = generate(m_block32, p, 15, temperature=0.8, top_k=10, seed=1)
    assert torch.equal(a, b)


def test_top_k_one_is_greedy(m_block32):
    p = torch.randint(0, 100, (1, 4))
    assert torch.equal(generate(m_block32, p, 10, temperature=1.0, top_k=1, seed=0),
                       generate(m_block32, p, 10, temperature=0))

import json

import numpy as np
import torch

from ctg.data.tokens import TokenStream, encode_split
from ctg.tokenizer.bpe import BPETokenizer


def _tok():
    t = BPETokenizer()
    t.train([json.loads(l)["note"] for l in open("tests/fixtures/notes.jsonl")], 300)
    return t


def _jsonl(tmp_path, texts):
    p = tmp_path / "train.jsonl"
    p.write_text("".join(json.dumps({"id": str(i), "source": "s", "text": t, "summary": None}) + "\n"
                         for i, t in enumerate(texts)))
    return p


def test_encode_split_roundtrips_with_separators(tmp_path):
    tok = _tok()
    texts = ["Denies chest pain.", "µg dose BID"]
    n = encode_split(_jsonl(tmp_path, texts), tok, tmp_path / "train.bin")
    ids = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)
    assert n == len(ids)
    assert tok.decode(ids.tolist()) == "Denies chest pain.<|endoftext|>µg dose BID<|endoftext|>"


def test_batch_targets_are_inputs_shifted(tmp_path):
    path = tmp_path / "x.bin"
    np.arange(1000, dtype=np.uint16).tofile(path)
    s = TokenStream(path, block_size=16, seed=0)
    x, y = s.batch(4, "cpu")
    assert x.shape == (4, 16) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y, x + 1)


def test_same_seed_and_state_give_same_batches(tmp_path):
    path = tmp_path / "x.bin"
    np.arange(5000, dtype=np.uint16).tofile(path)
    a = TokenStream(path, 8, seed=3)
    a.batch(2, "cpu")
    state = a.state_dict()
    xa, _ = a.batch(2, "cpu")
    b = TokenStream(path, 8, seed=99)
    b.load_state_dict(state)
    xb, _ = b.batch(2, "cpu")
    assert torch.equal(xa, xb)


def test_vocab_must_fit_uint16(tmp_path):
    import pytest
    tok = BPETokenizer()
    tok.special = {"<|endoftext|>": 70000}
    with pytest.raises(ValueError):
        encode_split(_jsonl(tmp_path, ["a"]), tok, tmp_path / "o.bin")

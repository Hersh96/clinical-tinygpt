import json

import pytest

from ctg.tokenizer.bpe import BPETokenizer

CASES = ["", "a", "Pt denies chest pain.", "Metformin 500 mg PO BID", "µg °C – naïve 🙂",
         "x" * 5000, "  leading and trailing  \n\n", "HbA1c 7.2%; BP 140/90 mmHg"]


@pytest.fixture(scope="module")
def corpus():
    lines = [json.loads(l)["note"] for l in open("tests/fixtures/notes.jsonl")]
    return lines * 3


@pytest.fixture(scope="module")
def trained_tok(corpus):
    t = BPETokenizer()
    t.train(corpus, vocab_size=400)
    return t


def test_roundtrip_any_utf8(trained_tok):
    for s in CASES:
        assert trained_tok.decode(trained_tok.encode(s)) == s


def test_untrained_tokenizer_is_plain_bytes():
    t = BPETokenizer()
    assert t.encode("µ") == list("µ".encode("utf-8"))


def test_vocab_size_exact(corpus):
    t = BPETokenizer()
    t.train(corpus, vocab_size=300)
    assert len(t.vocab) == 300


def test_training_is_deterministic(corpus):
    a, b = BPETokenizer(), BPETokenizer()
    a.train(corpus, 300)
    b.train(corpus, 300)
    assert a.merges == b.merges


def test_first_merge_is_most_frequent_pair():
    t = BPETokenizer()
    t.train(["aaab aaab aaab"], 258)  # 256 bytes + 1 merge + <|endoftext|>
    assert t.merges[0] == (97, 97)


def test_merges_compress(trained_tok, corpus):
    text = corpus[0]
    assert len(trained_tok.encode(text)) < 0.7 * len(text.encode("utf-8"))


def test_save_load_identical(tmp_path, trained_tok):
    trained_tok.save(tmp_path / "m.json")
    data = json.loads((tmp_path / "m.json").read_text())
    assert set(data) == {"vocab_size", "pattern", "merges", "special"}
    t2 = BPETokenizer.load(tmp_path / "m.json")
    s = "Lisinopril 10 mg daily"
    assert t2.encode(s) == trained_tok.encode(s)


def test_endoftext_is_single_token(trained_tok):
    eot = trained_tok.special["<|endoftext|>"]
    assert eot == 399
    assert trained_tok.encode("<|endoftext|>", allow_special=True) == [eot]
    assert trained_tok.decode([eot]) == "<|endoftext|>"


def test_special_text_not_special_by_default(trained_tok):
    assert trained_tok.special["<|endoftext|>"] not in trained_tok.encode("<|endoftext|>")


def test_token_strings(trained_tok):
    pieces = trained_tok.token_strings("Pt denies chest pain.")
    assert "".join(pieces) == "Pt denies chest pain."


def _naive_merges(texts, n_merges):
    """Slow reference: recount every pair from scratch before each merge."""
    import regex
    from collections import Counter
    from ctg.tokenizer.bpe import GPT2_PATTERN, _merge
    words = Counter()
    for t in texts:
        words.update(regex.findall(GPT2_PATTERN, t))
    seqs = {w: list(w.encode()) for w in words}
    merges = []
    for k in range(n_merges):
        c = Counter()
        for w, ids in seqs.items():
            for p in zip(ids, ids[1:]):
                c[p] += words[w]
        if not c:
            break
        best = min(c, key=lambda p: (-c[p], p))
        merges.append(best)
        seqs = {w: _merge(ids, best, 256 + k) for w, ids in seqs.items()}
    return merges


def test_incremental_training_matches_naive(corpus):
    t = BPETokenizer()
    t.train(corpus, 256 + 80 + 1)
    assert t.merges == _naive_merges(corpus, 80)

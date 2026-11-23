from ctg.experiments.tokenizer_compare import EXAMPLE_TERMS, compare


def char_tok(s):
    return [ord(c) for c in s]


def test_tokens_per_word_and_chars_per_token():
    r = compare({"char": char_tok}, ["a b c"])
    assert abs(r["char"]["tokens_per_word"] - 5 / 3) < 1e-9
    assert r["char"]["chars_per_token"] == 1.0


def test_examples_cover_fixed_terms():
    r = compare({"char": char_tok}, ["x"], pieces={"char": lambda s: list(s)})
    assert [e["term"] for e in r["char"]["examples"]] == EXAMPLE_TERMS
    assert r["char"]["examples"][0]["pieces"] == list("metformin")


def test_empty_texts_do_not_divide_by_zero():
    r = compare({"char": char_tok}, [""])
    assert r["char"]["tokens_per_word"] == 0.0

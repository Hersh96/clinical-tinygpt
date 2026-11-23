import json

from ctg.data.prepare import clean_text, prepare_corpus, split_for


def test_split_is_deterministic_and_proportional():
    ids = [f"doc{i}" for i in range(20000)]
    a = [split_for(i, seed=0) for i in ids]
    b = [split_for(i, seed=0) for i in ids]
    assert a == b
    assert abs(a.count("train") / len(a) - 0.90) < 0.01
    assert abs(a.count("val") / len(a) - 0.05) < 0.01


def test_split_changes_with_seed():
    ids = [f"doc{i}" for i in range(2000)]
    assert [split_for(i, 0) for i in ids] != [split_for(i, 1) for i in ids]


def test_prepare_has_no_overlap(tmp_path, tiny_fixture_cfg):
    paths = prepare_corpus(tiny_fixture_cfg, tmp_path)
    assert set(paths) == {"train", "val", "test"}
    ids = {k: {json.loads(l)["id"] for l in open(p)} for k, p in paths.items()}
    assert not (ids["train"] & ids["test"]) and not (ids["train"] & ids["val"]) and not (ids["val"] & ids["test"])
    assert sum(len(v) for v in ids.values()) == 20


def test_prepare_records_have_schema(tmp_path, tiny_fixture_cfg):
    paths = prepare_corpus(tiny_fixture_cfg, tmp_path)
    rec = json.loads(open(paths["train"]).readline())
    assert set(rec) == {"id", "source", "text", "summary"}
    assert rec["id"].startswith("fixture_notes:")


def test_prepare_writes_licenses(tmp_path, tiny_fixture_cfg):
    prepare_corpus(tiny_fixture_cfg, tmp_path)
    assert "synthetic test fixture" in (tmp_path / "LICENSES.md").read_text()


def test_missing_license_is_an_error(tmp_path, tiny_fixture_cfg):
    del tiny_fixture_cfg["sources"][0]["license"]
    import pytest
    with pytest.raises(ValueError, match="license"):
        prepare_corpus(tiny_fixture_cfg, tmp_path)


def test_clean_normalizes_whitespace_keeps_unicode():
    assert clean_text("BP  120/80\r\n\r\n\r\nµg dose") == "BP 120/80\n\nµg dose"

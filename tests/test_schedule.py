import pytest

from ctg.train.schedule import lr_at

KW = dict(max_lr=1e-3, warmup=10, total=100, min_lr_ratio=0.1)


def test_starts_at_zero():
    assert lr_at(0, **KW) == 0.0


def test_peaks_at_end_of_warmup():
    assert lr_at(10, **KW) == pytest.approx(1e-3)


def test_ends_at_min_lr():
    assert lr_at(100, **KW) == pytest.approx(1e-4)
    assert lr_at(150, **KW) == pytest.approx(1e-4)


def test_monotone_after_warmup():
    lrs = [lr_at(s, **KW) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_linear_warmup():
    assert lr_at(5, **KW) == pytest.approx(5e-4)

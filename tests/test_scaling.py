import numpy as np
import pytest

from ctg.experiments.scaling import fit_power_law, predict


def _synthetic(noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    n = np.geomspace(1e6, 1e8, 8)
    loss = 400 * n ** -0.3 + 1.8
    return n.tolist(), (loss * (1 + noise * rng.standard_normal(len(n)))).tolist()


def test_recovers_exponent_from_noisy_points():
    fit = fit_power_law(*_synthetic())
    assert abs(fit["alpha"] - 0.3) < 0.05
    assert fit["r2"] > 0.98


def test_exact_points_fit_exactly():
    n, loss = _synthetic(noise=0)
    fit = fit_power_law(n, loss)
    assert fit["alpha"] == pytest.approx(0.3, abs=1e-3)
    assert fit["c"] == pytest.approx(1.8, abs=1e-2)
    assert predict(fit, 1e7) == pytest.approx(400 * 1e7 ** -0.3 + 1.8, rel=1e-3)


def test_prediction_decreases_with_size():
    fit = fit_power_law(*_synthetic())
    preds = [predict(fit, n) for n in np.geomspace(1e6, 1e9, 10)]
    assert all(a > b for a, b in zip(preds, preds[1:]))


def test_four_points_like_the_sweep():
    n = [4.8e6, 14.2e6, 37.4e6, 85.0e6]
    loss = [400 * x ** -0.3 + 1.8 for x in n]
    fit = fit_power_law(n, loss)
    assert 0 < fit["alpha"] < 1 and fit["c"] >= 0


def test_needs_three_points():
    with pytest.raises(ValueError):
        fit_power_law([1e6, 2e6], [3.0, 2.9])

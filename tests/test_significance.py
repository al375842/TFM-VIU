"""M3 tests: statistical significance of H1 (binomial) and H2 (McNemar)."""

import numpy as np
import pandas as pd
import pytest

from tfm import metrics


def test_naive_floor_uses_the_train_majority_class():
    """The dummy learns the majority in train; test accuracy follows from it.

    Regression against max(p, 1-p): when the drift flips sign the floor must
    fall below 0.50 (BTC-USD and ETH-USD, 2025).
    """
    y_train = pd.Series([1] * 60 + [0] * 40)  # majority = 1
    y_test = pd.Series([1] * 40 + [0] * 60)  # but test is majority 0
    floor = metrics.naive_baseline_accuracy(y_train, y_test)
    assert floor == pytest.approx(0.40)  # not 0.60


def test_binomial_flags_a_clear_improvement():
    rng = np.random.default_rng(0)
    y_true = pd.Series(rng.integers(0, 2, 400))
    y_pred = y_true.copy()
    y_pred[:80] = 1 - y_pred[:80]  # 80% accuracy against a 0.50 floor
    out = metrics.binomial_test_vs_baseline(y_true, y_pred, 0.50)
    assert out["n_correct"] == 320
    assert out["p_value"] < 0.001


def test_binomial_is_one_tailed():
    """A model far below the floor must not come out significant."""
    y_true = pd.Series([1] * 100)
    y_pred = pd.Series([0] * 100)  # 0% accuracy
    out = metrics.binomial_test_vs_baseline(y_true, y_pred, 0.60)
    assert out["p_value"] == pytest.approx(1.0)


def test_mcnemar_uses_only_discordant_pairs():
    y_true = pd.Series([1] * 100)
    a = pd.Series([1] * 100)  # always right
    b = pd.Series([1] * 70 + [0] * 30)  # right 70, wrong 30
    out = metrics.mcnemar_test(y_true, a, b)
    assert (out["b"], out["c"], out["n_discordant"]) == (30, 0, 30)
    assert out["p_value"] < 0.001


def test_mcnemar_on_identical_predictions_is_undefined():
    y_true = pd.Series([1, 0, 1, 0])
    same = pd.Series([1, 1, 1, 1])
    out = metrics.mcnemar_test(y_true, same, same)
    assert out["n_discordant"] == 0
    assert np.isnan(out["p_value"])

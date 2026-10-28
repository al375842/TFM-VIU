"""Regression checks for target availability and financial evaluation windows."""

import numpy as np
import pandas as pd
import pytest

from tfm import backtest, features, lstm, validation


@pytest.mark.parametrize("horizon", [1, 5])
def test_training_labels_are_available_before_holdout_and_cv(horizon):
    index = pd.bdate_range("2023-01-02", "2025-03-31")
    close = pd.Series(100 + np.sin(np.arange(len(index))) + np.arange(len(index)), index=index)
    raw = pd.DataFrame({"Close": close, "Volume": 100.0})
    raw.loc[index[250:255], "Volume"] = 0.0
    dataset = features.build_dataset(raw, n=horizon, technical=False)
    label_end = pd.Series(index, index=index).shift(-horizon)
    train, test = validation.temporal_split(dataset)
    assert (label_end.loc[train.index] < pd.Timestamp("2025-01-01")).all()
    for a, b in validation.make_cv(gap=horizon).split(train):
        assert (label_end.loc[train.index[a]] < train.index[b[0]]).all()
    assert test.index.equals(dataset.loc["2025":].index)


def test_lstm_uses_the_same_purged_training_boundary(monkeypatch):
    index = pd.bdate_range("2024-10-01", "2025-01-10")
    dataset = pd.DataFrame({"x": 1.0, "target": 1}, index=index)
    received = {}

    def capture(ds, cols, **kwargs):
        received.update(kwargs)
        return pd.Series(dtype=float)

    monkeypatch.setattr(lstm, "fit_predict_range", capture)
    lstm.fit_predict(dataset, ["x"])
    train, _ = validation.temporal_split(dataset)
    assert received["train_end"] == train.index[-1]
    assert received["train_end"] < pd.Timestamp("2024-12-31")


def test_financial_window_does_not_extend_a_stale_signal():
    index = pd.bdate_range("2025-11-01", "2025-12-31")
    prices = pd.DataFrame({"Close": np.arange(len(index)) + 100.0}, index=index)
    signals = index[index <= "2025-11-21"]
    result = backtest.evaluation_prices(prices, signals)
    assert result.index[-1] == pd.Timestamp("2025-11-21")
    assert result.index.equals(signals)
    with pytest.raises(ValueError, match="signal"):
        backtest.evaluation_prices(prices, [])


def test_volume_gap_and_first_recovery_are_never_imputed():
    index = pd.bdate_range("2025-01-01", periods=8)
    raw = pd.DataFrame({"Close": 100.0, "Volume": [10, 12, 0, 0, 20, 30, 40, 50]}, index=index)
    change = features.build_features(raw)["vol_change"]
    assert change.iloc[2:5].isna().all()
    assert change.iloc[5] == pytest.approx(0.5)


def test_rolling_lstm_receives_only_the_declared_training_window(monkeypatch):
    from types import SimpleNamespace
    from tfm import walkforward

    index = pd.bdate_range("2010-01-01", "2025-12-31")
    dataset = pd.DataFrame({"x": np.arange(len(index)), "target": 1}, index=index)
    fold = validation.make_walkforward_folds(index, mode="rolling")[-1]
    received = {}

    def capture(ds, cols, **kwargs):
        received["index"] = ds.index
        return pd.Series(1, index=ds.loc[kwargs["test_start"]:kwargs["test_end"]].index)

    monkeypatch.setattr(lstm, "fit_predict_range", capture)
    fit = SimpleNamespace(specs={}, frozen_params={}, with_lstm=True)
    train, test = dataset.loc[fold.train_start:fold.train_end], dataset.loc[fold.test_start:fold.test_end]
    walkforward._fold_predictions(fold, train[["x"]], train.target, test[["x"]],
                                  fit=fit, ticker="X", dataset=dataset, cols=["x"])
    assert received["index"][0] == fold.train_start
    assert received["index"][0] > dataset.index[0]

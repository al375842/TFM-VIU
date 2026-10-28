"""M2 tests: feature correctness, target definition and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from tfm import features


@pytest.fixture
def ohlcv():
    """Synthetic 300-session OHLCV series with a reproducible random walk."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1e5, 1e6, 300).astype(float),
        }
    )


def test_log_returns_definition(ohlcv):
    out = features.build_features(ohlcv)
    expected = np.log(ohlcv["Close"] / ohlcv["Close"].shift(1))
    pd.testing.assert_series_equal(out["log_ret"], expected, check_names=False)


def test_target_binary_and_shifted(ohlcv):
    target = features.build_target(ohlcv["Close"], n=1)
    close = ohlcv["Close"]
    # y_t = 1 iff P_{t+1} > P_t
    manual = (close.shift(-1) > close).astype(int)
    assert (target.dropna().astype(int) == manual.iloc[:-1]).all()
    # last N rows must be invalid (no future price available)
    assert target.iloc[-1] is pd.NA or pd.isna(target.iloc[-1])


def test_features_are_causal(ohlcv):
    """Changing the future must not change the feature value at time t."""
    t = 200
    base = features.build_features(ohlcv).iloc[:t]
    modified = ohlcv.copy()
    modified.iloc[t + 1 :, modified.columns.get_loc("Close")] *= 10  # tamper future
    tampered = features.build_features(modified).iloc[:t]
    pd.testing.assert_frame_equal(base, tampered)


def test_dataset_has_no_nan(ohlcv):
    ds = features.build_dataset(ohlcv, technical=False)
    assert not ds.isna().any().any()
    assert set(ds["target"].unique()) <= {0, 1}


def test_zero_volume_yields_no_infinite_feature(ohlcv):
    """Unreported volume (0) must not leak an inf into the model matrix.

    Regression: ^IBEX has 180 zero-volume sessions and pct_change turned the
    0 -> x transition into inf, which survives dropna and breaks the scaler.
    """
    tampered = ohlcv.copy()
    tampered.iloc[100:110, tampered.columns.get_loc("Volume")] = 0.0
    out = features.build_features(tampered)
    assert not np.isinf(out.to_numpy(dtype=float)).any()

    ds = features.build_dataset(tampered, technical=False)
    assert not np.isinf(ds.to_numpy(dtype=float)).any()
    assert not ds.isna().any().any()

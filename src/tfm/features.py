"""M2 - Features and target, free of temporal leakage.

Turns raw OHLCV into a model-ready dataset: predictors (log returns, moving
averages, rolling volatility, volume change, RSI, MACD), a binary direction
target over N sessions, and a dropna for the NaN left by the rolling windows
and the target shift.

Every predictor looks backwards only (its value at t uses rows up to t), so
the features can be built once over the whole series, walk-forward included.
What depends on the split is the scaler, fitted on training rows inside the
Pipeline, and the purge of the last training labels.

Thesis: §4.3 (Listings 4.4 to 4.6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Core predictors from adjusted prices and volume (Listing 4.4)."""
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    # log returns: additive over time and more symmetric than simple ones
    out["log_ret"] = np.log(close / close.shift(1))

    # trend: short and long moving averages, plus their ratio
    out["sma_short"] = close.rolling(window=config.SMA_SHORT_WINDOW).mean()
    out["sma_long"] = close.rolling(window=config.SMA_LONG_WINDOW).mean()
    out["sma_ratio"] = out["sma_short"] / out["sma_long"]

    # risk context: rolling std of the log returns
    out["volatility"] = out["log_ret"].rolling(window=config.VOLATILITY_WINDOW).std()

    # Yahoo reports volume 0 when it is not published (180 ^IBEX sessions),
    # not when nothing traded: treat it as missing so pct_change never gives inf
    volume = df["Volume"].replace(0, np.nan)
    out["vol_change"] = volume.pct_change(fill_method=None)

    # safety net: an inf survives dropna and then breaks the scaler
    return out.replace([np.inf, -np.inf], np.nan)


def add_technical_indicators(features: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Add RSI (momentum) and MACD (trend) from `ta`; past closes only."""
    try:
        import ta as ta_lib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The `ta` library is required for technical indicators: pip install ta"
        ) from exc

    features = features.copy()

    features[f"rsi_{config.RSI_LENGTH}"] = ta_lib.momentum.RSIIndicator(
        close=close, window=config.RSI_LENGTH
    ).rsi()

    macd = ta_lib.trend.MACD(
        close=close,
        window_fast=config.MACD_PARAMS["fast"],
        window_slow=config.MACD_PARAMS["slow"],
        window_sign=config.MACD_PARAMS["signal"],
    )
    features["macd"] = macd.macd()
    features["macd_signal"] = macd.macd_signal()
    features["macd_hist"] = macd.macd_diff()
    return features


def build_target(
    close: pd.Series,
    n: int = config.TARGET_HORIZON_N,
    threshold: float = config.TARGET_THRESHOLD,
) -> pd.Series:
    """Binary target: 1 if the price rises after N sessions, else 0.

    shift(-N) reads the future on purpose: that is what a target is, and no
    feature may do it. With tau > 0, y=1 iff P_{t+N}/P_t - 1 > tau.
    """
    future_close = close.shift(-n)
    if threshold > 0:
        target = ((future_close / close - 1) > threshold).astype(int)
    else:
        target = (future_close > close).astype(int)
    # the last N rows have no future price: mark them for the dropna
    target.iloc[-n:] = pd.NA
    return target.rename("target")


def build_dataset(
    df: pd.DataFrame,
    n: int = config.TARGET_HORIZON_N,
    threshold: float = config.TARGET_THRESHOLD,
    technical: bool = True,
) -> pd.DataFrame:
    """Full M2 output for one asset: features + target, leakage-safe."""
    features = build_features(df)
    if technical:
        features = add_technical_indicators(features, df["Close"])
    target = build_target(df["Close"], n=n, threshold=threshold)
    dataset = features.join(target)
    dataset = dataset.dropna()
    dataset["target"] = dataset["target"].astype(int)
    dataset.attrs["target_horizon"] = n
    return dataset


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """All columns except the target."""
    return [c for c in dataset.columns if c != "target"]

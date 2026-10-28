"""M4 tests: the simulation must actually trade the signals it is given."""

import numpy as np
import pandas as pd
import pytest

from tfm import backtest

pytest.importorskip("backtesting")


def _prices(level: float, n: int = 120) -> pd.DataFrame:
    """Rising series around a given price level (index levels are ~20_000)."""
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(level * np.linspace(1.0, 1.2, n), index=idx)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": 1e6,
        }
    )


@pytest.mark.parametrize("level", [50.0, 250.0, 20_000.0])
def test_strategy_trades_at_every_price_level(level):
    """Regression: with whole-unit trading no order filled at index levels.

    The base Backtest class could not buy a ~20_000 point index with 10_000
    of capital, and the flat 0.0 was misread as "does not beat B&H".
    """
    prices = _prices(level)
    signal = pd.Series(1, index=prices.index)  # always long
    out = backtest.run_strategy_backtest(prices, signal, "AAPL")
    assert out["metrics"]["cumulative_return"] > 0.0


@pytest.mark.parametrize("level", [50.0, 20_000.0])
def test_result_does_not_depend_on_the_price_level(level):
    """Same trajectory at any quotation level must give the same return.

    A whole-unit simulation leaves idle cash proportional to price/capital,
    which biased the comparison precisely on the highest-quoted assets.
    """
    signal = pd.Series(1, index=_prices(level).index)
    ref = backtest.run_strategy_backtest(_prices(1.0), signal, "AAPL")
    out = backtest.run_strategy_backtest(_prices(level), signal, "AAPL")
    assert out["metrics"]["cumulative_return"] == pytest.approx(
        ref["metrics"]["cumulative_return"], abs=1e-4
    )

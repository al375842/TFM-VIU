"""Financial metrics tests: conditional annualisation, Sharpe, drawdown."""

import numpy as np
import pandas as pd
import pytest

from tfm import metrics


def test_annualization_factor_conditional():
    assert metrics.annualization_factor("BTC-USD") == pytest.approx(np.sqrt(365))
    assert metrics.annualization_factor("ETH-USD") == pytest.approx(np.sqrt(365))
    assert metrics.annualization_factor("AAPL") == pytest.approx(np.sqrt(252))
    assert metrics.annualization_factor("^GSPC") == pytest.approx(np.sqrt(252))


def test_sharpe_uses_asset_calendar():
    returns = pd.Series([0.01, -0.005, 0.008, 0.002, -0.001] * 50)
    s_equity = metrics.sharpe_ratio(returns, "AAPL", rf=0.0)
    s_crypto = metrics.sharpe_ratio(returns, "BTC-USD", rf=0.0)
    assert s_crypto / s_equity == pytest.approx(np.sqrt(365 / 252))


def test_max_drawdown_known_case():
    # 100 -> 120 -> 60 -> 90: worst fall is 60/120 - 1 = -50%
    equity = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert metrics.max_drawdown(equity) == pytest.approx(-0.5)


def test_cumulative_return():
    equity = pd.Series([100.0, 110.0, 121.0])
    assert metrics.cumulative_return(equity) == pytest.approx(0.21)


def test_win_rate():
    trades = pd.Series([0.02, -0.01, 0.03, -0.02])
    assert metrics.win_rate(trades) == pytest.approx(0.5)

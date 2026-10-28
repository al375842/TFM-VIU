"""M4 - Backtesting.

Simulates a long/cash strategy against passive Buy & Hold. Rule
(thesis §4.6.1): long when the model predicts up, cash otherwise,
no shorts. Backtesting.py fills orders at the next bar's open, so a signal
from the close of t is executed at the open of t+1, with no look-ahead.
"""

from __future__ import annotations

import pandas as pd

from . import config, metrics


def evaluation_prices(prices: pd.DataFrame, signal_index) -> pd.DataFrame:
    """Price bars from the first to the last signal date, and no further."""
    index = pd.DatetimeIndex(signal_index)
    if index.empty:
        raise ValueError("At least one signal date is required")
    return prices.loc[index.min():index.max()]


def run_strategy_backtest(
    prices: pd.DataFrame,
    predictions: pd.Series,
    ticker: str,
    cash: float = config.INITIAL_CASH,
    commission: float = config.COMMISSION,
) -> dict:
    """Simulate the long/cash ML strategy with Backtesting.py.

    `predictions` is binary (1 = long, 0 = cash), indexed like `prices`.
    Returns the equity curve, the stats object and the financial metrics.
    """
    from backtesting import Strategy  # lazy import
    from backtesting.lib import FractionalBacktest

    df = prices[["Open", "High", "Low", "Close", "Volume"]].copy()
    # a session without a prediction keeps the previous position (cash at first)
    df["Signal"] = predictions.reindex(df.index).ffill().fillna(0).astype(int)

    class MLDirectionalStrategy(Strategy):
        """Long while the model predicts up; flat otherwise."""

        def init(self):
            pass

        def next(self):
            signal = self.data.Signal[-1]
            if signal == 1 and not self.position:
                self.buy()  # filled at next bar's open
            elif signal == 0 and self.position:
                self.position.close()

    # FractionalBacktest, not Backtest: with whole units, an index at ~20_000
    # points cannot be bought with this cash, and the flat 0.0 was misread as
    # "does not beat Buy & Hold". Index levels are still a synthetic,
    # non-investable proxy.
    bt = FractionalBacktest(
        df,
        MLDirectionalStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        finalize_trades=True,  # close the last position so it counts in win_rate
    )
    stats = bt.run()
    equity = stats["_equity_curve"]["Equity"]
    trades = stats["_trades"]
    trade_returns = trades["ReturnPct"] if len(trades) else pd.Series(dtype=float)

    return {
        "equity": equity,
        "stats": stats,
        "metrics": metrics.financial_metrics(
            equity, ticker, trade_returns=trade_returns
        ),
    }


def run_buy_and_hold(
    prices: pd.DataFrame,
    ticker: str,
    cash: float = config.INITIAL_CASH,
) -> dict:
    """Passive benchmark: buy at the start and hold.

    Computed from the adjusted closes: it is always fully invested, so no
    order simulation is needed.
    """
    close = prices["Close"].dropna()
    equity = cash * close / close.iloc[0]
    return {
        "equity": equity,
        "metrics": metrics.financial_metrics(equity, ticker),
    }


def compare_vs_benchmark(strategy: dict, benchmark: dict) -> dict[str, bool]:
    """Success criterion (thesis §4.6.3): beat Buy & Hold on both
    return and Sharpe. Beating it on only one of them does not add value."""
    s, b = strategy["metrics"], benchmark["metrics"]
    beats_return = s["cumulative_return"] > b["cumulative_return"]
    beats_sharpe = s["sharpe"] > b["sharpe"]
    return {
        "beats_return": beats_return,
        "beats_sharpe": beats_sharpe,
        "adds_value": beats_return and beats_sharpe,
    }

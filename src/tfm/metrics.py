"""Evaluation metrics: classification (M3) and financial (M4).

Financial metrics are annualised by trading calendar: sqrt(252) for
exchange-traded assets, sqrt(365) for crypto, which trades every day.
Thesis: §3.5.2 (Listing 4.15).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from . import config

# ---------------------------------------------------------------------------
# Classification metrics (M3)
# ---------------------------------------------------------------------------


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Accuracy, precision, recall and F1 for a binary direction target."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean of the per-class recalls, reported next to accuracy, never instead
    of it. A constant classifier scores 0.50 whatever the base rate, so folds
    with different base rates share the same reference."""
    return float(balanced_accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Statistical significance (M3) - hypotheses H1 and H2
# ---------------------------------------------------------------------------


def naive_baseline_accuracy(y_train, y_test) -> float:
    """Accuracy the naive reference reaches on the test set.

    The dummy learns the majority class from the training set, so its test
    accuracy is that class's share of the test set, not max(p, 1-p). When the
    trend reverses, the reference can fall below 0.50 (BTC, 2025).
    """
    majority = pd.Series(y_train).value_counts().idxmax()
    return float((pd.Series(y_test).reset_index(drop=True) == majority).mean())


def binomial_test_vs_baseline(y_true, y_pred, baseline_accuracy: float) -> dict:
    """One-tailed binomial test for H1: does the model beat the naive floor?

    Under H0 each test case is a hit with probability p0 = `baseline_accuracy`,
    so the hits follow Binomial(n, p0). One-tailed because H1 only claims the
    model is better. Exploratory: it treats the estimated baseline as fixed
    and ignores serial dependence.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    hits = int((y_true == y_pred).sum())
    p0 = float(np.clip(baseline_accuracy, 1e-12, 1 - 1e-12))
    result = stats.binomtest(hits, n, p0, alternative="greater")
    return {
        "n_test": n,
        "n_correct": hits,
        "baseline_accuracy": float(baseline_accuracy),
        "p_value": float(result.pvalue),
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict:
    """Exact McNemar test for H2: do two models differ on the same cases?

    Only discordant cases count: b = A right and B wrong, c = the reverse.
    Under H0 b ~ Binomial(b + c, 0.5). Exact rather than chi-square, which is
    unreliable when b + c is small.
    """
    y_true = np.asarray(y_true)
    correct_a = y_true == np.asarray(y_pred_a)
    correct_b = y_true == np.asarray(y_pred_b)
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:  # identical predictions: nothing to test
        return {"b": b, "c": c, "n_discordant": 0, "p_value": np.nan}
    result = stats.binomtest(b, b + c, 0.5, alternative="two-sided")
    return {"b": b, "c": c, "n_discordant": b + c, "p_value": float(result.pvalue)}


# ---------------------------------------------------------------------------
# Financial metrics (M4) - thesis Listing 4.15
# ---------------------------------------------------------------------------


def annualization_factor(ticker: str) -> float:
    """sqrt(365) for crypto (24/7), sqrt(252) for exchange-traded assets."""
    periods = 365 if ticker in config.CRYPTO else 252
    return np.sqrt(periods)


def periods_per_year(ticker: str) -> int:
    """365 for crypto, 252 for exchange-traded assets."""
    return 365 if ticker in config.CRYPTO else 252


def sharpe_ratio(
    returns: pd.Series, ticker: str, rf: float = config.RISK_FREE_RATE
) -> float:
    """Annualised Sharpe ratio of a daily return series.

    `rf` is an annual rate, converted here to a per-session rate.
    """
    returns = returns.dropna()
    if returns.std() == 0 or len(returns) == 0:
        return np.nan
    rf_periodic = rf / periods_per_year(ticker)
    excess = returns.mean() - rf_periodic
    return float((excess / returns.std()) * annualization_factor(ticker))


def annualized_volatility(returns: pd.Series, ticker: str) -> float:
    """Standard deviation of the per-session returns, annualised."""
    return float(returns.dropna().std() * annualization_factor(ticker))


def cumulative_return(equity: pd.Series) -> float:
    """Total return of an equity curve: final/initial - 1."""
    equity = equity.dropna()
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall of the equity curve (a negative number)."""
    equity = equity.dropna()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def win_rate(trade_returns: pd.Series) -> float:
    """Fraction of trades with a strictly positive return.

    Always counted per trade, never per session: they are different
    quantities.
    """
    trade_returns = trade_returns.dropna()
    if len(trade_returns) == 0:
        return np.nan
    return float((trade_returns > 0).mean())


def financial_metrics(
    equity: pd.Series,
    ticker: str,
    trade_returns: pd.Series | None = None,
    rf: float = config.RISK_FREE_RATE,
) -> dict[str, float]:
    """Financial metrics reported in Ch. 5.

    `win_rate` is NaN for Buy & Hold (no trades are passed): a single position
    held all period says nothing about the hit rate.
    """
    returns = equity.pct_change()
    return {
        "cumulative_return": cumulative_return(equity),
        "sharpe": sharpe_ratio(returns, ticker, rf=rf),
        "volatility": annualized_volatility(returns, ticker),
        "max_drawdown": max_drawdown(equity),
        "win_rate": (
            np.nan if trade_returns is None else win_rate(trade_returns)
        ),
    }

"""M3 - Temporal validation: three safeguards against look-ahead bias.

  1. `temporal_split`: 2025 onwards is the test set, never used to fit or tune.
  2. `make_cv`: TimeSeriesSplit inside the training window, for tuning.
  3. `make_walkforward_folds`: repeats (1) once per calendar year, purging
     the boundary. A robustness check only: every headline figure in Ch. 5
     comes from the single 2025 test set.

Thesis: §4.4.1 (Listing 4.7) and §4.4.7 (Listing 4.11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from . import config
from .models import ModelSpec


def temporal_split(
    dataset: pd.DataFrame, test_start: str = config.TEST_START,
    *, purge_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split, purging labels that read the test set."""
    if purge_rows is None:
        purge_rows = dataset.attrs.get("target_horizon", config.TARGET_HORIZON_N)
    if purge_rows < 0:
        raise ValueError("purge_rows must be nonnegative")
    train = dataset.loc[dataset.index < test_start]
    test = dataset.loc[dataset.index >= test_start]
    if purge_rows:
        train = train.iloc[:-purge_rows]
    return train, test


def make_cv(n_splits: int = config.CV_N_SPLITS,
            gap: int = config.TARGET_HORIZON_N) -> TimeSeriesSplit:
    """TimeSeriesSplit with a gap of N rows before each validation block."""
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def tune_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = config.CV_SCORING,
):
    """Fit a ModelSpec, tuning its grid with temporal CV when it has one.

    Returns (fitted_estimator, best_params). Thesis: Listing 4.7.
    """
    if not spec.param_grid:
        est = spec.estimator.fit(X_train, y_train)
        return est, {}

    grid = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.param_grid,
        cv=make_cv(gap=X_train.attrs.get("target_horizon", config.TARGET_HORIZON_N)),
        scoring=scoring,
        n_jobs=config.N_JOBS,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# ---------------------------------------------------------------------------
# Walk-forward folds (robustness analysis)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardFold:
    """One train/test block. `train_end` is already purged: it is the last
    row used for fitting, not the last row before the test block."""

    fold_id: int
    test_year: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # already purged
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    purge_rows: int


def make_walkforward_folds(
    index,
    *,
    mode: str = config.WF_MODE,
    first_test_year: int = config.WF_FIRST_TEST_YEAR,
    last_test_year: int = config.WF_LAST_TEST_YEAR,
    min_train_years: int = config.WF_MIN_TRAIN_YEARS,
    rolling_train_years: int = config.WF_ROLLING_TRAIN_YEARS,
    purge_rows: int | None = config.WF_PURGE_ROWS,
) -> list[WalkForwardFold]:
    """Calendar-year folds available for one asset.

    `index` is the asset's own trading calendar, never a shared one. Two
    guards make the number of folds differ between assets, on purpose:

    * `min_train_years` skips folds with too little history, so BTC (from
      2014) and ETH (from 2017) get fewer folds. Compare assets by the share
      of folds, not by the count.
    * `purge_rows` drops the last k training rows: their labels use prices
      at t+N, which already fall in the test block (lopezdeprado2018).

    `anchored` grows the training window from the first session; `rolling`
    keeps a fixed window of `rolling_train_years`.
    """
    if mode not in ("anchored", "rolling"):
        raise ValueError(f"Unknown walk-forward mode: {mode!r}")
    if purge_rows is None:
        purge_rows = config.TARGET_HORIZON_N

    index = pd.DatetimeIndex(index).sort_values()
    folds: list[WalkForwardFold] = []
    fold_id = 0

    for year in range(first_test_year, last_test_year + 1):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        next_year = pd.Timestamp(year=year + 1, month=1, day=1)
        test_pos = np.flatnonzero((index >= test_start) & (index < next_year))
        if len(test_pos) == 0:  # asset did not trade that year
            continue

        # everything strictly before the test block is candidate training data
        train_hi = int(test_pos[0])  # exclusive upper bound, in positions
        if mode == "rolling":
            window_start = test_start - pd.DateOffset(years=rolling_train_years)
            train_lo = int(np.searchsorted(index, window_start, side="left"))
        else:
            train_lo = 0
        if train_hi <= train_lo:
            continue

        # measure history in calendar time, not rows: a crypto year has 365
        # sessions and an equity year about 252
        span_years = (test_start - index[train_lo]).days / 365.25
        if span_years < min_train_years:
            continue

        train_hi_purged = train_hi - purge_rows
        if train_hi_purged <= train_lo:
            continue

        fold_id += 1
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                test_year=year,
                train_start=index[train_lo],
                train_end=index[train_hi_purged - 1],
                test_start=index[test_pos[0]],
                test_end=index[test_pos[-1]],
                n_train=train_hi_purged - train_lo,
                n_test=len(test_pos),
                purge_rows=purge_rows,
            )
        )
    return folds


def split_by_fold(X, y, fold: WalkForwardFold) -> tuple:
    """(X_train, y_train, X_test, y_test) of one fold.

    Both ends inclusive; `fold.train_end` is already purged, so the gap
    between blocks is exactly `purge_rows` wide.
    """
    train = slice(fold.train_start, fold.train_end)
    test = slice(fold.test_start, fold.test_end)
    return X.loc[train], y.loc[train], X.loc[test], y.loc[test]


def folds_to_frame(ticker: str, folds: list[WalkForwardFold]) -> pd.DataFrame:
    """Fold design as a table, one row per (ticker, fold)."""
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "fold_id": f.fold_id,
                "test_year": f.test_year,
                "train_start": f.train_start.date(),
                "train_end": f.train_end.date(),
                "test_start": f.test_start.date(),
                "test_end": f.test_end.date(),
                "n_train": f.n_train,
                "n_test": f.n_test,
                "purge_rows": f.purge_rows,
            }
            for f in folds
        ]
    )

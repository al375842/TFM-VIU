"""M3 extension (2) - Walk-forward validation (robustness analysis).

The single split asks "did it work in 2025?"; this asks "does it work year
after year?". The series is walked in calendar-year blocks: at each step the
models are refitted on the data before the block and scored on the block
only, which gives one set of metrics per year.

It never touches the single-split tables of Ch. 5: everything is written
under its own `walkforward_*` names. The two leakage guards live in
`validation.make_walkforward_folds`.

Thesis: §4.4.7 (design) and §5.3 (results).
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone

from . import backtest as backtest_mod
from . import config, evaluation, features, metrics, validation
# build_model_specs is re-exported: the tests and notebook 06 use it from here
from .models import ModelSpec, build_model_specs, select_model_specs
from .validation import WalkForwardFold


@dataclass(frozen=True)
class _FoldFit:
    """How every fold is fitted: models, tuning mode and optional LSTM."""

    specs: dict[str, ModelSpec]
    retune: bool
    frozen_params: dict
    with_lstm: bool


@dataclass(frozen=True)
class _Collectors:
    """Optional lists for the outputs that do not fit in a metrics row."""

    financial: list | None = None
    prediction: list | None = None


# ---------------------------------------------------------------------------
# Hyper-parameters: retuned per fold, or frozen from the single-split run
# ---------------------------------------------------------------------------


def load_frozen_params(path=None) -> dict[tuple[str, str], dict]:
    """Hyper-parameters selected by the single-split experiment.

    Only for `retune=False` (smoke tests): those values were chosen with data
    up to 2024, so a 2016 fold would use a configuration picked with knowledge
    of its own future. Fine for checking the plumbing, never for results.
    """
    path = path or config.TABLES_DIR / "selected_hyperparameters.csv"
    table = pd.read_csv(path)
    frozen: dict[tuple[str, str], dict] = {}
    for _, row in table.iterrows():
        raw = str(row["selected_params"]).strip()
        params = ast.literal_eval(raw) if raw.startswith("{") else {}
        frozen[(row["ticker"], row["model"])] = params
    return frozen


def _fit_one(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    retune: bool,
    frozen: dict,
):
    """Fit a single model on one fold's training window."""
    if retune:
        return validation.tune_model(spec, X_train, y_train)
    est = clone(spec.estimator)
    if frozen:
        est.set_params(**frozen)
    return est.fit(X_train, y_train), frozen


# ---------------------------------------------------------------------------
# Stability test
# ---------------------------------------------------------------------------


def fold_sign_test(n_folds: int, n_above: int) -> float:
    """One-sided sign test on the number of folds that beat the naive floor.

    H0: the model beats its floor in half of the folds, like a coin toss.

    An approximation: under a strict null that probability is not exactly
    0.5, and the folds share training data. Read it as a measure of stability
    across years, not as proof of superiority. The per-fold binomial test also
    assumes independent trials and treats the estimated baseline as fixed.
    """
    if n_folds == 0:
        return float("nan")
    return float(
        stats.binomtest(int(n_above), int(n_folds), 0.5, alternative="greater").pvalue
    )


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------


def _fold_predictions(fold, X_train, y_train, X_test, *, fit, ticker,
                      dataset, cols) -> dict[str, pd.Series]:
    """Predictions of every model on the test block of one fold."""
    preds: dict[str, pd.Series] = {}
    for name, spec in fit.specs.items():
        model, selected_params = _fit_one(
            spec, X_train, y_train,
            retune=fit.retune, frozen=fit.frozen_params.get((ticker, name), {}),
        )
        preds[name] = pd.Series(model.predict(X_test), index=X_test.index)
        preds[name].attrs["selected_params"] = selected_params

    if fit.with_lstm:
        from . import lstm as lstm_mod  # lazy: TensorFlow is optional

        preds["lstm"] = lstm_mod.fit_predict_range(
            dataset.loc[fold.train_start:], cols,
            train_end=fold.train_end,     # already purged by the fold
            test_start=fold.test_start,
            test_end=fold.test_end,
        ).reindex(X_test.index)
    return preds


def _fold_rows(ticker, fold, preds, y_test, baseline_acc,
               prediction_out) -> list[dict]:
    """One classification row per model for this fold, plus its predictions."""
    rows = []
    for name, y_pred in preds.items():
        clf = metrics.classification_metrics(y_test, y_pred)
        h1 = metrics.binomial_test_vs_baseline(y_test, y_pred, baseline_acc)
        rows.append(
            {
                "ticker": ticker,
                "model": name,
                "fold_id": fold.fold_id,
                "test_year": fold.test_year,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "base_rate": baseline_acc,
                **clf,
                "balanced_accuracy": metrics.balanced_accuracy(y_test, y_pred),
                "share_predicted_up": float(np.mean(y_pred)),
                "above_baseline": bool(clf["accuracy"] > baseline_acc),
                "p_value_binomial": h1["p_value"],
                "selected_params": json.dumps(y_pred.attrs.get("selected_params", {}),
                                               sort_keys=True),
            }
        )
        if prediction_out is not None:
            prediction_out.append(
                pd.DataFrame(
                    {"ticker": ticker, "model": name,
                     "test_year": fold.test_year, "date": y_pred.index,
                     "prediction": y_pred.to_numpy()}
                )
            )
    return rows


def _fold_backtest(ticker, fold, prices, preds,
                   financial_out) -> dict[str, pd.Series]:
    """M4 inside one fold: returns the daily returns of B&H and each model."""
    prices_fold = backtest_mod.evaluation_prices(prices, [fold.test_start, fold.test_end])
    bh = backtest_mod.run_buy_and_hold(prices_fold, ticker)
    returns = {"buy_and_hold": bh["equity"].pct_change().fillna(0.0)}
    if financial_out is not None:
        financial_out.append(
            {"ticker": ticker, "model": "buy_and_hold",
             "fold_id": fold.fold_id, "test_year": fold.test_year,
             **bh["metrics"]}
        )
    for name, y_pred in preds.items():
        st = backtest_mod.run_strategy_backtest(prices_fold, y_pred, ticker)
        returns[name] = st["equity"].pct_change().fillna(0.0)
        if financial_out is not None:
            financial_out.append(
                {"ticker": ticker, "model": name,
                 "fold_id": fold.fold_id, "test_year": fold.test_year,
                 **st["metrics"],
                 **backtest_mod.compare_vs_benchmark(st, bh)}
            )
    return returns


def _print_fold(fold, rows, baseline_acc) -> None:
    """Progress line: the best model of the fold against that fold's floor."""
    best = max(rows, key=lambda r: r["balanced_accuracy"], default=None)
    if best is not None:
        print(f"    {fold.test_year}: n_test={fold.n_test:>3} "
              f"floor={baseline_acc:.3f} best={best['model']} "
              f"bacc={best['balanced_accuracy']:.3f}")


def _run_fold(fold, ticker, dataset, cols, X, y, *, fit, collectors, prices,
              verbose) -> tuple[list[dict], dict[str, pd.Series]]:
    """Train, score and (when `prices` is given) backtest a single fold.

    Returns its classification rows and the daily returns per model; both are
    empty if the training window has a single class.
    """
    X_train, y_train, X_test, y_test = validation.split_by_fold(X, y, fold)
    if y_train.nunique() < 2:  # degenerate window: nothing to learn
        if verbose:
            print(f"  [skip] {ticker} {fold.test_year}: single-class train")
        return [], {}

    # the floor comes from this fold's training window, so it follows the
    # trend of each period instead of being fixed for the whole history
    baseline_acc = metrics.naive_baseline_accuracy(y_train, y_test)
    preds = _fold_predictions(fold, X_train, y_train, X_test, fit=fit,
                              ticker=ticker, dataset=dataset, cols=cols)
    rows = _fold_rows(ticker, fold, preds, y_test, baseline_acc,
                      collectors.prediction)
    returns = (
        {} if prices is None
        else _fold_backtest(ticker, fold, prices, preds, collectors.financial)
    )
    if verbose:
        _print_fold(fold, rows, baseline_acc)
    return rows, returns


def _no_folds(ticker: str, verbose: bool) -> pd.DataFrame:
    """Empty result for an asset whose history yields no fold at all."""
    if verbose:
        print(f"  [skip] {ticker}: no fold meets the history requirement")
    return pd.DataFrame()


def _collect_equity(ticker, fold_returns, equity_out) -> None:
    """Chain the per-fold returns of each model into one compounded curve."""
    if equity_out is None:
        return
    chunks: dict[str, list[pd.Series]] = {}
    for returns in fold_returns:
        for name, series in returns.items():
            chunks.setdefault(name, []).append(series)
    for name, series_list in chunks.items():
        equity_out.append(_compound(series_list, ticker, name))


def _compound(chunks: list[pd.Series], ticker: str, model: str) -> pd.DataFrame:
    """Compound capital across folds, restarting positions at each boundary.

    The first return of each fold is zero for both curves, and positions are
    opened and closed again every year, so the Buy & Hold curve here is not a
    continuous buy-and-hold either.
    """
    returns = pd.concat(chunks).sort_index()
    equity = config.INITIAL_CASH * (1.0 + returns).cumprod()
    return pd.DataFrame(
        {"ticker": ticker, "model": model, "date": equity.index,
         "equity": equity.to_numpy()}
    )


# ---------------------------------------------------------------------------
# Per-asset sweep
# ---------------------------------------------------------------------------


def run_walkforward_asset(
    ticker: str,
    dataset: pd.DataFrame,
    model_specs: dict[str, ModelSpec],
    *,
    retune: bool = config.WF_RETUNE_EACH_FOLD,
    backtest: bool = False,
    prices: pd.DataFrame | None = None,
    folds: list[WalkForwardFold] | None = None,
    frozen_params: dict | None = None,
    with_lstm: bool = False,
    equity_out: list | None = None,
    financial_out: list | None = None,
    prediction_out: list | None = None,
    verbose: bool = True,
    **fold_kwargs,
) -> pd.DataFrame:
    """Fold-by-fold training and evaluation for one asset.

    Returns one row per (model, fold). What does not fit in a row (equity
    curves, financial metrics, raw predictions) goes to the optional
    collectors. `backtest=True` also simulates each fold and compounds the
    fold returns into one curve (see `_compound`).
    """
    if backtest and prices is None:
        raise ValueError("backtest=True needs the raw OHLCV frame in `prices`")
    folds = folds or validation.make_walkforward_folds(dataset.index, **fold_kwargs)
    if not folds:
        return _no_folds(ticker, verbose)

    cols = features.feature_columns(dataset)
    X, y = dataset[cols], dataset["target"]
    fit = _FoldFit(specs=model_specs, retune=retune,
                   frozen_params=frozen_params or {}, with_lstm=with_lstm)
    collectors = _Collectors(financial=financial_out, prediction=prediction_out)

    rows: list[dict] = []
    fold_returns: list[dict] = []
    for fold in folds:
        fold_rows, returns = _run_fold(
            fold, ticker, dataset, cols, X, y, fit=fit, collectors=collectors,
            prices=prices if backtest else None, verbose=verbose,
        )
        rows.extend(fold_rows)
        fold_returns.append(returns)

    if backtest:
        _collect_equity(ticker, fold_returns, equity_out)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_walkforward(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Collapse the fold rows into one row per (ticker, model).

    `n_folds` differs by asset (short crypto histories), so only the share of
    folds above the floor is comparable across assets, never the count.
    """
    if fold_results.empty:
        return pd.DataFrame()
    g = fold_results.groupby(["ticker", "model"])
    out = pd.DataFrame(
        {
            "n_folds": g.size(),
            "mean_accuracy": g["accuracy"].mean(),
            "std_accuracy": g["accuracy"].std(ddof=1),
            "median_accuracy": g["accuracy"].median(),
            "mean_balanced_accuracy": g["balanced_accuracy"].mean(),
            "mean_base_rate": g["base_rate"].mean(),
            "n_folds_above_baseline": g["above_baseline"].sum().astype(int),
            "mean_share_predicted_up": g["share_predicted_up"].mean(),
            # number of folds whose H1 binomial test gives p < 0.05
            "n_folds_p_below_05": g["p_value_binomial"].apply(
                lambda s: int((s < 0.05).sum())),
        }
    )
    out["share_folds_above"] = out["n_folds_above_baseline"] / out["n_folds"]
    out["p_value_sign_test"] = [
        fold_sign_test(n, k)
        for n, k in zip(out["n_folds"], out["n_folds_above_baseline"])
    ]
    return out[
        ["n_folds", "mean_accuracy", "std_accuracy", "median_accuracy",
         "mean_balanced_accuracy", "mean_base_rate", "n_folds_above_baseline",
         "share_folds_above", "p_value_sign_test", "mean_share_predicted_up",
         "n_folds_p_below_05"]
    ].sort_index()


def overall_financial(equity: pd.DataFrame) -> pd.DataFrame:
    """Financial metrics of the compounded curve, per (ticker, model).

    The per-fold table covers each year separately; this one covers all years
    at once, with positions restarted at each boundary.
    """
    rows = []
    for (ticker, model), part in equity.groupby(["ticker", "model"]):
        curve = pd.Series(part["equity"].to_numpy(),
                          index=pd.DatetimeIndex(part["date"])).sort_index()
        rows.append({"ticker": ticker, "model": model,
                     **metrics.financial_metrics(curve, ticker)})
    return pd.DataFrame(rows).set_index(["ticker", "model"]).sort_index()


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def _save_tables(fold_frames: list, clf_frames: list) -> dict:
    """Persist the fold design, the per-fold metrics and their summary."""
    # skip assets with no fold: their empty frames would only add NaN columns
    folds_table = pd.concat(
        [f for f in fold_frames if not f.empty], ignore_index=True
    ).set_index(["ticker", "fold_id"])
    clf_table = pd.concat(
        [f for f in clf_frames if not f.empty], ignore_index=True)
    summary = aggregate_walkforward(clf_table)

    evaluation.save_table(folds_table, "walkforward_folds")
    evaluation.save_table(
        clf_table.set_index(["ticker", "model", "fold_id"]).sort_index(),
        "walkforward_classification")
    evaluation.save_table(summary, "walkforward_summary")
    return {"folds": folds_table, "classification": clf_table,
            "summary": summary}


def _save_backtest_tables(financial_out: list, equity_out: list) -> dict:
    """Persist the per-fold financial metrics and the compounded curves."""
    fin = pd.DataFrame(financial_out).set_index(
        ["ticker", "model", "fold_id"]).sort_index()
    evaluation.save_table(fin, "walkforward_financial")
    equity = pd.concat(equity_out, ignore_index=True)
    equity.to_csv(config.TABLES_DIR / "walkforward_equity.csv", index=False)
    overall = overall_financial(equity)
    evaluation.save_table(overall, "walkforward_financial_overall")
    return {"financial": fin, "equity": equity, "overall": overall}


def run_walkforward(
    tickers: list[str] | None = None,
    offline: bool = True,
    technical: bool = True,
    model_names: list[str] | None = None,
    *,
    mode: str = config.WF_MODE,
    first_test_year: int = config.WF_FIRST_TEST_YEAR,
    last_test_year: int = config.WF_LAST_TEST_YEAR,
    retune: bool = config.WF_RETUNE_EACH_FOLD,
    with_backtest: bool = False,
    with_lstm: bool = False,
) -> dict:
    """Run the walk-forward robustness analysis over a list of assets."""
    from . import data as data_mod

    tickers = tickers or config.TICKERS
    raw = data_mod.load_raw(tickers) if offline else data_mod.download_data(tickers)
    specs = select_model_specs(model_names)
    frozen = None if retune else load_frozen_params()

    fold_frames, clf_frames = [], []
    equity_out, financial_out, prediction_out = [], [], []
    started = time.perf_counter()

    for ticker in tickers:
        print(f"=== {ticker} (walk-forward) ===")
        t0 = time.perf_counter()
        dataset = features.build_dataset(raw[ticker], technical=technical)
        folds = validation.make_walkforward_folds(
            dataset.index, mode=mode, first_test_year=first_test_year,
            last_test_year=last_test_year,
        )
        fold_frames.append(validation.folds_to_frame(ticker, folds))
        clf_frames.append(
            run_walkforward_asset(
                ticker, dataset, specs,
                retune=retune, backtest=with_backtest, prices=raw[ticker],
                folds=folds, frozen_params=frozen, with_lstm=with_lstm,
                equity_out=equity_out, financial_out=financial_out,
                prediction_out=prediction_out,
            )
        )
        print(f"  [{len(folds)} folds, {time.perf_counter() - t0:.1f}s]")

    result = _save_tables(fold_frames, clf_frames)
    if prediction_out:  # CSV only: too long to be a useful LaTeX fragment
        pd.concat(prediction_out, ignore_index=True).to_csv(
            config.TABLES_DIR / "walkforward_predictions.csv", index=False)
    if with_backtest and financial_out:
        result |= _save_backtest_tables(financial_out, equity_out)

    print(f"\n[walk-forward finished in {time.perf_counter() - started:.1f}s]")
    evaluation.build_walkforward_figures()
    return result

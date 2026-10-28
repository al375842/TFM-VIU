"""End-to-end orchestrator: M1 -> M2 -> M3 -> M4 -> M5.

Runs the whole experiment for a list of tickers and writes every table and
figure of Ch. 5 into results/. Each module also runs on its own from its
notebook (notebooks/01..05).

CLI usage (from the repo root, after `pip install -e .`):

    python -m tfm.pipeline                    # full 14-asset universe
    python -m tfm.pipeline --tickers AAPL "^GSPC"
    python -m tfm.pipeline --offline          # reuse persisted raw data
    python -m tfm.pipeline --no-technical     # skip the `ta` indicators
    python -m tfm.pipeline --offline --lstm   # include the LSTM extension

The walk-forward analysis (thesis §5.3) runs from the same entry point
behind a flag. With the flag it runs instead of the single-split experiment,
so the Ch. 5 tables are never overwritten:

    python -m tfm.pipeline --offline --walk-forward
    python -m tfm.pipeline --offline --walk-forward --wf-backtest --lstm
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import backtest, config, data, evaluation, features, metrics, validation
from .models import select_model_specs

# one list of rows per output table, filled asset by asset
_COLLECTIONS = ("classification", "hyperparameters", "predictions",
                "importances", "mcnemar", "financial", "equity")


def _equity_frame(equity: pd.Series, ticker: str, model: str) -> pd.DataFrame:
    """Long-format equity curve, normalised to 1.0 at the start.

    Normalising makes curves comparable across assets in one panel; the
    absolute figures stay in the financial table.
    """
    equity = equity.dropna()
    return pd.DataFrame(
        {
            "ticker": ticker,
            "model": model,
            "date": equity.index,
            "equity": (equity / equity.iloc[0]).to_numpy(),
        }
    )


def _load_data(tickers: list[str], offline: bool) -> dict[str, pd.DataFrame]:
    """M1: reuse the persisted series, or download and persist them."""
    if offline:
        return data.load_raw(tickers)
    raw = data.download_data(tickers)
    data.save_raw(raw)
    return raw


def _fit_models(specs, ticker, X_train, y_train, X_test, y_test):
    """M3: tune and fit every model, then predict the test block.

    Returns (fitted, predictions, metric rows, hyper-parameter rows).
    """
    fitted: dict[str, object] = {}
    preds: dict[str, pd.Series] = {}
    rows: list[dict] = []
    params: list[dict] = []
    for name, spec in specs.items():
        model, best_params = validation.tune_model(spec, X_train, y_train)
        fitted[name] = model
        y_pred = model.predict(X_test)
        preds[name] = pd.Series(y_pred, index=X_test.index)
        rows.append({"ticker": ticker, "model": name,
                     **metrics.classification_metrics(y_test, y_pred)})
        params.append(
            {
                "ticker": ticker,
                "model": name,
                # "-" for models without a grid (the naive reference)
                "selected_params": str(best_params) if best_params else "-",
                # share of the test set predicted as "up": 1.0 means the model
                # collapsed into the trivial majority-class rule
                "share_predicted_up": float(y_pred.mean()),
            }
        )
        print(f"  {name}: f1={rows[-1]['f1']:.3f} "
              f"acc={rows[-1]['accuracy']:.3f} {best_params}")
    return fitted, preds, rows, params


def _fit_lstm(ticker, dataset, cols, X_test, y_test):
    """M3 extension: the LSTM needs sequences, so it is fitted separately.

    Returns rows in the same format as `_fit_models`, for the same tables.
    """
    from . import lstm as lstm_mod  # lazy: TensorFlow is optional

    y_pred = lstm_mod.fit_predict(dataset, cols).reindex(X_test.index)
    row = {"ticker": ticker, "model": "lstm",
           **metrics.classification_metrics(y_test, y_pred)}
    params = {
        "ticker": ticker,
        "model": "lstm",
        "selected_params": (
            f"T={config.LSTM_WINDOW_T}, units={config.LSTM_UNITS}, "
            f"dropout={config.LSTM_DROPOUT}, "
            f"epochs={config.LSTM_EPOCHS} (fixed, no grid)"
        ),
        "share_predicted_up": float(y_pred.mean()),
    }
    print(f"  lstm: f1={row['f1']:.3f} acc={row['accuracy']:.3f}")
    return y_pred, row, params


def _prediction_frames(ticker: str, preds: dict) -> list[pd.DataFrame]:
    """Dated raw predictions: signals, not the positions executed later.

    No aggregate metric keeps them, so they are saved separately.
    """
    return [
        pd.DataFrame({"ticker": ticker, "model": name, "date": y_pred.index,
                      "prediction": y_pred.to_numpy()})
        for name, y_pred in preds.items()
    ]


def _importance_rows(ticker: str, fitted: dict, cols: list[str]) -> list[dict]:
    """Feature importances of the tree models, one row per feature."""
    rows: list[dict] = []
    for name, model in fitted.items():
        if not hasattr(model, "feature_importances_"):
            continue
        rows.extend(
            {"ticker": ticker, "model": name,
             "feature": feat, "importance": float(imp)}
            for feat, imp in zip(cols, model.feature_importances_)
        )
    return rows


def _mcnemar_rows(ticker: str, y_test, preds: dict) -> list[dict]:
    """H2: McNemar test for every pair of ML models on the same test cases."""
    contenders = [n for n in preds if n != "naive"]
    rows = []
    for i, a in enumerate(contenders):
        for b in contenders[i + 1:]:
            rows.append({"ticker": ticker, "model_a": a, "model_b": b,
                         **metrics.mcnemar_test(y_test, preds[a], preds[b])})
    return rows


def _add_h1_tests(rows: list[dict], preds: dict, y_train, y_test) -> float:
    """H1: add the naive floor and its binomial p-value to each metric row.

    Returns the floor: a property of the asset, not of the model.
    """
    baseline_acc = metrics.naive_baseline_accuracy(y_train, y_test)
    for row in rows:
        h1 = metrics.binomial_test_vs_baseline(
            y_test, preds[row["model"]], baseline_acc
        )
        row["base_rate"] = baseline_acc
        # share of "up" in training: the naive model learns its majority class
        # there, and that class sets the test floor
        row["base_rate_train"] = float(y_train.mean())
        row["p_value"] = h1["p_value"]
    return baseline_acc


def _run_backtests(ticker, prices_test, preds, backtest_models):
    """M4 + M5: Buy & Hold first, then each strategy against it.

    Returns (financial rows, equity frames); draws one figure per model.
    """
    bh = backtest.run_buy_and_hold(prices_test, ticker)
    fin_rows = [{"ticker": ticker, "model": "buy_and_hold", **bh["metrics"]}]
    frames = [_equity_frame(bh["equity"], ticker, "buy_and_hold")]
    if not backtest_models:
        return fin_rows, frames

    # iterate the predictions, not the fitted estimators: the LSTM has
    # predictions but no sklearn-compatible estimator
    for name, y_pred in preds.items():
        st = backtest.run_strategy_backtest(prices_test, y_pred, ticker)
        verdict = backtest.compare_vs_benchmark(st, bh)
        fin_rows.append({"ticker": ticker, "model": name,
                         **st["metrics"], **verdict})
        frames.append(_equity_frame(st["equity"], ticker, name))
        evaluation.plot_equity_curves(st["equity"], bh["equity"], ticker, name)
    return fin_rows, frames


def _process_ticker(ticker, prices, specs, *, technical, backtest_models,
                    with_lstm) -> dict | None:
    """Full M2 -> M5 chain for one asset.

    One list of rows per output table, or None if the asset has no test data.
    """
    # ------------------------------------------------------------------ M2
    dataset = features.build_dataset(prices, technical=technical)
    train, test = validation.temporal_split(dataset)
    if len(test) == 0:
        print(f"  [skip] no test data for {ticker} after {config.TEST_START}")
        return None
    cols = features.feature_columns(dataset)
    X_train, y_train = train[cols], train["target"]
    X_test, y_test = test[cols], test["target"]

    # ------------------------------------------------------------------ M3
    fitted, preds, clf_rows, param_rows = _fit_models(
        specs, ticker, X_train, y_train, X_test, y_test
    )
    if with_lstm:
        y_pred, row, params = _fit_lstm(ticker, dataset, cols, X_test, y_test)
        preds["lstm"] = y_pred
        clf_rows.append(row)
        param_rows.append(params)

    pred_frames = _prediction_frames(ticker, preds)
    importances = _importance_rows(ticker, fitted, cols)
    baseline_acc = _add_h1_tests(clf_rows, preds, y_train, y_test)
    print(f"  [naive floor = {baseline_acc:.4f}]")
    mcnemar_rows = _mcnemar_rows(ticker, y_test, preds)

    # ------------------------------------------------------------- M4 + M5
    fin_rows, equity_frames = _run_backtests(
        ticker, backtest.evaluation_prices(prices, test.index), preds, backtest_models
    )
    evaluation.plot_roc(fitted, X_test, y_test, ticker)
    for name in ("random_forest", "xgboost"):
        if name in fitted:
            evaluation.plot_feature_importance(fitted[name], cols, ticker, name)

    return {
        "classification": clf_rows,
        "hyperparameters": param_rows,
        "predictions": pred_frames,
        "importances": importances,
        "mcnemar": mcnemar_rows,
        "financial": fin_rows,
        "equity": equity_frames,
    }


def _save_outputs(collected: dict, ranges: pd.DataFrame) -> dict:
    """M5: assemble the result tables and persist them to results/tables."""
    clf_table = evaluation.classification_table(collected["classification"])
    fin_table = evaluation.financial_table(collected["financial"])
    mcnemar_table = evaluation.mcnemar_table(collected["mcnemar"])
    param_table = (
        pd.DataFrame(collected["hyperparameters"])
        .set_index(["ticker", "model"]).sort_index()
    )
    evaluation.save_table(clf_table, "classification_metrics")
    evaluation.save_table(fin_table, "financial_metrics")
    evaluation.save_table(mcnemar_table, "mcnemar_tests")
    evaluation.save_table(param_table, "selected_hyperparameters")
    evaluation.save_table(ranges, "date_ranges")

    # the two long tables (~26k and ~22k rows) are saved as CSV only
    if collected["equity"]:
        pd.concat(collected["equity"], ignore_index=True).to_csv(
            config.TABLES_DIR / "equity_curves.csv", index=False)
    if collected["predictions"]:
        pd.concat(collected["predictions"], ignore_index=True).to_csv(
            config.TABLES_DIR / "predictions.csv", index=False)
    if collected["importances"]:
        evaluation.save_table(
            pd.DataFrame(collected["importances"]).set_index(
                ["ticker", "model", "feature"]).sort_index(),
            "feature_importances")

    return {
        "classification": clf_table,
        "financial": fin_table,
        "mcnemar": mcnemar_table,
        "hyperparameters": param_table,
        "ranges": ranges,
    }


def run_pipeline(
    tickers: list[str] | None = None,
    offline: bool = False,
    technical: bool = True,
    model_names: list[str] | None = None,
    backtest_models: bool = True,
    with_lstm: bool = False,
) -> dict:
    """Run the full experiment and return the consolidated results."""
    tickers = tickers or config.TICKERS
    raw = _load_data(tickers, offline)
    ranges = data.report_date_ranges(raw)
    print(ranges, "\n")
    specs = select_model_specs(model_names)

    collected: dict[str, list] = {name: [] for name in _COLLECTIONS}
    for ticker in tickers:
        print(f"=== {ticker} ===")
        result = _process_ticker(
            ticker, raw[ticker], specs, technical=technical,
            backtest_models=backtest_models, with_lstm=with_lstm,
        )
        if result is None:
            continue
        for name, rows in result.items():
            collected[name].extend(rows)

    return _save_outputs(collected, ranges)


def main() -> None:
    parser = argparse.ArgumentParser(description="TFM experimental pipeline")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--offline", action="store_true",
                        help="reuse persisted raw data instead of downloading")
    parser.add_argument("--no-technical", action="store_true",
                        help="skip the `ta` technical indicators")
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of models (naive logreg random_forest xgboost svm)")
    parser.add_argument("--no-backtest", action="store_true",
                        help="skip M4 (classification metrics only)")
    parser.add_argument("--lstm", action="store_true",
                        help="also train the LSTM extension (needs TensorFlow)")

    # walk-forward is off by default: without the flag, the run is the
    # single-split experiment, unchanged
    wf = parser.add_argument_group("walk-forward (robustness analysis)")
    wf.add_argument("--walk-forward", action="store_true",
                    help="run the walk-forward analysis INSTEAD of the "
                         "single-split sweep")
    wf.add_argument("--wf-mode", choices=("anchored", "rolling"),
                    default=config.WF_MODE,
                    help="expanding (anchored) or fixed-width (rolling) "
                         "training window")
    wf.add_argument("--wf-first-test-year", type=int,
                    default=config.WF_FIRST_TEST_YEAR,
                    help="first calendar year used as a test fold")
    wf.add_argument("--wf-last-test-year", type=int,
                    default=config.WF_LAST_TEST_YEAR,
                    help="last calendar year used as a test fold")
    wf.add_argument("--wf-no-retune", action="store_true",
                    help="freeze the hyper-parameters of the single-split run "
                         "instead of retuning per fold (fast smoke mode)")
    wf.add_argument("--wf-backtest", action="store_true",
                    help="also backtest each fold and compound the equity "
                         "curve across folds")
    args = parser.parse_args()

    if args.walk_forward:
        from . import walkforward

        walkforward.run_walkforward(
            tickers=args.tickers,
            offline=args.offline,
            technical=not args.no_technical,
            model_names=args.models,
            mode=args.wf_mode,
            first_test_year=args.wf_first_test_year,
            last_test_year=args.wf_last_test_year,
            retune=not args.wf_no_retune,
            with_backtest=args.wf_backtest,
            with_lstm=args.lstm,
        )
        return

    run_pipeline(
        tickers=args.tickers,
        offline=args.offline,
        technical=not args.no_technical,
        model_names=args.models,
        backtest_models=not args.no_backtest,
        with_lstm=args.lstm,
    )


if __name__ == "__main__":
    main()

"""M5 - Evaluation and visualisation.

Builds the tables and figures of Ch. 5 out of the results of M3 and M4:
comparison tables, ROC curves, feature importance and equity curves vs
Buy & Hold. Everything is written to results/tables and results/figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay

from . import config


def _ensure_dirs() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def save_table(df: pd.DataFrame, name: str) -> None:
    """Persist a results table as CSV and as a LaTeX booktabs fragment."""
    _ensure_dirs()
    df.to_csv(config.TABLES_DIR / f"{name}.csv")
    df.to_latex(
        config.TABLES_DIR / f"{name}.tex", float_format="%.4f", escape=True
    )


def classification_table(results: list[dict]) -> pd.DataFrame:
    """Long-format table: one row per (asset, model) with ML metrics."""
    return pd.DataFrame(results).set_index(["ticker", "model"]).sort_index()


def financial_table(results: list[dict]) -> pd.DataFrame:
    """One row per (asset, model/benchmark): metrics + the vs-B&H verdict."""
    return pd.DataFrame(results).set_index(["ticker", "model"]).sort_index()


def mcnemar_table(results: list[dict]) -> pd.DataFrame:
    """Pairwise McNemar tests (H2): one row per (asset, model A, model B)."""
    if not results:
        return pd.DataFrame(
            columns=["ticker", "model_a", "model_b", "b", "c",
                     "n_discordant", "p_value"]
        ).set_index(["ticker", "model_a", "model_b"])
    return pd.DataFrame(results).set_index(
        ["ticker", "model_a", "model_b"]
    ).sort_index()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_roc(fitted_models: dict, X_test, y_test, ticker: str) -> None:
    """ROC curves of every model with predict_proba, for one asset.

    The naive model would only redraw the chance diagonal, so it is left out.
    """
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in fitted_models.items():
        if name != "naive" and hasattr(model, "predict_proba"):
            RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)  # random-classifier diagonal
    ax.set_title(f"ROC curves - {ticker} (test 2025)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"roc_{_safe(ticker)}.png", dpi=200)
    plt.close(fig)


def plot_feature_importance(model, feature_names: list[str], ticker: str,
                            model_name: str, top_n: int = 15) -> None:
    """Bar chart of tree-based feature importances (RF / XGBoost)."""
    _ensure_dirs()
    if not hasattr(model, "feature_importances_"):
        return
    imp = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=True)
        .tail(top_n)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax)
    ax.set_title(f"Feature importance - {model_name} - {ticker}")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(
        config.FIGURES_DIR / f"featimp_{model_name}_{_safe(ticker)}.png", dpi=200
    )
    plt.close(fig)


def plot_equity_curves(strategy_equity: pd.Series, benchmark_equity: pd.Series,
                       ticker: str, model_name: str) -> None:
    """Strategy vs Buy & Hold equity curves over the test period."""
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 5))
    strategy_equity.plot(ax=ax, label=f"ML strategy ({model_name})")
    benchmark_equity.plot(ax=ax, label="Buy & Hold", linestyle="--")
    ax.set_title(f"Equity curve - {ticker} (test 2025)")
    ax.set_ylabel("equity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        config.FIGURES_DIR / f"equity_{model_name}_{_safe(ticker)}.png", dpi=200
    )
    plt.close(fig)


def _safe(ticker: str) -> str:
    return ticker.replace("^", "").replace(".", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Cross-asset summary figures (Ch. 5)
#
# The figures above show one asset at a time; these compare all of them. They
# read the persisted tables, so they can be rebuilt without refitting.
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# one fixed colour per model, in this order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MODEL_COLOR = {
    "logreg": SERIES[0],
    "random_forest": SERIES[1],
    "xgboost": SERIES[2],
    "svm": SERIES[3],
    "lstm": SERIES[4],
}
MODEL_LABEL = {
    "logreg": "Reg. logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "SVM (RBF)",
    "lstm": "LSTM",
}
# plot only the models present in the tables, so the figures work with or
# without the optional LSTM
ALL_MODELS = list(MODEL_COLOR)


def _models_present(df: pd.DataFrame) -> list[str]:
    found = set(df.model)
    return [m for m in ALL_MODELS if m in found]

# Diverging pair (warm/cool) and its neutral midpoint
DIVERGING_POS, DIVERGING_NEG, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
DE_EMPHASIS = "#c3c2b7"


def _style_axes(ax, *, xgrid: bool = False, ygrid: bool = True) -> None:
    """Minimal axis style: light grid, no frame, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=1.0, linestyle="-")
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)


def _titles(fig, title: str, subtitle: str) -> None:
    """Left-aligned title and subtitle at the top of the figure."""
    fig.text(0.012, 0.975, title, ha="left", va="top",
             fontsize=15, fontweight="600", color=INK)
    fig.text(0.012, 0.928, subtitle, ha="left", va="top",
             fontsize=10.5, color=INK_SECONDARY)


def _save(fig, name: str) -> None:
    _ensure_dirs()
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(config.TABLES_DIR / f"{name}.csv")


def _pct(x: float, digits: int = 0) -> str:
    """Percentage with a Spanish decimal comma. Format the number only:
    replacing "." in a whole sentence would also replace its full stops."""
    return f"{x:.{digits}%}".replace(".", ",").replace("%", " %")


def plot_h1_gap() -> None:
    """Accuracy minus the asset's naive floor, for every model (H1).

    The zero line is the naive classifier of each asset, not a fixed 50%.
    """
    clf = _load("classification_metrics")
    models = _models_present(clf)
    clf = clf[clf.model.isin(models)].copy()
    clf["gap"] = clf.accuracy - clf.base_rate

    order = (clf.groupby("ticker").gap.mean().sort_values().index.tolist())
    # one lollipop per model, plus a blank row so the asset groups stay apart
    step = len(models) + 1
    ypos, ticks, labels = {}, [], []
    y = 0.0
    for ticker in order:
        ticks.append(y + (len(models) - 1) / 2)
        labels.append(ticker)
        for i, m in enumerate(models):
            ypos[(ticker, m)] = y + i
        y += step

    fig, ax = plt.subplots(figsize=(9.5, 2.2 + 0.72 * len(order)))
    fig.patch.set_facecolor(SURFACE)
    for _, r in clf.iterrows():
        yy = ypos[(r.ticker, r.model)]
        c = MODEL_COLOR[r.model]
        ax.plot([0, r.gap], [yy, yy], color=c, linewidth=1.6,
                solid_capstyle="round", zorder=2)
        ax.plot(r.gap, yy, "o", color=c, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)

    ax.axvline(0, color=INK, linewidth=1.4, zorder=4)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=10, color=INK_SECONDARY)
    ax.set_ylim(-1.5, y - step + len(models) + 0.5)
    ax.invert_yaxis()
    _style_axes(ax, xgrid=True, ygrid=False)
    ax.set_xlabel("Accuracy menos el suelo naïve del activo",
                  fontsize=10, color=INK_SECONDARY, labelpad=10)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.2f}".replace(".", ","))
    )
    # room on the right for the annotation
    lo, hi = clf.gap.min(), clf.gap.max()
    ax.set_xlim(lo - 0.012, hi + 0.045)

    # label the pairs with p < 0.05
    for _, r in clf.loc[clf.p_value < 0.05].iterrows():
        ax.annotate(
            f"único con p < 0,05  (p = {r.p_value:.4f})".replace(".", ","),
            xy=(r.gap, ypos[(r.ticker, r.model)]),
            xytext=(8, 0), textcoords="offset points",
            va="center", ha="left", fontsize=9, color=INK, fontweight="600",
        )

    handles = [
        plt.Line2D([], [], color=MODEL_COLOR[m], marker="o", linestyle="-",
                   linewidth=1.6, markersize=7, label=MODEL_LABEL[m])
        for m in models
    ]
    # legend above the plot: the data fills the whole panel
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.008),
              frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
              ncol=len(models), columnspacing=1.5, handletextpad=0.5)
    # keep the top strip free for the title and the legend
    fig.subplots_adjust(top=0.855)

    n_above = int((clf.gap > 0).sum())
    _titles(
        fig,
        "Accuracy frente a la referencia ingenua de cada activo",
        f"Diferencia de accuracy frente al suelo naïve de cada activo. "
        f"{n_above} de {len(clf)} pares quedan por encima.\nA la izquierda de la línea, el modelo "
        f"acierta menos que predecir siempre «sube». Test de 2025.",
    )
    _save(fig, "summary_h1_gap")


def plot_pvalue_distributions() -> None:
    """Histograms of the raw H1 and H2 p-values (descriptive only).

    With discrete tests, p-values need not be uniform even under H0.
    """
    clf = _load("classification_metrics")
    h1 = clf[clf.model.isin(_models_present(clf))].p_value.dropna().to_numpy()
    h2 = _load("mcnemar_tests").p_value.dropna().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.patch.set_facecolor(SURFACE)
    bins = [i / 10 for i in range(11)]

    panels = [
        (axes[0], h1, "H1 · binomial frente al suelo naïve",
         f"{len(h1)} pares activo-modelo"),
        (axes[1], h2, "H2 · McNemar entre modelos",
         f"{len(h2)} pares de modelos"),
    ]
    for ax, vals, title, note in panels:
        heights, _ = np.histogram(vals, bins=bins)
        centers = [(bins[i] + bins[i + 1]) / 2 for i in range(10)]
        # a bit narrower than the bin, to leave a gap between bars
        ax.bar(centers, heights, width=0.086, color=SERIES[0], zorder=2)
        ax.axvline(0.05, color=INK_MUTED, linewidth=1.2, zorder=3)
        ax.axvspan(0, 0.05, color=DIVERGING_NEG, alpha=0.07, zorder=1)
        ax.set_ylim(0, max(heights) * 1.28)
        n_sig = int((vals < 0.05).sum())
        ax.annotate(
            f"p < 0,05\n{n_sig} de {len(vals)}",
            xy=(0.06, max(heights) * 1.17), ha="left", va="top", fontsize=9.5,
            color=INK, fontweight="600", zorder=5,
        )
        _style_axes(ax)
        # the y axis counts tests: integer ticks only
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title(title, fontsize=11, color=INK, fontweight="600",
                     loc="left", pad=10)
        ax.set_xlabel(f"p-valor · {note}", fontsize=9.5,
                      color=INK_SECONDARY, labelpad=8)
        ax.set_xlim(0, 1)
        ax.set_ylabel("nº de contrastes", fontsize=9.5, color=INK_SECONDARY)

    _titles(
        fig,
        "Distribución de p-valores exploratorios",
        "P-valores brutos, sin corrección por comparaciones múltiples. "
        "La línea vertical marca 0,05.\nLas pruebas discretas y la dependencia "
        "entre observaciones impiden interpretar una distribución uniforme como nulo exacto.",
    )
    fig.subplots_adjust(top=0.78, wspace=0.22)
    _save(fig, "summary_pvalues")


def plot_h3_exposure() -> None:
    """Buy & Hold return in 2025 against how many models beat it, per asset.

    Descriptive: the association alone does not identify its cause.
    """
    fin = _load("financial_metrics")
    bh = fin[fin.model == "buy_and_hold"].set_index("ticker")
    st = fin[fin.model.isin(_models_present(fin))]

    agg = st.groupby("ticker").agg(
        wins=("adds_value", "sum"),
        strat_vol=("volatility", "mean"),
    )
    agg["bh_return"] = bh.cumulative_return
    agg["bh_vol"] = bh.volatility
    agg["vol_ratio"] = agg.strat_vol / agg.bh_vol
    r = agg.bh_return.corr(agg.wins)

    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.patch.set_facecolor(SURFACE)

    ax.axvspan(agg.bh_return.min() - 0.05, 0, color=DIVERGING_NEG,
               alpha=0.06, zorder=0)
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)

    # labels alternate above/below when two assets sit close on the same row
    span = agg.bh_return.max() - agg.bh_return.min()
    for _, level in agg.groupby("wins"):
        level = level.sort_values("bh_return")
        offset, prev_x = 13, None
        for ticker, row in level.iterrows():
            if prev_x is not None and (row.bh_return - prev_x) < 0.10 * span:
                offset = -20 if offset == 13 else 13
            else:
                offset = 13
            prev_x = row.bh_return
            color = SERIES[1] if row.bh_return < 0 else SERIES[0]
            ax.plot(row.bh_return, row.wins, "o", markersize=11, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate(ticker, xy=(row.bh_return, row.wins),
                        xytext=(0, offset), textcoords="offset points",
                        ha="center", fontsize=9, color=INK_SECONDARY)

    _style_axes(ax)
    ax.set_xlabel("Rentabilidad de Buy & Hold en 2025", fontsize=10,
                  color=INK_SECONDARY, labelpad=10)
    n_models = len(_models_present(fin))
    ax.set_ylabel("Modelos que baten a B&H\n(rentabilidad y Sharpe)",
                  fontsize=10, color=INK_SECONDARY)
    ax.set_yticks(range(n_models + 1))
    ax.set_ylim(-0.5, n_models + 0.7)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:+.0%}".replace("%", " %"))
    )
    ax.annotate(
        "Buy & Hold en pérdidas", xy=(agg.bh_return.min() / 2, n_models + 0.45),
        ha="center", fontsize=9, color=SERIES[1], fontweight="600",
    )

    r_txt = f"{r:.3f}".replace(".", ",")
    losers = agg[agg.bh_return < 0]
    winners = agg.nlargest(2, "bh_return")
    _titles(
        fig,
        "Cuanto peor le fue al activo, más fácil resultó «batirlo»",
        f"Cada punto es un activo. En los {len(losers)} con Buy & Hold en "
        f"pérdidas, {int(losers.wins.min())} de los {n_models} modelos aportan "
        f"valor; en los dos más alcistas, {int(winners.wins.max())}.\n"
        f"Correlación descriptiva r = {r_txt}. Esta asociación no identifica "
        "por sí sola la causa de las diferencias financieras.",
    )
    fig.subplots_adjust(top=0.86)
    _save(fig, "summary_h3_exposure")


def plot_risk_return() -> None:
    """Volatility and 2025 return of every strategy next to its Buy & Hold.

    One row per asset: volatility on the left, cumulative return on the right.
    """
    fin = _load("financial_metrics")
    bh = fin[fin.model == "buy_and_hold"].set_index("ticker")
    st = fin[fin.model.isin(_models_present(fin))]

    order = bh.cumulative_return.sort_values().index.tolist()
    ypos = {t: i for i, t in enumerate(order)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    panels = [
        (axes[0], "volatility", "Volatilidad anualizada", "{:.0%}", False),
        (axes[1], "cumulative_return", "Rentabilidad acumulada 2025",
         "{:+.0%}", True),
    ]
    for ax, col, xlabel, fmt, zero_line in panels:
        for ticker in order:
            yy = ypos[ticker]
            vals = st.loc[st.ticker == ticker, col]
            b = bh.loc[ticker, col]
            ax.plot([min(vals.min(), b), max(vals.max(), b)], [yy, yy],
                    color=DE_EMPHASIS, linewidth=1.3,
                    solid_capstyle="round", zorder=1)
            ax.plot(vals, [yy] * len(vals), "o", markersize=7.5,
                    color=SERIES[0], markeredgecolor=SURFACE,
                    markeredgewidth=1.6, zorder=3)
            ax.plot(b, yy, "o", markersize=11.5, color=SERIES[1],
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)

        if zero_line:
            ax.axvline(0, color=AXIS, linewidth=1.0, zorder=2)
        _style_axes(ax, xgrid=True, ygrid=False)
        ax.set_xlabel(xlabel, fontsize=10, color=INK_SECONDARY, labelpad=10)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _, f=fmt: f.format(v).replace("%", " %")))

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=10, color=INK_SECONDARY)
    axes[0].set_ylim(-0.8, len(order) - 0.2)

    lower = int((st.volatility.to_numpy()
                 < bh.loc[st.ticker].volatility.to_numpy()).sum())
    higher = int((st.cumulative_return.to_numpy()
                  > bh.loc[st.ticker].cumulative_return.to_numpy()).sum())

    handles = [
        plt.Line2D([], [], color=SERIES[0], marker="o", linestyle="none",
                   markersize=7.5,
                   label=f"Estrategia ML ({len(_models_present(fin))} por activo)"),
        plt.Line2D([], [], color=SERIES[1], marker="o", linestyle="none",
                   markersize=11.5, label="Buy & Hold del activo"),
    ]
    axes[0].legend(handles=handles, loc="lower left",
                   bbox_to_anchor=(0, 1.005), frameon=False, fontsize=9.5,
                   labelcolor=INK_SECONDARY, ncol=2, handletextpad=0.5)

    _titles(
        fig,
        "Rentabilidad y volatilidad frente a Buy & Hold",
        f"Mismo eje, una dimensión por panel. {lower} de {len(st)} estrategias "
        f"reducen la volatilidad frente a su Buy & Hold, pero solo {higher} de "
        f"{len(st)} mejoran la rentabilidad.\nLa menor volatilidad no implica "
        f"por sí sola un Sharpe mayor. Activos ordenados por "
        f"rentabilidad del Buy & Hold.",
    )
    fig.subplots_adjust(top=0.85, wspace=0.06)
    _save(fig, "summary_risk_return")


def plot_model_bias() -> None:
    """Share of test sessions each model predicts as "up", per asset.

    A model that predicts the majority class every session is the naive
    reference under another name, whatever its accuracy.
    """
    hp = _load("selected_hyperparameters")
    hp = hp[hp.model.isin(_models_present(hp))]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(SURFACE)

    order = list(reversed(_models_present(hp)))
    for i, m in enumerate(order):
        vals = hp.loc[hp.model == m, "share_predicted_up"]
        mean = vals.mean()
        ax.plot([vals.min(), vals.max()], [i, i], color=DE_EMPHASIS,
                linewidth=1.4, solid_capstyle="round", zorder=1)
        ax.plot(vals, [i] * len(vals), "o", markersize=8,
                color=MODEL_COLOR[m], markeredgecolor=SURFACE,
                markeredgewidth=1.6, alpha=0.9, zorder=3)
        ax.plot(mean, i, "|", markersize=22, markeredgewidth=2.4,
                color=INK, zorder=4)
        ax.annotate(f"media {mean:.2f}".replace(".", ","),
                    xy=(mean, i), xytext=(0, 15), textcoords="offset points",
                    ha="center", fontsize=9, color=INK, fontweight="600")

    ax.axvline(1.0, color=DIVERGING_NEG, linewidth=1.6, zorder=2)
    ax.annotate("colapso en el\nclasificador trivial", xy=(1.0, -0.62),
                xytext=(-8, 0), textcoords="offset points",
                ha="right", va="bottom", fontsize=9, color=DIVERGING_NEG,
                fontweight="600")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in order], fontsize=10,
                       color=INK_SECONDARY)
    ax.set_ylim(-0.85, len(order) - 0.4)
    _style_axes(ax, xgrid=True, ygrid=False)
    ax.set_xlabel("Proporción del test predicha como «sube» · un punto por activo",
                  fontsize=10, color=INK_SECONDARY, labelpad=10)
    ax.set_xlim(0, 1.06)

    means = hp.groupby("model").share_predicted_up.mean()
    top, bottom = means.idxmax(), means.idxmin()
    collapsed = int((hp.share_predicted_up >= 0.99).sum())
    pct_top = f"{means[top]:.0%}".replace("%", " %")
    pct_bottom = f"{means[bottom]:.0%}".replace("%", " %")
    _titles(
        fig,
        "Los modelos lineales y de kernel se rinden ante la clase mayoritaria",
        f"{MODEL_LABEL[top]} predice «sube» en el {pct_top} de las sesiones "
        f"de media, frente al {pct_bottom} de {MODEL_LABEL[bottom]}.\n"
        f"{collapsed} pares llegan al 100 %: dejan de ser un modelo y pasan a "
        f"ser la referencia ingenua con otro nombre.",
    )
    fig.subplots_adjust(top=0.82)
    _save(fig, "summary_model_bias")


def plot_equity_grid() -> None:
    """Every backtest equity curve on one page, normalised to 1 at the start."""
    eq = _load("equity_curves")
    eq["date"] = pd.to_datetime(eq["date"])
    fin = _load("financial_metrics")
    bh = fin[fin.model == "buy_and_hold"].set_index("ticker")
    tickers = [t for t in config.TICKERS if t in set(eq.ticker)]

    ncols, nrows = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12), sharex=False)
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel()

    for ax, ticker in zip(axes, tickers):
        sub = eq[eq.ticker == ticker]
        for m in _models_present(eq):
            s = sub[sub.model == m]
            ax.plot(s.date, s.equity, color=SERIES[0], linewidth=1.1,
                    alpha=0.55, zorder=2)
        b = sub[sub.model == "buy_and_hold"]
        ax.plot(b.date, b.equity, color=INK_MUTED, linewidth=2.0, zorder=3)
        ax.axhline(1.0, color=GRID, linewidth=1.0, zorder=1)

        _style_axes(ax)
        ax.set_title(ticker, fontsize=10.5, color=INK, fontweight="600",
                     loc="left", pad=6)
        ret = bh.loc[ticker, "cumulative_return"]
        ax.annotate(f"B&H {ret:+.1%}".replace(".", ",").replace("%", " %"),
                    xy=(0.98, 0.06), xycoords="axes fraction", ha="right",
                    fontsize=8.5, color=INK_MUTED)
        ax.tick_params(labelsize=8)
        ax.set_xticks([])

    for ax in axes[len(tickers):]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([], [], color=SERIES[0], linewidth=1.6, alpha=0.7,
                   label=f"Estrategias ML ({len(_models_present(eq))} modelos)"),
        plt.Line2D([], [], color=INK_MUTED, linewidth=2.0,
                   label="Buy & Hold (referencia)"),
    ]
    fig.legend(handles=handles, loc="lower right", frameon=False,
               fontsize=10, labelcolor=INK_SECONDARY,
               bbox_to_anchor=(0.98, 0.055), ncol=2)

    _titles(
        fig,
        "Catorce backtests, capital normalizado a 1 al inicio de 2025",
        "Capital de cada estrategia frente al benchmark, sin costes de transacción.\n"
        "Cada activo se evalúa entre su primera y última señal válida de 2025; "
        "el IBEX termina el 21 de noviembre.",
    )
    fig.subplots_adjust(top=0.86, hspace=0.34, wspace=0.16, bottom=0.08)
    _save(fig, "summary_equity_grid")


def plot_exposure_map() -> None:
    """Dated "up" predictions per asset and model (signals, not positions)."""
    pr = _load("predictions")
    pr["date"] = pd.to_datetime(pr["date"])
    models = _models_present(pr)
    # ML models only: the naive model predicts "up" almost every day and
    # would inflate the average share of positive signals
    pr = pr[pr.model.isin(models)]
    tickers = [t for t in config.TICKERS if t in set(pr.ticker)]

    rows = [(t, m) for t in tickers for m in models]
    fig, ax = plt.subplots(figsize=(13, 0.17 * len(rows) + 3.4))
    fig.patch.set_facecolor(SURFACE)

    y, yticks, ylabels = 0.0, [], []
    for ticker in tickers:
        yticks.append(y + (len(models) - 1) / 2)
        ylabels.append(ticker)
        for m in models:
            s = pr[(pr.ticker == ticker) & (pr.model == m)].sort_values("date")
            long_days = s[s.prediction == 1].date
            ax.plot(long_days, [y] * len(long_days), "|",
                    color=MODEL_COLOR[m], markersize=5.5,
                    markeredgewidth=1.1, zorder=2)
            share = s.prediction.mean()
            ax.annotate(f"{share:.0%}".replace("%", " %"),
                        xy=(1.005, y), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=7.5,
                        color=MODEL_COLOR[m])
            y += 1
        y += 1.2  # gap between assets

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylim(-1, y - 0.6)
    ax.invert_yaxis()
    _style_axes(ax, xgrid=True, ygrid=False)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Sesiones de 2025 · una marca por predicción de subida",
                  fontsize=10, color=INK_SECONDARY, labelpad=10)

    handles = [
        plt.Line2D([], [], color=MODEL_COLOR[m], marker="|", linestyle="none",
                   markersize=9, markeredgewidth=1.8, label=MODEL_LABEL[m])
        for m in models
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.004),
              frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
              ncol=len(models), columnspacing=1.5, handletextpad=0.5)

    share_txt = f"{pr.prediction.mean():.0%}".replace("%", " %")
    _titles(
        fig,
        "Señales positivas por activo y modelo",
        f"Cada marca es una predicción de subida; {share_txt} de las señales "
        f"agrupadas son positivas.\nEl porcentaje por fila es su frecuencia de "
        f"señales positivas. Las órdenes se ejecutan después: no equivale a tiempo invertido.",
    )
    fig.subplots_adjust(top=0.88, right=0.955)
    _save(fig, "summary_exposure_map")


def plot_gap_heatmap() -> None:
    """Accuracy minus the naive floor for every asset and model (H1).

    Blue beats the asset's naive floor; red falls short of it.
    """
    clf = _load("classification_metrics")
    models = _models_present(clf)
    clf = clf[clf.model.isin(models)].copy()
    clf["gap"] = clf.accuracy - clf.base_rate

    grid = clf.pivot(index="ticker", columns="model", values="gap")
    grid = grid.reindex(index=grid.mean(axis=1).sort_values().index,
                        columns=models)
    pv = clf.pivot(index="ticker", columns="model",
                   values="p_value").reindex_like(grid)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "gap", [DIVERGING_NEG, NEUTRAL, DIVERGING_POS])
    lim = float(np.nanmax(np.abs(grid.to_numpy())))

    fig, ax = plt.subplots(figsize=(1.55 * len(models) + 3.4,
                                    0.46 * len(grid) + 3.2))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(grid.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim,
                   aspect="auto")

    for i in range(len(grid)):
        for j in range(len(models)):
            v = grid.iat[i, j]
            if np.isnan(v):
                continue
            # white text on dark cells, for contrast
            strong = abs(v) > lim * 0.55
            star = " *" if pv.iat[i, j] < 0.05 else ""
            ax.text(j, i, f"{v:+.3f}".replace(".", ",") + star,
                    ha="center", va="center", fontsize=8.5,
                    color="white" if strong else INK,
                    fontweight="600" if star else "normal")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=9.5,
                       color=INK_SECONDARY)
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels(grid.index, fontsize=9.5, color=INK_SECONDARY)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    # thin gaps between cells instead of cell borders
    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8.5, colors=INK_MUTED, length=0)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:+.2f}".replace(".", ",")))
    cbar.set_label("accuracy − suelo naïve", fontsize=9, color=INK_SECONDARY)

    _titles(
        fig,
        "Diferencia de accuracy frente al naïve por activo y modelo",
        "Diferencia entre el accuracy de cada modelo y el suelo naïve de su "
        "activo. Azul lo supera, rojo se queda corto.\n"
        "Los asteriscos marcan p-valores binomiales brutos inferiores a 0,05; "
        "no incorporan corrección por multiplicidad.",
    )
    fig.subplots_adjust(top=0.88)
    _save(fig, "summary_gap_heatmap")


def plot_feature_importance_summary() -> None:
    """Feature importances of the tree models across all assets.

    For each variable: the value in every asset and their mean.
    """
    imp = _load("feature_importances")
    models = [m for m in _models_present(imp)]

    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 6.4),
                             sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)

    order = (imp.groupby("feature").importance.mean()
             .sort_values().index.tolist())
    for ax, m in zip(axes, models):
        sub = imp[imp.model == m]
        means = sub.groupby("feature").importance.mean().reindex(order)
        for i, feat in enumerate(order):
            vals = sub.loc[sub.feature == feat, "importance"]
            ax.plot([vals.min(), vals.max()], [i, i], color=DE_EMPHASIS,
                    linewidth=1.3, solid_capstyle="round", zorder=1)
            ax.plot(vals, [i] * len(vals), "o", markersize=5,
                    color=MODEL_COLOR[m], alpha=0.55,
                    markeredgecolor=SURFACE, markeredgewidth=1, zorder=2)
            ax.plot(means[feat], i, "o", markersize=9, color=MODEL_COLOR[m],
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9.5, color=INK_SECONDARY)
        _style_axes(ax, xgrid=True, ygrid=False)
        ax.set_title(MODEL_LABEL[m], fontsize=11, color=INK,
                     fontweight="600", loc="left", pad=10)
        ax.set_xlabel("importancia", fontsize=9.5, color=INK_SECONDARY,
                      labelpad=8)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:.3f}".replace(".", ",")))
        # reference line: the same importance (1/n) for every variable
        ax.axvline(1 / len(order), color=INK, linewidth=1.2, zorder=4)

    def es(x: float, digits: int = 3) -> str:
        """Number with a Spanish decimal comma."""
        return f"{x:.{digits}f}".replace(".", ",")

    means = {m: imp[imp.model == m].groupby("feature").importance.mean()
             for m in models}
    spans = {m: s.max() - s.min() for m, s in means.items()}
    narrowest = min(spans, key=spans.get)
    leaders = " · ".join(f"{MODEL_LABEL[m]} «{means[m].idxmax()}»"
                         for m in models)
    _titles(
        fig,
        "Importancias internas de los modelos de árboles",
        f"La línea vertical reparte el peso por igual entre {len(order)} variables; "
        f"no representa una hipótesis nula.\nEn "
        f"{MODEL_LABEL[narrowest]}, entre la primera y la última variable solo "
        f"hay {es(spans[narrowest])}. Variables líderes por familia: {leaders}.\n"
        "Las correlaciones y el promedio entre activos limitan la interpretación predictiva.",
    )
    fig.subplots_adjust(top=0.80, wspace=0.28)
    _save(fig, "summary_feature_importance")


def plot_time_in_market() -> None:
    """Excess return over Buy & Hold against the share of positive signals.

    Descriptive only: it does not attribute a cause.
    """
    pr = _load("predictions")
    fin = _load("financial_metrics")
    bh = fin[fin.model == "buy_and_hold"].set_index("ticker")
    st = fin[fin.model.isin(_models_present(fin))].copy()

    share = pr.groupby(["ticker", "model"]).prediction.mean()
    st["time_in"] = st.set_index(["ticker", "model"]).index.map(share)
    st["excess"] = st.cumulative_return - st.ticker.map(bh.cumulative_return)
    st["bh_return"] = st.ticker.map(bh.cumulative_return)

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    fig.patch.set_facecolor(SURFACE)

    ax.axhline(0, color=INK, linewidth=1.2, zorder=2)
    losing = st[st.bh_return < 0]
    winning = st[st.bh_return >= 0]
    for sub, color, label, size in (
        (winning, SERIES[0], "Activo alcista en 2025", 8),
        (losing, SERIES[1], "Activo bajista en 2025", 9),
    ):
        ax.plot(sub.time_in, sub.excess, "o", markersize=size, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                label=label)
        # horizontal line at each group's mean
        ax.axhline(sub.excess.mean(), color=color, linewidth=1.2,
                   alpha=0.55, zorder=1)
        ax.annotate(f"media {sub.excess.mean():+.0%}".replace("%", " %"),
                    xy=(0.004, sub.excess.mean()),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points",
                    fontsize=8.5, color=color, fontweight="600")

    _style_axes(ax)
    ax.set_xlabel("Proporción de señales positivas", fontsize=10,
                  color=INK_SECONDARY, labelpad=10)
    ax.set_ylabel("Rentabilidad menos la de Buy & Hold", fontsize=10,
                  color=INK_SECONDARY)
    for axis, fmt in ((ax.xaxis, "{:.0%}"), (ax.yaxis, "{:+.0%}")):
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _, f=fmt: f.format(v).replace("%", " %")))
    ax.legend(loc="upper left", frameon=False, fontsize=9.5,
              labelcolor=INK_SECONDARY)

    def es2(x: float) -> str:
        return f"{x:+.2f}".replace(".", ",")

    up_ok, dn_ok = int((winning.excess > 0).sum()), int((losing.excess > 0).sum())
    _titles(
        fig,
        "Frecuencia de señales y exceso de rentabilidad",
        f"Cada punto es un par activo-modelo. Sobre los activos que cayeron, "
        f"los {dn_ok} de {len(losing)} pares baten al Buy & Hold\n"
        f"(exceso medio {losing.excess.mean():+.0%}); sobre los alcistas solo "
        f"lo logran {up_ok} de {len(winning)}, con {winning.excess.mean():+.0%}"
        ".\nCorrelación descriptiva entre señales positivas y exceso "
        f"(r = {es2(losing.time_in.corr(losing.excess))} y "
        f"{es2(winning.time_in.corr(winning.excess))}).\nNo mide exposición ejecutada "
        "ni permite atribuir un mecanismo causal.".replace(
            "%", " %"),
    )
    fig.subplots_adjust(top=0.86)
    _save(fig, "summary_time_in_market")


def plot_precision_recall() -> None:
    """Precision against recall per model, over constant-F1 curves.

    Always predicting "up" gives recall 1 and still a high F1 (0.689 on AAPL),
    which is why F1 was replaced as the tuning criterion.
    """
    clf = _load("classification_metrics")
    models = _models_present(clf)

    fig, axes = plt.subplots(1, len(models), figsize=(2.9 * len(models), 5.6),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)

    rec = np.linspace(0.02, 1.0, 300)
    for k, (ax, m) in enumerate(zip(axes, models)):
        for f in (0.5, 0.6, 0.7, 0.8):
            prec = f * rec / (2 * rec - f)
            ok = (prec > 0) & (prec <= 1.02)
            ax.plot(rec[ok], prec[ok], color=GRID, linewidth=1.0, zorder=1)
            # contours labelled once, on the last panel
            if ok.any() and k == len(models) - 1:
                ax.annotate(f"F1 {f:.1f}".replace(".", ","),
                            xy=(1.0, prec[ok][-1]), xytext=(4, 0),
                            textcoords="offset points", ha="left",
                            va="center", fontsize=8, color=INK_MUTED, zorder=1)

        naive = clf[clf.model == "naive"]
        ax.plot(naive.recall, naive.precision, "o", markersize=6,
                color=DE_EMPHASIS, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=2)
        sub = clf[clf.model == m]
        ax.plot(sub.recall, sub.precision, "o", markersize=7,
                color=MODEL_COLOR[m], markeredgecolor=SURFACE,
                markeredgewidth=1.4, zorder=3)

        _style_axes(ax, xgrid=False, ygrid=False)
        ax.set_title(MODEL_LABEL[m], fontsize=10.5, color=INK,
                     fontweight="600", loc="left", pad=8)
        ax.set_xlim(0, 1.06)
        ax.set_ylim(0.42, 0.74)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
        ax.set_xlabel("recall", fontsize=9.5, color=INK_SECONDARY, labelpad=8)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v:.2f}".replace(".", ",")))
    axes[0].set_ylabel("precision", fontsize=9.5, color=INK_SECONDARY)

    handles = [
        plt.Line2D([], [], color=DE_EMPHASIS, marker="o", linestyle="none",
                   markersize=6, label="Referencia ingenua (recall = 1)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.845),
               frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)

    _titles(
        fig,
        "El F1 premia justo al modelo que no mira los datos",
        "Cada punto es un activo; las curvas grises son valores constantes de "
        "F1. La referencia ingenua responde «sube» a todo, así que aterriza "
        "en recall = 1 y aun así\nse sitúa sobre curvas altas. Los modelos que "
        "se le acercan por la derecha están colapsando en ella.\nPor eso el "
        "criterio interno de selección pasó a balanced_accuracy.",
    )
    fig.subplots_adjust(top=0.70, wspace=0.12)
    _save(fig, "summary_precision_recall")


def plot_drawdown() -> None:
    """Drawdown curves: how far each strategy and its Buy & Hold fall below
    their running peak, session by session."""
    eq = _load("equity_curves")
    eq["date"] = pd.to_datetime(eq["date"])
    models = _models_present(eq)
    tickers = [t for t in config.TICKERS if t in set(eq.ticker)]

    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel()

    for ax, ticker in zip(axes, tickers):
        sub = eq[eq.ticker == ticker]
        b = sub[sub.model == "buy_and_hold"].sort_values("date")
        dd_b = b.equity.to_numpy() / np.maximum.accumulate(b.equity.to_numpy()) - 1
        ax.fill_between(b.date, dd_b, 0, color=INK_MUTED, alpha=0.35, zorder=2)
        ax.plot(b.date, dd_b, color=INK_MUTED, linewidth=1.6, zorder=3)

        worst_strategy = 0.0
        for m in models:
            s = sub[sub.model == m].sort_values("date")
            e = s.equity.to_numpy()
            dd = e / np.maximum.accumulate(e) - 1
            worst_strategy = min(worst_strategy, dd.min())
            ax.plot(s.date, dd, color=SERIES[0], linewidth=1.0, alpha=0.6,
                    zorder=4)

        _style_axes(ax)
        ax.set_title(ticker, fontsize=10.5, color=INK, fontweight="600",
                     loc="left", pad=6)
        ax.set_ylim(min(dd_b.min(), worst_strategy) * 1.14, 0.012)
        ax.annotate(
            f"B&H {_pct(dd_b.min())}",
            xy=(0.98, 0.05), xycoords="axes fraction", ha="right",
            fontsize=8.5, color=INK_MUTED)
        ax.set_xticks([])
        # round tick steps, so no decimals are needed
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(
            nbins=5, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: _pct(v)))
        ax.tick_params(labelsize=8)

    for ax in axes[len(tickers):]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([], [], color=SERIES[0], linewidth=1.6, alpha=0.7,
                   label=f"Estrategias ML ({len(models)} modelos)"),
        plt.Line2D([], [], color=INK_MUTED, linewidth=2.0,
                   label="Buy & Hold (referencia)"),
    ]
    fig.legend(handles=handles, loc="lower right", frameon=False, fontsize=10,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.98, 0.06), ncol=2)

    _titles(
        fig,
        "Bajo el agua: la caída desde el máximo previo, sesión a sesión",
        "El área gris es lo que perdió el Buy & Hold respecto a su propio "
        "máximo; las líneas azules, lo mismo para cada estrategia.\n"
        "La figura describe drawdown; no identifica por sí sola el mecanismo "
        "del riesgo o del Sharpe. Periodo de prueba: 2025.",
    )
    fig.subplots_adjust(top=0.87, hspace=0.34, wspace=0.22, bottom=0.07)
    _save(fig, "summary_drawdown")


def plot_base_rates() -> None:
    """Share of "up" sessions per asset, in training and in test.

    This is the naive floor H1 must beat: learnt on training data and scored
    on test data, so it depends on the asset, not on the model.
    """
    clf = _load("classification_metrics")
    if "base_rate_train" not in clf.columns:
        raise FileNotFoundError("classification_metrics.csv lacks base_rate_train")
    g = clf.groupby("ticker").agg(train=("base_rate_train", "first"),
                                  test=("base_rate", "first"))
    g = g.sort_values("test")

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(g) + 3.4))
    fig.patch.set_facecolor(SURFACE)

    ax.axvline(0.5, color=INK, linewidth=1.3, zorder=4)
    for i, (ticker, row) in enumerate(g.iterrows()):
        ax.plot([row.train, row.test], [i, i], color=DE_EMPHASIS,
                linewidth=1.6, solid_capstyle="round", zorder=1)
        ax.plot(row.train, i, "o", markersize=8, color=SERIES[0],
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
        ax.plot(row.test, i, "o", markersize=10, color=SERIES[1],
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3)

    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index, fontsize=10, color=INK_SECONDARY)
    ax.set_ylim(-0.8, len(g) - 0.2)
    _style_axes(ax, xgrid=True, ygrid=False)
    ax.set_xlabel("Proporción de sesiones al alza", fontsize=10,
                  color=INK_SECONDARY, labelpad=10)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:.0%}".replace("%", " %")))
    ax.annotate("azar (50 %)", xy=(0.5, len(g) - 0.55), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=INK,
                fontweight="600")

    handles = [
        plt.Line2D([], [], color=SERIES[0], marker="o", linestyle="none",
                   markersize=8, label="Entrenamiento (2000–2024)"),
        plt.Line2D([], [], color=SERIES[1], marker="o", linestyle="none",
                   markersize=10, label="Prueba (2025)"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.005),
              frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY, ncol=2)

    below = int((g.test < 0.5).sum())
    _titles(
        fig,
        "El listón de H1 no es el 50 %: lo pone cada activo",
        f"Proporción de sesiones al alza en entrenamiento y en prueba. El "
        f"recorrido va del {_pct(g.test.min(), 1)} al "
        f"{_pct(g.test.max(), 1)} en 2025, así que un mismo accuracy puede "
        f"ser un éxito o un fracaso\nsegún el activo. En {below} de {len(g)} "
        f"el listón cae por debajo del azar: ahí el modelo trivial acierta "
        f"menos que una moneda.",
    )
    fig.subplots_adjust(top=0.84)
    _save(fig, "summary_base_rates")


def plot_return_distribution() -> None:
    """Daily return distribution of the strategies and of Buy & Hold.

    A long/cash strategy earns exactly zero on the days it holds cash; the
    right panel shows the share of exactly-zero returns.
    """
    eq = _load("equity_curves")
    eq = eq.sort_values(["ticker", "model", "date"])
    eq["ret"] = eq.groupby(["ticker", "model"]).equity.pct_change()
    models = _models_present(eq)

    strat = eq[eq.model.isin(models)].ret.dropna()
    bh = eq[eq.model == "buy_and_hold"].ret.dropna()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                  gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor(SURFACE)

    bins = np.linspace(-0.05, 0.05, 81)
    for vals, color, label in ((bh, INK_MUTED, "Buy & Hold"),
                               (strat, SERIES[0], "Estrategias ML")):
        h, edges = np.histogram(vals, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, h / h.sum(), color=color, linewidth=1.8, zorder=3,
                label=label)
    _style_axes(ax)
    ax.set_xlabel("Rentabilidad diaria", fontsize=10, color=INK_SECONDARY,
                  labelpad=10)
    ax.set_ylabel("proporción de sesiones", fontsize=9.5, color=INK_SECONDARY)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:+.0%}".replace("%", " %")))
    ax.legend(loc="upper left", frameon=False, fontsize=9.5,
              labelcolor=INK_SECONDARY)

    flat_s = float((strat.abs() < 1e-12).mean())
    flat_b = float((bh.abs() < 1e-12).mean())
    ax2.bar([0, 1], [flat_b, flat_s], width=0.5,
            color=[INK_MUTED, SERIES[0]], zorder=3)
    for i, v in enumerate((flat_b, flat_s)):
        ax2.annotate(f"{v:.0%}".replace("%", " %"), xy=(i, v),
                     xytext=(0, 6), textcoords="offset points", ha="center",
                     fontsize=10, color=INK, fontweight="600")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Buy & Hold", "Estrategias ML"], fontsize=9.5,
                        color=INK_SECONDARY)
    ax2.set_ylim(0, max(flat_s, flat_b) * 1.3)
    _style_axes(ax2)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:.0%}".replace("%", " %")))
    ax2.set_title("Sesiones con rentabilidad exactamente cero",
                  fontsize=10, color=INK, fontweight="600", loc="left", pad=10)

    _titles(
        fig,
        "Distribución de retornos diarios y frecuencia de retornos nulos",
        f"Distribución de rentabilidades diarias de las {len(models)} "
        f"estrategias frente al Buy & Hold.\nEl {flat_s:.0%} de los retornos "
        f"de las estrategias son exactamente nulos. Esta frecuencia no equivale "
        f"al tiempo en liquidez ni demuestra una mejora mecánica del Sharpe.".replace(
            "%", " %"),
    )
    fig.subplots_adjust(top=0.82, wspace=0.24)
    _save(fig, "summary_return_distribution")


# ---------------------------------------------------------------------------
# Methodology figures (Ch. 3 / Ch. 4): they illustrate design decisions with
# the project's own data.
# ---------------------------------------------------------------------------


def plot_volatility_by_asset() -> None:
    """Annualised volatility per asset, coloured by asset class."""
    from . import data as data_mod
    from . import metrics as metrics_mod

    rows = []
    for ticker, df in data_mod.load_raw().items():
        close = df["Close"].dropna()
        ret = np.log(close / close.shift(1)).dropna()
        rows.append(
            {
                "ticker": ticker,
                "type": config.asset_type(ticker),
                "vol": metrics_mod.annualized_volatility(ret, ticker),
            }
        )
    d = pd.DataFrame(rows).sort_values("vol")

    label = {"index": "Índice", "etf": "ETF", "stock": "Acción",
             "crypto": "Criptomoneda"}
    color = {"index": SERIES[0], "etf": SERIES[2], "stock": SERIES[3],
             "crypto": SERIES[1]}

    fig, ax = plt.subplots(figsize=(9.5, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.barh(d.ticker, d.vol, height=0.62,
            color=[color[t] for t in d.type], zorder=3)
    for y, v in enumerate(d.vol):
        ax.annotate(_pct(v, 1), xy=(v, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK_SECONDARY)

    _style_axes(ax, xgrid=True, ygrid=False)
    ax.set_xlabel("Volatilidad anualizada de los retornos logarítmicos",
                  fontsize=10, color=INK_SECONDARY, labelpad=10)
    ax.set_xlim(0, d.vol.max() * 1.16)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: _pct(v)))
    ax.tick_params(axis="y", labelsize=9.5, colors=INK_SECONDARY)

    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                          color=color[k], label=label[k])
               for k in ("index", "etf", "stock", "crypto")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5,
              labelcolor=INK_SECONDARY)

    crypto = d[d.type == "crypto"].vol.mean()
    rest = d[d.type != "crypto"].vol.mean()
    _titles(
        fig,
        "Las criptomonedas operan en otro régimen de riesgo",
        "Volatilidad anualizada sobre el histórico completo de cada activo. "
        f"La media de las dos criptomonedas ({_pct(crypto)}) es "
        f"{crypto / rest:.1f}".replace(".", ",") + " veces la del resto de la muestra "
        
        f"({_pct(rest)}).\nEs la diferencia que justifica anualizar con la "
        "raíz de 365 en lugar de 252 por su distinto calendario; "
        "el nivel de volatilidad no determina ese factor.",
    )
    fig.subplots_adjust(top=0.86)
    _save(fig, "metodologia_volatilidad")


def plot_timeseriessplit() -> None:
    """Expanding-window CV scheme used to tune the hyper-parameters.

    Drawn from the real splitter, so the proportions match the experiment.
    """
    from .validation import make_cv

    n = 240  # illustrative length; only the pattern matters
    cv = make_cv()

    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor(SURFACE)
    for k, (train, test) in enumerate(cv.split(np.arange(n))):
        ax.barh(k, len(train), height=0.55, color=SERIES[0], zorder=3)
        ax.barh(k, len(test), left=test[0], height=0.55,
                color=SERIES[1], zorder=3)
        ax.barh(k, n - test[-1] - 1, left=test[-1] + 1,
                height=0.55, color=GRID, zorder=2)

    ax.set_yticks(range(config.CV_N_SPLITS))
    ax.set_yticklabels([f"Pliegue {k + 1}" for k in range(config.CV_N_SPLITS)],
                       fontsize=10, color=INK_SECONDARY)
    ax.invert_yaxis()
    _style_axes(ax, xgrid=False, ygrid=False)
    ax.set_xlabel("Observaciones ordenadas cronológicamente", fontsize=10,
                  color=INK_SECONDARY, labelpad=10)
    ax.set_xticks([])
    ax.set_xlim(0, n)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=SERIES[0], label="Entrenamiento"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=SERIES[1], label="Validación"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=GRID, label="Aún no utilizado"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02),
              frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY, ncol=3)

    _titles(
        fig,
        "La validación nunca entrena con datos posteriores a los que evalúa",
        f"Particiones con {config.CV_N_SPLITS} pliegues. La ventana de "
        "entrenamiento se amplía en cada pliegue y el tramo de validación es "
        f"posterior, con {config.TARGET_HORIZON_N} fila(s) de purga entre bloques.\n"
        "El año reservado como prueba queda fuera de "
        "este esquema por completo.",
    )
    fig.subplots_adjust(top=0.74)
    _save(fig, "metodologia_timeseriessplit")


def _calendar_facts() -> dict:
    """Sessions, first date and December 2025 dates of every raw series."""
    from . import data as data_mod

    facts = {}
    for ticker, df in data_mod.load_raw().items():
        idx = df.index
        facts[ticker] = {
            "sessions": len(idx),
            "start": idx.min(),
            "dec": idx[(idx.year == 2025) & (idx.month == 12)],
        }
    return facts


def _distinct_calendars(raw: dict) -> list[str]:
    """One ticker per distinct session count, longest history first.

    The US assets share a calendar, so one bar each would repeat it.
    """
    pool = [t for t in raw if raw[t]["start"].year <= 2000]
    seen, common = set(), []
    for ticker in sorted(pool, key=lambda t: -raw[t]["sessions"]):
        n = raw[ticker]["sessions"]
        if n not in seen:
            seen.add(n)
            common.append(ticker)
    return common


def _plot_session_gap(ax, raw: dict, common: list[str]) -> None:
    """Left panel: sessions of each calendar minus the US one.

    Differences, not absolute counts: at ~6500 sessions a 74-session gap
    would be invisible.
    """
    base = raw["^GSPC"]["sessions"]
    diff = [raw[t]["sessions"] - base for t in common]
    ax.barh(range(len(common)), diff, height=0.6,
            color=[SERIES[0] if v else DE_EMPHASIS for v in diff], zorder=3)
    for y, (v, t) in enumerate(zip(diff, common)):
        ax.annotate(f"{v:+d}" if v else "referencia",
                    xy=(max(v, 0), y), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK_SECONDARY)
    ax.set_yticks(range(len(common)))
    ax.set_yticklabels([f"{t}  ({raw[t]['sessions']} sesiones)"
                        for t in common], fontsize=9.5, color=INK_SECONDARY)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, linewidth=1.2, zorder=4)
    _style_axes(ax, xgrid=True, ygrid=False)
    ax.set_xlim(-4, max(diff) * 1.5)
    ax.set_xlabel("Sesiones de más respecto al calendario estadounidense",
                  fontsize=10, color=INK_SECONDARY, labelpad=10)
    ax.set_title("Mismo periodo, distinto número de sesiones", fontsize=11,
                 color=INK, fontweight="600", loc="left", pad=10)


def _plot_december_sessions(ax2, raw: dict) -> None:
    """Right panel: December 2025, where the divergence becomes visible."""
    detail = [t for t in ("^IBEX", "^GDAXI", "^GSPC") if t in raw]
    for k, ticker in enumerate(detail):
        days = raw[ticker]["dec"]
        ax2.plot(days, [k] * len(days), "|", color=SERIES[0], markersize=14,
                 markeredgewidth=2, zorder=3)
    ax2.set_yticks(range(len(detail)))
    ax2.set_yticklabels(detail, fontsize=9.5, color=INK_SECONDARY)
    ax2.set_ylim(-0.7, len(detail) - 0.3)
    ax2.invert_yaxis()
    _style_axes(ax2, xgrid=True, ygrid=False)
    ax2.set_xlabel("Sesiones de diciembre de 2025", fontsize=10,
                   color=INK_SECONDARY, labelpad=10)
    ax2.set_title("Fráncfort no abre el 31 de diciembre", fontsize=11,
                  color=INK, fontweight="600", loc="left", pad=10)
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d"))
    ax2.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=5))
    ax2.tick_params(axis="x", labelsize=9)


def plot_trading_calendars() -> None:
    """Sessions traded by each market over the same period (project data)."""
    raw = _calendar_facts()
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor(SURFACE)
    _plot_session_gap(ax, raw, _distinct_calendars(raw))
    _plot_december_sessions(ax2, raw)

    gap = raw["^IBEX"]["sessions"] - raw["^GSPC"]["sessions"]
    _titles(
        fig,
        "Los calendarios bursátiles no son homogéneos",
        f"Sobre el mismo periodo, el IBEX 35 negocia {gap} sesiones más que "
        "el S&P 500: son días en que un mercado abre y el otro no.\n"
        "Alinear las series en un índice común convertiría esos días en "
        "valores ausentes, de ahí que cada activo se procese con su propio "
        "calendario.",
    )
    fig.subplots_adjust(top=0.76, wspace=0.2)
    _save(fig, "metodologia_calendarios")


# ---------------------------------------------------------------------------
# Walk-forward figures (thesis §5.3), built from the persisted
# `walkforward_*` tables.
# ---------------------------------------------------------------------------

WF_TRAIN, WF_PURGE, WF_TEST = SERIES[0], DIVERGING_NEG, SERIES[1]


def _wf_diverging_cmap():
    """Red-to-blue colour map through the palette's neutral midpoint."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "wf", [DIVERGING_NEG, NEUTRAL, DIVERGING_POS]
    )


def _wf_fold_table(ticker: str) -> tuple[str, pd.DataFrame]:
    """Fold design of one asset, from the persisted table or rebuilt.

    Returns the ticker used, which changes if the requested one has no folds.
    """
    try:
        table = _load("walkforward_folds")
        part = table[table.ticker == ticker].copy()
        if part.empty:  # asset not in this run: use the first one that is
            ticker = table.ticker.iloc[0]
            part = table[table.ticker == ticker].copy()
    except FileNotFoundError:
        from . import data as data_mod
        from . import validation as val_mod

        idx = data_mod.load_raw([ticker])[ticker].index
        part = val_mod.folds_to_frame(
            ticker, val_mod.make_walkforward_folds(idx))
    for col in ("train_start", "train_end", "test_start", "test_end"):
        part[col] = pd.to_datetime(part[col])
    return ticker, part.sort_values("test_year").reset_index(drop=True)


def _plot_wf_fold_bars(ax, part: pd.DataFrame, ticker: str) -> None:
    """Left panel: one bar per fold, training window plus test block."""
    import matplotlib.dates as mdates

    for k, row in part.iterrows():
        tr0 = mdates.date2num(row.train_start)
        tr1 = mdates.date2num(row.train_end)
        te0 = mdates.date2num(row.test_start)
        te1 = mdates.date2num(row.test_end)
        ax.barh(k, tr1 - tr0, left=tr0, height=0.58, color=WF_TRAIN, zorder=3)
        ax.barh(k, te1 - te0, left=te0, height=0.58, color=WF_TEST, zorder=3)
        ax.annotate(f"{row.n_train:,}".replace(",", "."), xy=(tr1, k),
                    xytext=(-7, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=8, color=SURFACE, zorder=4)
    ax.set_yticks(range(len(part)))
    ax.set_yticklabels([f"Pliegue {int(r.fold_id)} · prueba {int(r.test_year)}"
                        for _, r in part.iterrows()],
                       fontsize=9.5, color=INK_SECONDARY)
    ax.invert_yaxis()
    _style_axes(ax, xgrid=True, ygrid=False)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(f"Un pliegue por año natural ({ticker})", fontsize=11,
                 color=INK, fontweight="600", loc="left", pad=10)


def _plot_wf_boundary(ax2, purge: int) -> None:
    """Right panel: the train/test boundary, session by session.

    At N=1 the purge is one session, invisible at the scale of the left panel.
    """
    cells = list(range(-6, 5))
    for x in cells:
        color = WF_TRAIN if x < -purge else (WF_PURGE if x < 0 else WF_TEST)
        ax2.barh(0, 0.86, left=x, height=0.9, color=color, zorder=3)
    ax2.set_xticks(cells)
    ax2.set_xticklabels([f"t{x:+d}" if x else "t" for x in cells], fontsize=8.5)
    ax2.set_yticks([])
    ax2.set_ylim(-1.5, 1.2)
    _style_axes(ax2, xgrid=False, ygrid=False)
    ax2.spines["bottom"].set_visible(False)
    ax2.annotate(
        f"purga\n({purge} fila{'s' if purge > 1 else ''})",
        xy=(-purge + 0.43, -0.5), xytext=(-purge + 0.43, -1.2),
        ha="center", va="top", fontsize=9, color=DIVERGING_NEG,
        arrowprops=dict(arrowstyle="-", color=DIVERGING_NEG, lw=1.0))
    ax2.set_title("La frontera, sesión a sesión", fontsize=11, color=INK,
                  fontweight="600", loc="left", pad=10)


def plot_wf_folds_diagram(ticker: str = "^GSPC") -> None:
    """The fold scheme: growing training window, purged boundary, test block.

    Drawn from the folds actually built, so the proportions are real.
    """
    ticker, part = _wf_fold_table(ticker)
    purge = int(part["purge_rows"].iloc[0])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [2.1, 1]})
    fig.patch.set_facecolor(SURFACE)
    _plot_wf_fold_bars(ax, part, ticker)
    _plot_wf_boundary(ax2, purge)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=WF_TRAIN, label="Entrenamiento"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=WF_PURGE, label="Purgado (etiqueta contaminada)"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=WF_TEST, label="Prueba"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.12),
              frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY, ncol=3)

    purged_phrase = ("La última fila" if purge == 1
                     else f"Las {purge} últimas filas")
    _titles(
        fig,
        "Cada año se evalúa con un modelo que no ha visto ese año",
        "La ventana de entrenamiento arranca siempre al principio del "
        "histórico y crece; el bloque de prueba es el año natural siguiente."
        f"\n{purged_phrase} del entrenamiento se descarta{'n' if purge > 1 else ''} "
        "porque su etiqueta se construye con precios que ya pertenecen "
        "al bloque de prueba.",
    )
    fig.subplots_adjust(top=0.74, wspace=0.16)
    _save(fig, "wf_folds_diagram")


def plot_wf_heatmap_balanced_accuracy() -> None:
    """Balanced accuracy per asset and test year, one panel per model.

    Its 0.50 reference is the same every year, so one colour scale fits the
    whole grid. The naive model scores exactly 0.50 and is not drawn.
    """
    clf = _load("walkforward_classification")
    models = _models_present(clf)
    if not models:
        return
    tickers = sorted(clf.ticker.unique())
    years = sorted(clf.test_year.unique())
    cmap = _wf_diverging_cmap()
    vmax = max(0.02, float((clf.balanced_accuracy - 0.5).abs().quantile(0.98)))

    ncols = min(3, len(models))
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    im = None
    for ax, model in zip(axes.ravel(), models):
        part = clf[clf.model == model].pivot_table(
            index="ticker", columns="test_year", values="balanced_accuracy")
        part = part.reindex(index=tickers, columns=years)
        ax.set_facecolor(GRID)  # cells with no fold stay grey
        im = ax.imshow(part.to_numpy(), cmap=cmap, vmin=0.5 - vmax,
                       vmax=0.5 + vmax, aspect="auto")
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, fontsize=8, rotation=90, color=INK_MUTED)
        ax.set_yticks(range(len(tickers)))
        ax.set_yticklabels(tickers, fontsize=8, color=INK_MUTED)
        ax.set_title(MODEL_LABEL[model], fontsize=10.5, color=INK,
                     fontweight="600", loc="left", pad=8)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
    # put the colour bar in the empty grid slot, if there is one
    spare = axes.ravel()[len(models):]
    for ax in spare:
        ax.set_visible(False)
    if len(spare):
        box = spare[0].get_position()
        cax = fig.add_axes([box.x0, box.y0 + box.height * 0.45,
                            box.width * 0.62, box.height * 0.055])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    else:
        cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label("Balanced accuracy (0,500 = suelo naïve)", fontsize=9.5,
                   color=INK_SECONDARY, labelpad=8)
    cbar.ax.tick_params(labelsize=8.5, colors=INK_MUTED, length=0)
    cbar.outline.set_visible(False)

    above = float((clf[clf.model != "naive"].balanced_accuracy > 0.5).mean())
    _titles(
        fig,
        "Balanced accuracy por activo, modelo y año",
        f"Balanced accuracy por activo y año de prueba. El {_pct(above)} de "
        "las celdas queda por encima de la referencia de 0,500. "
        "La figura es descriptiva.\nLas celdas grises son años sin pliegue: el "
        "activo no acumulaba histórico suficiente.",
    )
    fig.subplots_adjust(top=0.86 if nrows > 1 else 0.78)
    _save(fig, "wf_heatmap_balanced_accuracy")


def plot_wf_stability(model: str) -> None:
    """Accuracy minus the naive floor, fold by fold, for one model.

    Zero is each fold's own floor, which changes with asset and year; plotting
    the gap puts every fold on the same scale.
    """
    clf = _load("walkforward_classification")
    part = clf[clf.model == model]
    if part.empty:
        return
    gap = part.accuracy - part.base_rate
    color = MODEL_COLOR.get(model, DE_EMPHASIS)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(config.RANDOM_STATE)
    jitter = rng.uniform(-0.16, 0.16, len(part))
    ax.scatter(part.test_year + jitter, gap, s=34, color=color, alpha=0.75,
               linewidths=0, zorder=3)
    per_year = gap.groupby(part.test_year).mean()
    ax.plot(per_year.index, per_year.to_numpy(), color=INK, linewidth=1.6,
            zorder=4, label="Media anual")
    ax.axhline(0, color=INK, linewidth=1.2, zorder=2)
    _style_axes(ax, xgrid=False, ygrid=True)
    ax.set_xticks(sorted(part.test_year.unique()))
    ax.set_ylabel("Accuracy − suelo naïve", fontsize=10, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
              loc="upper right")

    share = float((gap > 0).mean())
    _titles(
        fig,
        f"{MODEL_LABEL.get(model, model)}: diferencias respecto al naïve por fold",
        "Un punto por activo y año de prueba. La línea cero es el suelo de "
        "la referencia ingenua en ese mismo pliegue. "
        f"{_pct(share)} de los {len(part)} pliegues quedan por encima.\n"
        "Los folds comparten histórico de entrenamiento y no son observaciones independientes.",
    )
    fig.subplots_adjust(top=0.80)
    _save(fig, f"wf_stability_{model}")


def plot_wf_equity(ticker: str, model: str) -> None:
    """Compounded walk-forward equity curve against Buy & Hold.

    Fold returns are chained instead of restarting the capital each January.
    """
    equity = _load("walkforward_equity")
    part = equity[equity.ticker == ticker]
    strat = part[part.model == model]
    bench = part[part.model == "buy_and_hold"]
    if strat.empty or bench.empty:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5))
    fig.patch.set_facecolor(SURFACE)
    for frame, label, color, ls in (
        (strat, f"Estrategia ML ({MODEL_LABEL.get(model, model)})",
         MODEL_COLOR.get(model, SERIES[0]), "-"),
        (bench, "Buy & Hold", DE_EMPHASIS, "--"),
    ):
        ax.plot(pd.to_datetime(frame.date), frame.equity, label=label,
                color=color, linewidth=1.6, linestyle=ls, zorder=3)
    _style_axes(ax, xgrid=False, ygrid=True)
    ax.set_ylabel("Capital (base 10.000)", fontsize=10, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
              loc="upper left")

    end_s = float(strat.equity.iloc[-1]) / config.INITIAL_CASH - 1
    end_b = float(bench.equity.iloc[-1]) / config.INITIAL_CASH - 1
    yrs = pd.to_datetime(strat.date).dt.year
    _titles(
        fig,
        f"{ticker}: la estrategia encadenada frente a no hacer nada",
        f"Capital compuesto a lo largo de los pliegues de {yrs.min()}-"
        f"{yrs.max()}, sin reiniciarlo en cada año. La estrategia acumula "
        f"{_pct(end_s, 1)} frente al {_pct(end_b, 1)}\nde la referencia "
        "pasiva sobre las mismas fechas, con reinicio de posiciones por fold "
        "y comisiones nulas.",
    )
    fig.subplots_adjust(top=0.80)
    _save(fig, f"wf_equity_{_safe(ticker)}_{model}")


def build_walkforward_figures() -> None:
    """Rebuild every walk-forward figure from the persisted tables."""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    for fn in (plot_wf_folds_diagram, plot_wf_heatmap_balanced_accuracy):
        try:
            fn()
            print(f"  {fn.__name__}")
        except FileNotFoundError as exc:
            print(f"  {fn.__name__}: skipped ({exc.filename})")
    try:
        clf = _load("walkforward_classification")
    except FileNotFoundError:
        return
    for model in _models_present(clf):
        plot_wf_stability(model)
        print(f"  plot_wf_stability({model})")
    try:
        equity = _load("walkforward_equity")
    except FileNotFoundError:
        return
    pairs = equity[equity.model != "buy_and_hold"][
        ["ticker", "model"]].drop_duplicates()
    for ticker, model in pairs.itertuples(index=False):
        plot_wf_equity(ticker, model)
    print(f"  plot_wf_equity x{len(pairs)}")


def build_methodology_figures() -> None:
    """Rebuild the Ch. 3 / Ch. 4 figures that illustrate design decisions."""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    for fn in (plot_volatility_by_asset, plot_timeseriessplit,
               plot_trading_calendars):
        fn()
        print(f"  {fn.__name__}")


def build_summary_figures() -> None:
    """Rebuild every cross-asset figure from the persisted tables."""
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    for fn in (plot_h1_gap, plot_gap_heatmap, plot_pvalue_distributions,
               plot_base_rates, plot_precision_recall, plot_h3_exposure,
               plot_risk_return, plot_model_bias, plot_equity_grid,
               plot_drawdown, plot_exposure_map, plot_time_in_market,
               plot_return_distribution, plot_feature_importance_summary):
        try:
            fn()
            print(f"  {fn.__name__}")
        except FileNotFoundError as exc:
            # its source table was not produced by this run
            print(f"  {fn.__name__}: skipped ({exc.filename})")

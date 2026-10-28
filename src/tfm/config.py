"""Every pipeline parameter (M1-M5) lives here; no module hardcodes its own.

Only exception: the hyper-parameter grids, which are defined in models.py.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repository root)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ---------------------------------------------------------------------------
# Asset universe: 14 assets, 4 classes (thesis: Table 3.1)
# ---------------------------------------------------------------------------
ASSETS: dict[str, list[str]] = {
    "index": ["^GSPC", "^IXIC", "^IBEX", "^GDAXI"],
    "etf": ["ACWI"],  # iShares ETF tracking MSCI ACWI; history from 2008
    "stock": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "ITX.MC"],
    "crypto": ["BTC-USD", "ETH-USD"],  # 24/7 calendar
}

TICKERS: list[str] = [t for group in ASSETS.values() for t in group]

# Initial 6-asset design, kept as a smaller documented alternative
TICKERS_CORE6: list[str] = ["^GSPC", "^IXIC",
                            "^IBEX", "^GDAXI", "AAPL", "ITX.MC"]

# Crypto trades every day: metrics.py annualises it with 365 days, not 252
CRYPTO: set[str] = set(ASSETS["crypto"])

# ---------------------------------------------------------------------------
# Dates and temporal split (thesis: §3.4)
# ---------------------------------------------------------------------------
START_DATE = "2000-01-01"
END_DATE = "2026-01-01"  # yfinance `end` is EXCLUSIVE: this keeps 31/12/2025
TEST_START = "2025-01-01"  # 2025 reserved as out-of-sample test set

# ---------------------------------------------------------------------------
# Feature engineering (thesis: §4.3.2)
# ---------------------------------------------------------------------------
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
VOLATILITY_WINDOW = 20
RSI_LENGTH = 14
MACD_PARAMS = {"fast": 12, "slow": 26, "signal": 9}

# ---------------------------------------------------------------------------
# Target: 1 if the price rises N sessions ahead (thesis: §4.3.3)
# N=1 matches the backtest, which re-reads the signal every session; a longer
# horizon would raise the naive floor. tau=0: with no commission, there is no
# cost to justify a threshold.
# ---------------------------------------------------------------------------
TARGET_HORIZON_N = 1
TARGET_THRESHOLD = 0.0  # tau; 0 = plain direction

SENSITIVITY_HORIZON_N = 5  # secondary sensitivity arm: declared, not run

# ---------------------------------------------------------------------------
# Modelling and validation (thesis: §4.4.1)
# ---------------------------------------------------------------------------
CV_N_SPLITS = 5
# Tuning criterion. Unlike binary F1, it does not reward always predicting the
# majority class: a constant classifier scores 0.50 (chance level, not the
# metric's minimum).
CV_SCORING = "balanced_accuracy"
RANDOM_STATE = 42
# Parallelise one level only: nesting both oversubscribes the CPU
N_JOBS = -1  # outer: GridSearchCV
N_JOBS_ESTIMATOR = 1  # inner: RandomForest / XGBoost

# ---------------------------------------------------------------------------
# Walk-forward validation (robustness analysis; thesis: §4.4.7, §5.3)
# A complement to the single split, not a replacement. Folds are calendar
# years, so each test block is comparable with the 2025 test set.
# ---------------------------------------------------------------------------
WF_MODE = "anchored"  # "anchored" (expanding) | "rolling" (fixed window)
WF_FIRST_TEST_YEAR = 2016
WF_LAST_TEST_YEAR = 2025  # matches the main experiment
WF_MIN_TRAIN_YEARS = 5  # a fold is skipped if the asset has less history
WF_ROLLING_TRAIN_YEARS = 10  # only used when WF_MODE == "rolling"
WF_RETUNE_EACH_FOLD = True  # re-run GridSearchCV inside each training window
# Training rows purged before each test block (None = TARGET_HORIZON_N):
# y[t] uses the price at t+N, so the last N labels would read the test block.
WF_PURGE_ROWS = None

# ---------------------------------------------------------------------------
# Backtesting (thesis: §4.6)
# ---------------------------------------------------------------------------
# Annual risk-free rate used by the Sharpe ratio. Idle cash earns nothing in
# the simulation; a positive rate is a separate sensitivity check.
RISK_FREE_RATE = 0.0

INITIAL_CASH = 10_000
COMMISSION = 0.0  # can be raised for sensitivity analysis

# ---------------------------------------------------------------------------
# LSTM extension (optional; thesis: §4.5)
# ---------------------------------------------------------------------------
LSTM_WINDOW_T = 30  # sliding input window, in sessions
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 64


def asset_type(ticker: str) -> str:
    """Return the category ('index', 'etf', 'stock', 'crypto') of a ticker."""
    for kind, tickers in ASSETS.items():
        if ticker in tickers:
            return kind
    raise ValueError(f"Unknown ticker: {ticker}")

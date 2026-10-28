"""M1 - Data download.

Downloads adjusted prices with yfinance (auto_adjust=True: dividends and
splits included) and saves them to data/raw/, so later runs can work offline.
Thesis: §4.2 (Listing 4.2).
"""

from __future__ import annotations

import pandas as pd

from . import config

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


def drop_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sessions whose OHLC prices are not strictly positive.

    Yahoo's retroactive dividend adjustment left 349 ITX.MC bars below zero
    (2001-2002). Those prices are meaningless and break log returns, ratios
    and RSI, so they are dropped, never repaired. Other assets are unaffected.
    """
    valid = (df[PRICE_COLUMNS] > 0).all(axis=1)
    return df.loc[valid]


def download_data(
    tickers: list[str] | None = None,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
) -> dict[str, pd.DataFrame]:
    """Download adjusted OHLCV history for each ticker.

    Returns {ticker: DataFrame}, each with only the sessions that asset
    traded. Calendars are never aligned across assets: that would create
    spurious NaN.
    """
    import yfinance as yf  # lazy import: offline runs do not need yfinance

    tickers = tickers or config.TICKERS
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
    )
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = raw[ticker] if len(tickers) > 1 else raw
        df = df[OHLCV_COLUMNS].dropna(how="all")
        # no close price = the asset did not trade that day
        df = df.dropna(subset=["Close"]).sort_index()
        data[ticker] = drop_invalid_prices(df)
    return data


def report_date_ranges(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Actual available date range per asset (thesis, Listing 4.3)."""
    rows = []
    for ticker, df in data.items():
        series = df["Close"].dropna()
        rows.append(
            {
                "ticker": ticker,
                "type": config.asset_type(ticker),
                "start": series.index.min().date(),
                "end": series.index.max().date(),
                "sessions": len(series),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def save_raw(data: dict[str, pd.DataFrame]) -> None:
    """Persist raw OHLCV series to data/raw/, one parquet file per asset."""
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, df in data.items():
        df.to_parquet(config.DATA_RAW_DIR / f"{_safe_name(ticker)}.parquet")


def load_raw(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load the raw series saved by `save_raw` (offline)."""
    tickers = tickers or config.TICKERS
    data = {}
    for ticker in tickers:
        path = config.DATA_RAW_DIR / f"{_safe_name(ticker)}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw data for {ticker}. Run download_data + save_raw first."
            )
        # also applied on load, so files saved before this check existed are
        # cleaned without downloading them again
        data[ticker] = drop_invalid_prices(pd.read_parquet(path))
    return data


def _safe_name(ticker: str) -> str:
    """Filesystem-safe name (^GSPC -> GSPC, ITX.MC -> ITX_MC)."""
    return ticker.replace("^", "").replace(".", "_").replace("-", "_")

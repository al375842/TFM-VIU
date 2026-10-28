"""M1 tests: price-domain guard on the downloaded series."""

import numpy as np
import pandas as pd

from tfm import data


def _bars(closes: list[float]) -> pd.DataFrame:
    """Minimal OHLCV frame with every price column equal to `closes`."""
    close = pd.Series(closes, index=pd.bdate_range("2020-01-01", periods=len(closes)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1e5,
        }
    )


def test_drop_invalid_prices_removes_non_positive_bars():
    """Regression: ITX.MC carried 349 negative bars from Yahoo's adjustment."""
    df = _bars([-0.12, -0.11, 0.68, 0.67, 0.70])
    out = data.drop_invalid_prices(df)
    assert len(out) == 3
    assert (out[data.PRICE_COLUMNS] > 0).all().all()
    # surviving rows keep their original dates and values
    assert out.index.equals(df.index[2:])


def test_drop_invalid_prices_is_noop_on_clean_series():
    df = _bars([10.0, 10.5, 11.0])
    pd.testing.assert_frame_equal(data.drop_invalid_prices(df), df)


def test_drop_invalid_prices_catches_partial_bars():
    """A bar is invalid if any of its OHLC prices is non-positive or missing."""
    df = _bars([10.0, 11.0, 12.0])
    df.iloc[1, df.columns.get_loc("Low")] = 0.0
    df.iloc[2, df.columns.get_loc("High")] = np.nan
    out = data.drop_invalid_prices(df)
    assert len(out) == 1

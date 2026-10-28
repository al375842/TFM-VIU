"""M3 tests: model registry and temporal validation split."""

import numpy as np
import pandas as pd

from tfm import validation
from tfm.models import build_model_specs


def test_registry_contains_five_core_models():
    specs = build_model_specs()
    assert set(specs) == {"naive", "logreg", "random_forest", "xgboost", "svm"}


def test_temporal_split_is_chronological():
    idx = pd.date_range("2023-01-01", "2025-06-30", freq="B")
    df = pd.DataFrame({"x": np.arange(len(idx)), "target": 0}, index=idx)
    train, test = validation.temporal_split(df, test_start="2025-01-01")
    assert train.index.max() < pd.Timestamp("2025-01-01")
    assert test.index.min() >= pd.Timestamp("2025-01-01")
    assert len(train) + len(test) + 1 == len(df)  # one row purged at the boundary


def test_tune_model_fits_without_grid():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
    y = pd.Series(rng.integers(0, 2, 100))
    spec = build_model_specs()["naive"]
    model, params = validation.tune_model(spec, X, y)
    assert params == {}
    assert set(model.predict(X)) <= {0, 1}

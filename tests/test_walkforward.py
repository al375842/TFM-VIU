"""M3 ext. tests: walk-forward folds, purging and leakage guards.

Every fold is a new chance to train on data from its own test block; these
tests guard against it, as `test_features_are_causal` does for the features.
"""

import numpy as np
import pandas as pd
import pytest

from tfm import config, data, features, lstm, metrics, validation, walkforward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _index(start: str, end: str) -> pd.DatetimeIndex:
    """Business-day index, the shape of an exchange-traded calendar."""
    return pd.bdate_range(start, end)


@pytest.fixture
def long_index() -> pd.DatetimeIndex:
    return _index("2000-01-03", "2025-12-31")


@pytest.fixture
def short_index() -> pd.DatetimeIndex:
    """An asset with a late start, like ETH-USD (from 2017)."""
    return _index("2017-11-09", "2025-12-31")


def _dataset(index: pd.DatetimeIndex, rng, n_features: int = 3) -> pd.DataFrame:
    """Feature matrix of pure noise plus a random binary target."""
    ds = pd.DataFrame(
        rng.normal(size=(len(index), n_features)),
        index=index,
        columns=[f"f{k}" for k in range(n_features)],
    )
    ds["target"] = rng.integers(0, 2, len(index))
    return ds


# ---------------------------------------------------------------------------
# 1. Temporal order
# ---------------------------------------------------------------------------


def test_train_always_ends_before_the_test_block(long_index):
    folds = validation.make_walkforward_folds(long_index)
    assert folds, "the long index must produce folds"
    for fold in folds:
        assert fold.train_start <= fold.train_end < fold.test_start <= fold.test_end


@pytest.mark.skipif(
    not (config.DATA_RAW_DIR / "GSPC.parquet").exists(),
    reason="raw data not persisted; run the M1 download first",
)
def test_train_always_ends_before_the_test_block_on_real_assets():
    """Same guarantee over the fourteen real calendars, not a synthetic one.

    Each asset uses its own index: calendars differ, so a shared one would
    be wrong here.
    """
    raw = data.load_raw()
    for ticker, df in raw.items():
        folds = validation.make_walkforward_folds(df.index)
        assert folds, f"{ticker} produced no fold"
        for fold in folds:
            assert fold.train_end < fold.test_start, f"{ticker} {fold.test_year}"
            # every fold must actually be inside the asset's own history
            assert fold.train_start >= df.index.min()
            assert fold.test_end <= df.index.max()


# ---------------------------------------------------------------------------
# 2. Purging at the boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("purge", [0, 1, 5])
def test_purge_drops_exactly_k_rows_before_the_test_block(long_index, purge):
    folds = validation.make_walkforward_folds(long_index, purge_rows=purge)
    for fold in folds:
        pos_train_end = long_index.get_loc(fold.train_end)
        pos_test_start = long_index.get_loc(fold.test_start)
        assert pos_test_start - pos_train_end == purge + 1
        assert fold.purge_rows == purge


def test_purge_defaults_to_the_target_horizon(long_index):
    """`WF_PURGE_ROWS = None` must resolve to N, not to zero."""
    folds = validation.make_walkforward_folds(long_index, purge_rows=None)
    assert all(f.purge_rows == config.TARGET_HORIZON_N for f in folds)


def test_no_training_label_reads_into_the_test_block(long_index):
    """The key leakage test at the boundary: y[t] depends on the price at t+N.

    A training row at position p has a label built from the price at p+N; if
    p+N falls in the test block, that label uses data the model is about to
    be scored on. The test fails with `purge_rows=0`, as intended.
    """
    n = config.TARGET_HORIZON_N
    folds = validation.make_walkforward_folds(long_index, purge_rows=n)
    for fold in folds:
        last_train = long_index.get_loc(fold.train_end)
        first_test = long_index.get_loc(fold.test_start)
        # the label of the last training row must resolve strictly before
        # the first test session
        assert last_train + n < first_test


def test_without_purging_the_boundary_label_would_leak(long_index):
    """Shows the check above can fail: without a purge, the boundary leaks."""
    n = config.TARGET_HORIZON_N
    folds = validation.make_walkforward_folds(long_index, purge_rows=0)
    leaks = [
        fold
        for fold in folds
        if long_index.get_loc(fold.train_end) + n
        >= long_index.get_loc(fold.test_start)
    ]
    assert leaks, "with no purge every fold must contaminate its boundary"


# ---------------------------------------------------------------------------
# 3. Coverage and non-overlap
# ---------------------------------------------------------------------------


def test_test_blocks_do_not_overlap_and_cover_the_years(long_index):
    folds = validation.make_walkforward_folds(long_index)
    years = [f.test_year for f in folds]
    expected = list(range(config.WF_FIRST_TEST_YEAR, config.WF_LAST_TEST_YEAR + 1))
    assert years == expected  # consecutive, no gaps, no repeats
    for previous, current in zip(folds, folds[1:]):
        assert previous.test_end < current.test_start
        assert previous.fold_id + 1 == current.fold_id
    for fold in folds:
        assert fold.test_start.year == fold.test_end.year == fold.test_year


def test_anchored_windows_grow_and_rolling_windows_do_not(long_index):
    anchored = validation.make_walkforward_folds(long_index, mode="anchored")
    rolling = validation.make_walkforward_folds(long_index, mode="rolling")
    assert all(f.train_start == anchored[0].train_start for f in anchored)
    assert [f.n_train for f in anchored] == sorted(f.n_train for f in anchored)
    # a fixed window slides instead of growing
    assert rolling[0].train_start < rolling[-1].train_start


def test_unknown_mode_is_rejected(long_index):
    with pytest.raises(ValueError, match="mode"):
        validation.make_walkforward_folds(long_index, mode="sliding")


# ---------------------------------------------------------------------------
# 4. Minimum history
# ---------------------------------------------------------------------------


def test_every_fold_has_the_required_history(long_index, short_index):
    for index in (long_index, short_index):
        for fold in validation.make_walkforward_folds(index):
            span = (fold.test_start - fold.train_start).days / 365.25
            assert span >= config.WF_MIN_TRAIN_YEARS


def test_a_short_asset_yields_fewer_folds_without_failing(long_index, short_index):
    long_folds = validation.make_walkforward_folds(long_index)
    short_folds = validation.make_walkforward_folds(short_index)
    assert 0 < len(short_folds) < len(long_folds)
    # its folds are the last years, never the first
    assert short_folds[-1].test_year == long_folds[-1].test_year
    assert short_folds[0].test_year > long_folds[0].test_year
    # fold ids stay 1..n so the table is readable per asset
    assert [f.fold_id for f in short_folds] == list(range(1, len(short_folds) + 1))


def test_an_asset_without_enough_history_yields_no_fold():
    folds = validation.make_walkforward_folds(_index("2024-01-01", "2025-12-31"))
    assert folds == []


# ---------------------------------------------------------------------------
# 5. The naive floor is learnt fold by fold
# ---------------------------------------------------------------------------


def test_naive_floor_comes_from_the_fold_train_not_the_full_history():
    """The majority class flips between periods; the floor must follow.

    "Up" dominates 2010-2024 and "down" dominates 2025: a floor computed
    over the whole history would report ~0.62, the fold's is ~0.30.
    """
    index = _index("2010-01-01", "2025-12-31")
    rng = np.random.default_rng(0)
    ds = _dataset(index, rng)
    is_2025 = index.year == 2025
    ds.loc[~is_2025, "target"] = rng.choice([0, 1], (~is_2025).sum(), p=[0.3, 0.7])
    ds.loc[is_2025, "target"] = rng.choice([0, 1], is_2025.sum(), p=[0.7, 0.3])

    fold = [
        f for f in validation.make_walkforward_folds(index) if f.test_year == 2025
    ][0]
    cols = features.feature_columns(ds)
    _, y_train, _, y_test = validation.split_by_fold(ds[cols], ds["target"], fold)

    floor = metrics.naive_baseline_accuracy(y_train, y_test)
    assert y_train.mean() > 0.6  # majority in train is "up"
    assert floor == pytest.approx(float(y_test.mean()), abs=1e-12)
    assert floor < 0.4  # and that majority is the wrong call in 2025


# ---------------------------------------------------------------------------
# 6. Leakage: the model must not be able to see its own test block
# ---------------------------------------------------------------------------


def _leakage_dataset(index) -> pd.DataFrame:
    """Target that is noise everywhere except inside the test year.

    There y = 1{f0 > 0} exactly, so a model that saw any of that block learns
    the rule and scores near 1.0, while a clean one stays near its floor.
    Leakage shows up as high accuracy, not as a subtle bias.
    """
    rng = np.random.default_rng(7)
    ds = _dataset(index, rng)
    test_year = ds.index.year == config.WF_LAST_TEST_YEAR
    ds.loc[test_year, "target"] = (ds.loc[test_year, "f0"] > 0).astype(int)
    return ds


def test_the_model_cannot_learn_the_rule_that_only_exists_in_the_test_block():
    index = _index("2005-01-03", f"{config.WF_LAST_TEST_YEAR}-12-31")
    ds = _leakage_dataset(index)
    specs = {k: v for k, v in walkforward.build_model_specs().items()
             if k == "random_forest"}
    folds = [
        f
        for f in validation.make_walkforward_folds(index)
        if f.test_year == config.WF_LAST_TEST_YEAR
    ]
    out = walkforward.run_walkforward_asset(
        "SYNTH", ds, specs, retune=False, folds=folds, verbose=False
    )
    accuracy = float(out.accuracy.iloc[0])
    assert accuracy < 0.65, (
        f"accuracy={accuracy:.3f}: the model reproduced a rule that only "
        "exists inside its own test block, so test rows reached the fit"
    )


def test_the_leakage_probe_does_fire_when_the_test_block_is_in_the_train_set():
    """Control: the same probe, deliberately contaminated, must detect the rule.

    Without it, the test above could pass because the model is weak rather
    than because the split is clean.
    """
    from sklearn.ensemble import RandomForestClassifier

    index = _index("2005-01-03", f"{config.WF_LAST_TEST_YEAR}-12-31")
    ds = _leakage_dataset(index)
    cols = features.feature_columns(ds)
    model = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=1)
    model.fit(ds[cols], ds["target"])  # trained on everything, test included

    is_test = ds.index.year == config.WF_LAST_TEST_YEAR
    contaminated = float(
        (model.predict(ds.loc[is_test, cols]) == ds.loc[is_test, "target"]).mean()
    )
    assert contaminated > 0.90


# ---------------------------------------------------------------------------
# 7. LSTM sequences, fold by fold
# ---------------------------------------------------------------------------


def test_fold_sequences_are_assigned_by_the_date_they_predict():
    """Analogue of `test_sequences_are_causal`, at fold granularity."""
    index = _index("2010-01-01", "2025-12-31")
    fold = [
        f for f in validation.make_walkforward_folds(index) if f.test_year == 2020
    ][0]
    window = config.LSTM_WINDOW_T

    seq_index, is_train, is_test = lstm.sequence_split(
        index, window=window, train_end=fold.train_end,
        test_start=fold.test_start, test_end=fold.test_end,
    )
    assert not (is_train & is_test).any()  # no sequence in both blocks
    assert seq_index[is_train].max() <= fold.train_end
    assert seq_index[is_test].min() >= fold.test_start
    assert seq_index[is_test].max() <= fold.test_end
    # nothing after the fold exists at all: a later year cannot influence it
    assert seq_index.max() <= fold.test_end


def test_the_first_test_sequence_reaches_back_into_training_rows():
    """Allowed and necessary: those sessions are already in the past.

    Building the sequences after the split would lose the first T-1
    predictions of each test block, so the LSTM would be scored on a
    different test set from the other five models.
    """
    index = _index("2010-01-01", "2025-12-31")
    fold = [
        f for f in validation.make_walkforward_folds(index) if f.test_year == 2020
    ][0]
    window = config.LSTM_WINDOW_T

    seq_index, _, is_test = lstm.sequence_split(
        index, window=window, train_end=fold.train_end,
        test_start=fold.test_start, test_end=fold.test_end,
    )
    first_test = seq_index[is_test][0]
    history_start = index[index.get_loc(first_test) - window + 1]
    assert history_start < fold.test_start  # reads into the training window
    assert first_test == fold.test_start  # but predicts the very first session


def test_fold_sequences_ignore_everything_after_the_fold():
    """Tampering with later years must not move this fold's sequences."""
    index = _index("2010-01-01", "2025-12-31")
    fold = [
        f for f in validation.make_walkforward_folds(index) if f.test_year == 2018
    ][0]
    kwargs = dict(window=config.LSTM_WINDOW_T, train_end=fold.train_end,
                  test_start=fold.test_start, test_end=fold.test_end)

    base, _, _ = lstm.sequence_split(index, **kwargs)
    extended, _, _ = lstm.sequence_split(index.append(_index("2026-01-01",
                                                             "2026-12-31")),
                                         **kwargs)
    assert base.equals(extended)


# ---------------------------------------------------------------------------
# Aggregation and the stability test
# ---------------------------------------------------------------------------


def test_sign_test_is_one_sided():
    """Beating the floor in 9 of 10 folds is notable; in 1 of 10 is not."""
    assert walkforward.fold_sign_test(10, 9) < 0.05
    assert walkforward.fold_sign_test(10, 1) > 0.95
    assert walkforward.fold_sign_test(10, 5) == pytest.approx(0.6230, abs=1e-3)
    assert np.isnan(walkforward.fold_sign_test(0, 0))


def test_aggregate_reports_shares_and_fold_counts():
    """Assets differ in fold count, so only the share is comparable."""
    rows = pd.DataFrame(
        [
            {"ticker": "A", "model": "logreg", "fold_id": k, "test_year": 2016 + k,
             "accuracy": 0.5 + 0.01 * k, "balanced_accuracy": 0.5,
             "base_rate": 0.5, "share_predicted_up": 0.6,
             "above_baseline": k % 2 == 0, "p_value_binomial": 0.4}
            for k in range(10)
        ]
        + [
            {"ticker": "B", "model": "logreg", "fold_id": k, "test_year": 2022 + k,
             "accuracy": 0.6, "balanced_accuracy": 0.55, "base_rate": 0.5,
             "share_predicted_up": 0.5, "above_baseline": True,
             "p_value_binomial": 0.01}
            for k in range(3)
        ]
    )
    summary = walkforward.aggregate_walkforward(rows)
    assert summary.loc[("A", "logreg"), "n_folds"] == 10
    assert summary.loc[("B", "logreg"), "n_folds"] == 3
    assert summary.loc[("A", "logreg"), "share_folds_above"] == pytest.approx(0.5)
    assert summary.loc[("B", "logreg"), "share_folds_above"] == pytest.approx(1.0)
    assert summary.loc[("B", "logreg"), "n_folds_p_below_05"] == 3


def test_folds_table_reconstructs_the_design(long_index):
    """`walkforward_folds.csv` must describe the design without the code."""
    folds = validation.make_walkforward_folds(long_index)
    table = validation.folds_to_frame("SYNTH", folds)
    assert list(table.columns) == [
        "ticker", "fold_id", "test_year", "train_start", "train_end",
        "test_start", "test_end", "n_train", "n_test", "purge_rows",
    ]
    assert len(table) == len(folds)
    assert (table.n_train > 0).all() and (table.n_test > 0).all()

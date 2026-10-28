"""M3 extension (optional) - LSTM network.

Not part of the core pipeline: TensorFlow is an optional extra ([lstm]). The
network is small (19,265 parameters) and trains on CPU in ~35 s per asset.
Thesis: §4.5 (Listings 4.12 and 4.13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config, validation


def build_sequences(
    X: np.ndarray, y: np.ndarray, window: int = config.LSTM_WINDOW_T
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows of length T: (n, feats) -> (n - T + 1, T, feats).

    Each window ends on the session it predicts: y[t] comes from
    X[t-T+1 .. t], row t included. That is not leakage: row t only holds
    features up to t (it is y[t] that depends on t+N), and the other models
    see row t too.
    """
    X_seq, y_seq = [], []
    for i in range(window - 1, len(X)):
        X_seq.append(X[i - window + 1 : i + 1])  # past T sessions, t included
        y_seq.append(y[i])
    return np.asarray(X_seq), np.asarray(y_seq)


def sequence_split(index, *, window, train_end, test_start, test_end=None):
    """Assign each sequence to train or test by the date it predicts.

    Returns (seq_index, is_train, is_test). Kept apart from the training code
    so it can be tested without TensorFlow. Rows after `test_end` are dropped
    first, so later data never reaches this fold. A sequence belongs to the
    block of the row it predicts, not of the rows it reads: the first test
    sequence reads T-1 training sessions, which are already in the past.
    """
    index = pd.DatetimeIndex(index)
    if test_end is not None:
        index = index[index <= test_end]
    seq_index = index[window - 1:]
    return seq_index, seq_index <= train_end, seq_index >= test_start


def build_lstm(n_features: int, window: int = config.LSTM_WINDOW_T):
    """Deliberately small LSTM, to limit cost and overfitting (Listing 4.13)."""
    try:
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required for the LSTM extension: "
            'pip install -e ".[lstm]". Training runs on CPU; no GPU needed.'
        ) from exc

    model = Sequential(
        [
            LSTM(config.LSTM_UNITS, input_shape=(window, n_features),
                 return_sequences=False),
            Dropout(config.LSTM_DROPOUT),  # regularisation
            Dense(1, activation="sigmoid"),  # binary direction output
        ]
    )
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def fit_predict_range(
    dataset,
    feature_cols: list[str],
    *,
    train_end,
    test_start,
    test_end=None,
    window: int = config.LSTM_WINDOW_T,
    verbose: int = 0,
) -> pd.Series:
    """Train on rows up to `train_end` and predict [`test_start`, `test_end`].

    Called once per walk-forward fold, with `train_end` already purged by the
    caller. Three details matter: rows after `test_end` are dropped first; the
    scaler is fitted on training rows only (there is no sklearn Pipeline here
    to do it); and sequences are built before being split, so the test block
    keeps its first T-1 predictions.
    """
    import keras  # lazy: TensorFlow is not a core dependency

    keras.backend.clear_session()
    keras.utils.set_random_seed(config.RANDOM_STATE)

    ds = dataset if test_end is None else dataset.loc[:test_end]
    is_train = ds.index <= train_end
    scaler = StandardScaler().fit(ds.loc[is_train, feature_cols])
    X = scaler.transform(ds[feature_cols]).astype("float32")
    y = ds["target"].to_numpy().astype("float32")

    X_seq, y_seq = build_sequences(X, y, window=window)
    seq_index, seq_is_train, seq_is_test = sequence_split(
        ds.index, window=window, train_end=train_end, test_start=test_start
    )

    model = build_lstm(n_features=len(feature_cols), window=window)
    model.fit(
        X_seq[seq_is_train],
        y_seq[seq_is_train],
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        shuffle=False,  # keep the chronological order of the sequences
        verbose=verbose,
    )

    proba = model.predict(X_seq[seq_is_test], verbose=verbose).ravel()
    return pd.Series(
        (proba > 0.5).astype(int), index=seq_index[seq_is_test], name="lstm"
    )


def fit_predict(
    dataset,
    feature_cols: list[str],
    test_start: str = config.TEST_START,
    window: int = config.LSTM_WINDOW_T,
    verbose: int = 0,
) -> pd.Series:
    """Single-split entry point: train before `test_start`, predict from it.

    Predictions are indexed exactly like the test rows of the other models,
    so accuracy, McNemar and the backtest compare like with like.
    """
    train, _ = validation.temporal_split(dataset, test_start=test_start)
    if train.empty:
        raise ValueError(f"No training rows remain after purging before {test_start}")
    return fit_predict_range(
        dataset,
        feature_cols,
        train_end=train.index[-1],
        test_start=test_start,
        test_end=dataset.index[-1],
        window=window,
        verbose=verbose,
    )

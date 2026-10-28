"""M3 extension tests: sequence construction (no TensorFlow required)."""

import numpy as np

from tfm import lstm


def test_window_closes_on_the_predicted_session():
    """To predict y[t] the window must include row t.

    Regression: it used to stop at t-1, so the LSTM was the only model that
    never saw the most recent session. Row t is causal: its features come
    from data up to t, while y[t] depends on t+N.
    """
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    X_seq, y_seq = lstm.build_sequences(X, y, window=3)

    # first sequence predicts y[2] from rows 0,1,2
    assert y_seq[0] == 2
    assert X_seq[0].ravel().tolist() == [0, 1, 2]
    # and every sequence ends on the row whose target it predicts
    for k, target in enumerate(y_seq):
        assert X_seq[k].ravel()[-1] == target


def test_sequence_count_and_shape():
    X = np.zeros((100, 4))
    y = np.zeros(100)
    X_seq, y_seq = lstm.build_sequences(X, y, window=30)
    assert X_seq.shape == (71, 30, 4)  # 100 - 30 + 1
    assert y_seq.shape == (71,)


def test_sequences_are_causal():
    """Altering the future must not change an earlier sequence."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = rng.integers(0, 2, 60)
    base, _ = lstm.build_sequences(X, y, window=10)

    tampered_X = X.copy()
    tampered_X[40:] *= 99
    tampered, _ = lstm.build_sequences(tampered_X, y, window=10)

    # sequences whose window ends before row 40 must be untouched
    assert np.array_equal(base[:31], tampered[:31])

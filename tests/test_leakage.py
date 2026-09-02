"""
The shuffled-label leakage test.

If a classifier trained on a random permutation of the training labels can
still predict the TRUE test labels above the base rate, the leak is not in
the model — it's in the features or the split (e.g. a feature that encodes
the label, or train/test rows that aren't actually independent).
"""

import numpy as np

from src.baselines import predict_shuffled_label, train_shuffled_label


def test_shuffled_label_does_not_learn_real_signal():
    rng = np.random.default_rng(0)
    n = 3000
    x0 = rng.normal(size=n)
    y = (x0 > 0).astype(int)  # perfectly separable on x0 alone
    X = np.column_stack([x0, rng.normal(size=(n, 4))])  # + 4 pure-noise columns

    split = n // 2
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = train_shuffled_label(X_train, y_train, seed=1)
    preds = predict_shuffled_label(model, X_test)
    acc = (preds == y_test).mean()
    base_rate = max(y_test.mean(), 1 - y_test.mean())

    # A real (non-shuffled) model on this data scores ~1.0 — the signal is
    # trivially separable. Training on permuted labels must destroy access
    # to it; anything scoring well above the base rate here means labels
    # are leaking through somewhere in the harness, not in this baseline.
    assert acc <= base_rate + 0.05, (
        f"shuffled-label model scored {acc:.3f} against a base rate of "
        f"{base_rate:.3f} on perfectly separable features — investigate for a leak"
    )

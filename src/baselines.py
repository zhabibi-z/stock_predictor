"""Stage 3b — Baseline Predictors.

No model's accuracy means anything until it's shown next to these three.
Every result table in this project puts them in the same table as the
trained models — not in prose beneath it.

  always_long        The null hypothesis for a long-only strategy on a
                      stock with positive secular drift: predict UP, every
                      day, unconditionally. Its accuracy on any fold IS
                      that fold's UP base rate.
  prev_day_momentum   Predicts Target[T] from the sign of Daily_Return[T]
                      — "the move already realized into the close of today
                      continues into tomorrow." Uses only information
                      available at the close of day T; no leakage.
  shuffled_label      Trains a real classifier (GaussianNB) on a random
                      permutation of the training labels, then predicts
                      the real test labels. Because the permutation
                      destroys any true label/feature relationship, its
                      accuracy should sit at the fold's base rate. Scoring
                      meaningfully above that is evidence of a leak
                      somewhere in the features or the split — not in this
                      model.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB


def train_always_long(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    return {"kind": "always_long"}


def predict_always_long(model: dict, X_test: np.ndarray) -> np.ndarray:
    return np.ones(len(X_test), dtype=int)


def train_prev_day_momentum(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    return {"kind": "prev_day_momentum"}


def predict_prev_day_momentum(model: dict, daily_return_today: np.ndarray) -> np.ndarray:
    """daily_return_today: raw (unscaled) Daily_Return for each test row."""
    return (np.asarray(daily_return_today) > 0).astype(int)


def train_shuffled_label(X_train: np.ndarray, y_train: np.ndarray, seed: int = 0) -> GaussianNB:
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y_train)
    model = GaussianNB()
    model.fit(X_train, y_shuffled)
    return model


def predict_shuffled_label(model: GaussianNB, X_test: np.ndarray) -> np.ndarray:
    return model.predict(X_test)

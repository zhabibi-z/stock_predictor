"""Stage 3 — Model Definitions, Training, and Classification Metrics.

Three models of increasing complexity:
  1. GaussianNB          — probabilistic baseline, optional balanced sample_weight
  2. Logistic Regression — direct classifier, optional balanced class_weight
  3. MLP                 — 128→64→32, BatchNorm, Dropout, ReduceLROnPlateau

Every model takes a `balanced: bool` flag rather than hardcoding balanced
weighting: on a stock with a ~54/46 up/down split, that split is the equity
risk premium, not class imbalance — forcing balance makes a model bet
against secular drift. Both settings should be run and compared, not one
silently chosen.

Metrics (compute_metrics, print_report, print_walk_forward_report,
aggregate_walk_forward) live here because they evaluate model outputs.

TensorFlow (~600MB installed) is imported lazily, inside the MLP functions
only, so the NB and Logistic Regression paths — and anything importing this
module just for metrics — never pay for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.naive_bayes        import GaussianNB
from sklearn.linear_model       import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics            import (
    accuracy_score, f1_score, precision_score, recall_score,
)

if TYPE_CHECKING:
    from tensorflow import keras


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Gaussian Naive Bayes
# ─────────────────────────────────────────────────────────────────────────────

def train_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, balanced: bool = True) -> GaussianNB:
    """
    Fit GaussianNB, optionally with balanced sample weights.

    GaussianNB has no class_weight parameter, so balanced weighting is
    achieved via sample_weight in fit().  compute_sample_weight("balanced")
    assigns each sample a weight inversely proportional to its class
    frequency, giving minority-class days the same aggregate influence as
    majority-class days.
    """
    weights = compute_sample_weight("balanced", y_train) if balanced else None
    model   = GaussianNB()
    model.fit(X_train, y_train, sample_weight=weights)
    return model


def predict_naive_bayes(model: GaussianNB, X_test: np.ndarray) -> np.ndarray:
    return model.predict(X_test)


def predict_proba_naive_bayes(model: GaussianNB, X_test: np.ndarray) -> np.ndarray:
    return model.predict_proba(X_test)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
# Replaces the earlier Ridge-regression-on-price-then-threshold path: Ridge
# regressed Next_Close (a dollar level dominated by SMA_5/SMA_20, R²=0.959
# vs 0.973 for a naive "tomorrow=today" forecast) and derived a direction
# from forecast > today_close — a laundered random walk, not a direction
# model, and "balanced" sample weights on an MSE loss over dollar prices
# never meant anything. Logistic Regression is a direct classifier: its
# class_weight="balanced" option is the real, principled version of what
# the old code's weighting scheme only imitated.

def train_logistic(
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    balanced: bool  = True,
    max_iter: int   = 1000,
) -> LogisticRegression:
    model = LogisticRegression(
        class_weight="balanced" if balanced else None,
        max_iter=max_iter,
    )
    model.fit(X_train, y_train)
    return model


def predict_proba_logistic(model: LogisticRegression, X_test: np.ndarray) -> np.ndarray:
    return model.predict_proba(X_test)[:, 1]


def predict_logistic(model: LogisticRegression, X_test: np.ndarray, threshold: float = 0.50) -> np.ndarray:
    return (predict_proba_logistic(model, X_test) >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — MLP (Keras)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mlp(
    input_dim:      int,
    seed:           int   = 42,
    hidden_layers:  tuple = (128, 64, 32),
    dropout_rates:  tuple = (0.30, 0.20, 0.15),
    learning_rate:  float = 1e-3,
) -> keras.Model:
    import tensorflow as tf
    from tensorflow import keras

    # tf.random.set_seed alone does not reproduce a Keras run: dropout masks,
    # BatchNorm's running stats update order, and the validation_split shuffle
    # all draw from sources it doesn't seed. set_random_seed seeds Python's
    # `random`, numpy, and TF together; enable_op_determinism additionally
    # forces deterministic kernel selection for ops that otherwise pick a
    # nondeterministic (faster) implementation. Verified: same seed, same
    # data, 10 epochs -> bit-identical predictions across two runs.
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    layers = [keras.layers.Input(shape=(input_dim,))]
    for i, (units, rate) in enumerate(zip(hidden_layers, dropout_rates)):
        layers.append(keras.layers.Dense(units, activation="relu"))
        if i < len(hidden_layers) - 1:
            layers.append(keras.layers.BatchNormalization())
        layers.append(keras.layers.Dropout(rate))
    layers.append(keras.layers.Dense(1, activation="sigmoid"))

    model = keras.Sequential(layers, name="StockMLP_v2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_mlp(
    X_train:            np.ndarray,
    y_train:            np.ndarray,
    epochs:             int   = 150,
    batch_size:         int   = 32,
    patience:           int   = 15,
    seed:               int   = 42,
    balanced:           bool  = True,
    hidden_layers:      tuple = (128, 64, 32),
    dropout_rates:      tuple = (0.30, 0.20, 0.15),
    learning_rate:      float = 1e-3,
    lr_reduce_factor:   float = 0.5,
    lr_reduce_patience: int   = 7,
    min_lr:             float = 1e-6,
) -> keras.Model:
    """
    Train the MLP with EarlyStopping, ReduceLROnPlateau, and optionally
    balanced class weights.

    Validation split uses the chronologically latest 10 % of the training
    fold (Keras takes validation_split from the tail), so no test-fold data
    leaks in.  ReduceLROnPlateau halves Adam's LR after `lr_reduce_patience`
    stagnant epochs, preventing oscillation near the optimum that a fixed LR
    causes late in training.
    """
    from tensorflow import keras

    cw_dict = None
    if balanced:
        classes = np.unique(y_train)
        raw_w   = compute_class_weight("balanced", classes=classes, y=y_train)
        cw_dict = dict(zip(classes.tolist(), raw_w.tolist()))

    model = _build_mlp(X_train.shape[1], seed=seed, hidden_layers=hidden_layers,
                        dropout_rates=dropout_rates, learning_rate=learning_rate)
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.10,
        class_weight=cw_dict,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=lr_reduce_factor,
                patience=lr_reduce_patience, min_lr=min_lr, verbose=0,
            ),
        ],
        verbose=0,
    )
    return model


def predict_mlp(
    model:     keras.Model,
    X_test:    np.ndarray,
    threshold: float = 0.50,
) -> np.ndarray:
    probs = model.predict(X_test, verbose=0).flatten()
    return (probs >= threshold).astype(int)


def predict_proba_mlp(model: keras.Model, X_test: np.ndarray) -> np.ndarray:
    return model.predict(X_test, verbose=0).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    model_name: str,
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
) -> dict:
    """
    Return a metrics dict with all four scores as percentages.

    zero_division=0 silences warnings when a class is never predicted,
    which can happen on a short test fold.
    """
    return {
        "Model":     model_name,
        "Accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
    }


def aggregate_walk_forward(fold_metrics: dict) -> dict:
    """Compute mean ± std across walk-forward folds for each model."""
    aggregated = {}
    for key, folds in fold_metrics.items():
        if not folds:
            continue
        agg = {"Model": folds[0]["Model"]}
        for metric in ("Accuracy", "Precision", "Recall", "F1-Score"):
            vals = np.array([f[metric] for f in folds])
            agg[metric] = f"{vals.mean():.2f} ± {vals.std():.2f}"
        aggregated[key] = agg
    return aggregated


def print_report(results: list) -> None:
    col = [32, 13, 14, 11, 13]
    sep = "+" + "+".join("─" * w for w in col) + "+"

    def _row(vals):
        cells = [f" {str(v):<{col[i] - 2}} " for i, v in enumerate(vals)]
        return "|" + "|".join(cells) + "|"

    print()
    print(sep)
    print(_row(["Model", "Accuracy", "Precision", "Recall", "F1-Score"]))
    print(sep)
    for r in results:
        print(_row([
            r["Model"],
            f"{r['Accuracy']} %",
            f"{r['Precision']} %",
            f"{r['Recall']} %",
            f"{r['F1-Score']} %",
        ]))
    print(sep)


def print_walk_forward_report(fold_metrics: dict) -> None:
    aggregated = aggregate_walk_forward(fold_metrics)

    col = [32, 18, 18, 18, 18]
    sep = "+" + "+".join("─" * w for w in col) + "+"

    def _row(vals):
        cells = [f" {str(v):<{col[i] - 2}} " for i, v in enumerate(vals)]
        return "|" + "|".join(cells) + "|"

    print()
    print("  Walk-Forward Summary  (mean ± std across all folds)")
    print(sep)
    print(_row(["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]))
    print(sep)
    for agg in aggregated.values():
        print(_row([agg["Model"], agg["Accuracy"], agg["Precision"], agg["Recall"], agg["F1-Score"]]))
    print(sep)

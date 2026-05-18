"""Stage 3 — Model Definitions, Training, and Classification Metrics.

Three models of increasing complexity:
  1. GaussianNB        — probabilistic baseline, balanced sample_weight
  2. Ridge Regression  — price-forecast → direction, balanced sample_weight
  3. MLP               — 128→64→32, BatchNorm, Dropout, ReduceLROnPlateau

Metrics (compute_metrics, print_report, print_walk_forward_report,
aggregate_walk_forward) live here because they evaluate model outputs.
"""

import numpy as np
from sklearn.naive_bayes       import GaussianNB
from sklearn.linear_model      import Ridge
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics           import (
    accuracy_score, f1_score, precision_score, recall_score,
)
import tensorflow as tf
from tensorflow import keras


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Gaussian Naive Bayes
# ─────────────────────────────────────────────────────────────────────────────

def train_naive_bayes(X_train: np.ndarray, y_train: np.ndarray) -> GaussianNB:
    """
    Fit GaussianNB with balanced sample weights.

    GaussianNB has no class_weight parameter, so balanced weighting is
    achieved via sample_weight in fit().  compute_sample_weight("balanced")
    assigns each sample a weight inversely proportional to its class
    frequency, giving minority-class days the same aggregate influence as
    majority-class days.
    """
    weights = compute_sample_weight("balanced", y_train)
    model   = GaussianNB()
    model.fit(X_train, y_train, sample_weight=weights)
    return model


def predict_naive_bayes(model: GaussianNB, X_test: np.ndarray) -> np.ndarray:
    return model.predict(X_test)


def predict_proba_naive_bayes(model: GaussianNB, X_test: np.ndarray) -> np.ndarray:
    return model.predict_proba(X_test)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Ridge Regression → binary direction
# ─────────────────────────────────────────────────────────────────────────────

def train_ridge(
    X_train:     np.ndarray,
    y_reg_train: np.ndarray,
    y_dir_train: np.ndarray,
    alpha:       float = 1.0,
) -> Ridge:
    """
    Fit Ridge with balanced sample weights derived from direction labels.

    Ridge has no class_weight parameter (it's a regressor), so weights come
    from the binary direction labels y_dir_train.  L2 regularisation shrinks
    correlated SMA/EMA coefficients without zeroing any predictor — giving
    better OOS forecasts than plain OLS on these highly correlated features.
    """
    weights = compute_sample_weight("balanced", y_dir_train)
    model   = Ridge(alpha=alpha)
    model.fit(X_train, y_reg_train, sample_weight=weights)
    return model


def predict_ridge_direction(
    model:       Ridge,
    X_test:      np.ndarray,
    today_close: np.ndarray,
) -> np.ndarray:
    """Forecast tomorrow's close; return 1 where forecast > today_close."""
    forecast = model.predict(X_test)
    return (forecast > today_close).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — MLP (Keras)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mlp(input_dim: int, seed: int = 42) -> keras.Model:
    tf.random.set_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.30),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.20),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="StockMLP_v2",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_mlp(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    epochs:     int = 150,
    batch_size: int = 32,
    patience:   int = 15,
    seed:       int = 42,
) -> keras.Model:
    """
    Train the MLP with EarlyStopping, ReduceLROnPlateau, and balanced
    class weights.

    Validation split uses the chronologically latest 10 % of the training
    fold (Keras takes validation_split from the tail), so no test-fold data
    leaks in.  ReduceLROnPlateau halves Adam's LR after 7 stagnant epochs,
    preventing oscillation near the optimum that a fixed LR causes late in
    training.
    """
    classes = np.unique(y_train)
    raw_w   = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes.tolist(), raw_w.tolist()))

    model = _build_mlp(X_train.shape[1], seed=seed)
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
                monitor="val_loss", factor=0.5,
                patience=7, min_lr=1e-6, verbose=0,
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

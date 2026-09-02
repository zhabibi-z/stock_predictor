"""Stage 3d — Threshold tuning.

The decision threshold on a predicted probability is a hyperparameter like
any other. Hardcoding 0.50 is an unstated assumption that the operating
point that maximizes accuracy happens to sit exactly at the midpoint of the
model's output — true only by coincidence. This module picks a threshold on
the validation split (never the holdout, never a walk-forward test fold)
and that one fixed choice is then applied everywhere else.

Selection criterion is balanced accuracy, not raw accuracy — tried first,
found to fail: on this ~54/46 split, grid-searching raw accuracy drives the
threshold down until the model predicts the majority class almost
unconditionally (recall -> 100%, precision -> the base rate), because that
degenerate point IS the raw-accuracy optimum whenever there's little real
separating signal. That isn't a tuned model, it's always_long wearing a
threshold. Balanced accuracy — the average of the per-class recalls —
penalizes collapsing to one class, so the selected threshold has to
actually discriminate between UP and DOWN to score well.
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def tune_threshold(y_val: np.ndarray, probs_val: np.ndarray, grid: np.ndarray = None) -> tuple:
    """
    Grid-search the threshold that maximizes BALANCED accuracy on
    (y_val, probs_val) — see module docstring for why not raw accuracy.
    Returns (best_threshold, best_validation_balanced_accuracy).
    """
    if grid is None:
        grid = np.linspace(0.30, 0.70, 41)
    best_t, best_score = 0.50, -1.0
    for t in grid:
        score = balanced_accuracy_score(y_val, (probs_val >= t).astype(int))
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score


def threshold_report(y_val: np.ndarray, probs_val: np.ndarray, threshold: float) -> dict:
    """Both accuracy and balanced accuracy at a given threshold, for reporting."""
    preds = (probs_val >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_val, preds),
        "balanced_accuracy": balanced_accuracy_score(y_val, preds),
    }

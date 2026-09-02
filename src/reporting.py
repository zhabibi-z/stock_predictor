"""Stage 3c — CI-aware reporting.

Every table this module prints puts baselines in the same table as the
trained models, and every accuracy figure carries either a Wilson/bootstrap
interval (deterministic models, at a given fold's n) or a mean ± std across
seeds (the MLP, whose training is stochastic — the seed itself is a source
of variance a point estimate hides).
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .intervals import bootstrap_ci, wilson_interval

_METRICS = ("Accuracy", "Precision", "Recall", "F1-Score")


def compute_metrics_ci(model_name: str, y_true, y_pred, n_boot: int = 2000, seed: int = 0) -> dict:
    """Point estimate + interval for all four metrics, in one dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    correct = int((y_true == y_pred).sum())

    row = {
        "Model": model_name,
        "n": n,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
    }
    row["Accuracy_CI"]  = wilson_interval(correct, n)
    row["Precision_CI"] = bootstrap_ci(y_true, y_pred, lambda t, p: precision_score(t, p, zero_division=0), n_boot=n_boot, seed=seed)
    row["Recall_CI"]    = bootstrap_ci(y_true, y_pred, lambda t, p: recall_score(t, p, zero_division=0), n_boot=n_boot, seed=seed + 1)
    row["F1-Score_CI"]  = bootstrap_ci(y_true, y_pred, lambda t, p: f1_score(t, p, zero_division=0), n_boot=n_boot, seed=seed + 2)
    return row


def aggregate_mean_std(rows: list, group_label: str) -> dict:
    """
    Collapse a list of metrics dicts — one per fold, or one per MLP training
    seed — into a single summary reporting mean ± std across that group.
    `group_label` is stored as the count of rows collapsed (n_folds / n_seeds).
    """
    if not rows:
        return None
    out = {"Model": rows[0]["Model"], group_label: len(rows)}
    for metric in _METRICS:
        vals = np.array([r[metric] for r in rows])
        out[metric] = float(vals.mean())
        out[f"{metric}_std"] = float(vals.std())
    return out


def fmt_row_ci(row: dict) -> list:
    """Format a compute_metrics_ci row as ['Model', 'pt% [lo,hi]', ...]."""
    def f(metric):
        pt = row[metric]
        lo, hi = row[f"{metric}_CI"]
        return f"{pt * 100:.2f}% [{lo * 100:.1f}, {hi * 100:.1f}]"
    return [row["Model"], f("Accuracy"), f("Precision"), f("Recall"), f("F1-Score")]


def fmt_row_mean_std(row: dict, group_label: str, group_word: str) -> list:
    """Format an aggregate_mean_std row as ['Model (folds=N)', 'mean% ± std', ...]."""
    def f(metric):
        return f"{row[metric] * 100:.2f}% ± {row[f'{metric}_std'] * 100:.2f}"
    label = f"{row['Model']} ({group_word}={row[group_label]})"
    return [label, f("Accuracy"), f("Precision"), f("Recall"), f("F1-Score")]


def print_table(title: str, header: list, rows: list) -> None:
    if rows:
        col = [max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(header)]
    else:
        col = [len(h) + 2 for h in header]
    sep = "+" + "+".join("─" * w for w in col) + "+"

    def _row(vals):
        cells = [f" {str(v):<{col[i] - 2}} " for i, v in enumerate(vals)]
        return "|" + "|".join(cells) + "|"

    print()
    print(f"  {title}")
    print(sep)
    print(_row(header))
    print(sep)
    for r in rows:
        print(_row(r))
    print(sep)


def check_leakage(shuffled_row: dict, base_rate: float, model_label: str = "shuffled-label") -> bool:
    """
    Returns True (and prints a loud warning) if the shuffled-label baseline's
    accuracy CI sits strictly above the fold's own base rate — i.e. a
    classifier trained on noise is beating the trivial predictor. That is
    not possible without a leak somewhere in the features or the split.
    """
    lo, hi = shuffled_row["Accuracy_CI"]
    if lo > base_rate:
        print(f"\n  {'!' * 60}")
        print(f"  LEAK DETECTED: {model_label} accuracy CI [{lo*100:.2f}%, {hi*100:.2f}%] "
              f"sits entirely above the base rate ({base_rate*100:.2f}%).")
        print("  A model trained on permuted labels cannot beat the base rate honestly.")
        print(f"  {'!' * 60}")
        return True
    return False

"""Confidence intervals for classification metrics.

At n≈350-410 (one walk-forward fold), a bare accuracy percentage is not a
result — a few points of gap between two models can be pure sampling
noise. Every accuracy figure this project prints carries a Wilson score
interval (closed-form, exact for a binomial proportion, no extra
dependency); precision/recall/F1 carry a paired bootstrap interval since
their denominators aren't fixed the way accuracy's is.
"""

import numpy as np

_Z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """Wilson score interval for a binomial proportion. Returns (low, high) as fractions in [0, 1]."""
    if n == 0:
        return (0.0, 0.0)
    z = _Z.get(round(confidence, 2), 1.9600)
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple:
    """Percentile bootstrap CI for metric_fn(y_true, y_pred), resampling rows in pairs."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = metric_fn(y_true[idx], y_pred[idx])
    lo_pct = (1 - confidence) / 2 * 100
    hi_pct = (1 + confidence) / 2 * 100
    return (float(np.percentile(stats, lo_pct)), float(np.percentile(stats, hi_pct)))


def fmt_pct_ci(point: float, lo: float, hi: float) -> str:
    """point/lo/hi are fractions in [0, 1]; formats as '52.80% [47.96, 57.64]'."""
    return f"{point * 100:.2f}% [{lo * 100:.2f}, {hi * 100:.2f}]"

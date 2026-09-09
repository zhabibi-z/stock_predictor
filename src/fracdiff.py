"""Stage 2b — Fixed-Width Window Fractional Differentiation (FFD).

Replaces the manual "Close/SMA - 1" patch (Phase 2) with the general,
principled fix for the same problem: raw prices are non-stationary, but
integer differencing (e.g. plain returns) throws away essentially all of
the series' memory/trend information along with the non-stationarity.
Fractional differentiation finds the minimum differencing order d in
[0, 1] that achieves stationarity while preserving as much memory
(correlation with the original series) as possible. See Lopez de Prado,
"Advances in Financial Machine Learning" (2018), ch. 5.

Weights
───────
For a series X and order d, the FFD weight at lag k is

    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k

which converges to 0 as k grows (for 0 < d <= 1). Truncating once
|w_k| < threshold gives a *fixed-width* window: the transform at time t
only ever looks backward width-1 steps, so it is as causal/leak-free as a
rolling mean of the same width — no forward-looking information, no
change to the leakage-prevention contract in features.py.

Finding d
─────────
find_min_ffd_d searches a grid of d values, computes the FFD series for
each, and runs an Augmented Dickey-Fuller test (statsmodels). The minimum
d whose ADF p-value is below the threshold is the answer: the least
amount of differencing that still achieves stationarity. This must be
run on the training data only (never validation or the holdout) — the
same rule as every other tuned parameter in this project — after which
the resulting fixed weights are applied causally across the entire
series, train through holdout.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def frac_diff_weights(d: float, threshold: float = 1e-4, max_width: int = 10_000) -> np.ndarray:
    """
    Weights for fixed-width-window fractional differentiation at order d,
    truncated once |w_k| falls below `threshold`. Returned oldest-lag-last,
    i.e. weights[0] corresponds to lag 0 (the current observation).
    """
    weights = [1.0]
    k = 1
    while k < max_width:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
    return np.array(weights)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """
    Apply fixed-width-window fractional differentiation to `series` at
    order d. The first `width - 1` rows are NaN (insufficient history),
    matching how every other rolling-window feature in this project
    handles its warm-up period.
    """
    weights = frac_diff_weights(d, threshold)
    width = len(weights)
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    for t in range(width - 1, n):
        window = values[t - width + 1 : t + 1][::-1]  # lag 0 first, matching weights[0]
        if np.isnan(window).any():
            continue
        out[t] = float(np.dot(weights, window))
    return pd.Series(out, index=series.index)


def find_min_ffd_d(
    series: pd.Series,
    d_grid: np.ndarray = None,
    threshold: float = 1e-4,
    adf_pvalue_max: float = 0.05,
) -> dict:
    """
    Grid-search the minimum d in d_grid whose FFD series passes an ADF
    stationarity test (p-value <= adf_pvalue_max). Must be called with
    training-only data.

    Returns a dict with the chosen d, its ADF p-value, the correlation
    between the FFD series and the original (the "memory preserved"
    evidence), and the full per-d diagnostic table for transparency.
    """
    if d_grid is None:
        d_grid = np.round(np.arange(0.0, 1.01, 0.05), 2)

    rows = []
    chosen = None
    for d in d_grid:
        ffd = frac_diff_ffd(series, float(d), threshold)
        valid = ffd.dropna()
        if len(valid) < 20:
            rows.append({"d": float(d), "adf_pvalue": float("nan"), "corr_with_original": float("nan"), "n_obs": len(valid)})
            continue
        # Plain-tuple return (not result_object=True): compatible with the
        # pinned statsmodels==0.14.2, which predates that keyword.
        adf_pvalue = adfuller(valid.values, maxlag=1, autolag=None)[1]
        corr = float(np.corrcoef(valid.values, series.reindex(valid.index).values)[0, 1])
        rows.append({"d": float(d), "adf_pvalue": float(adf_pvalue), "corr_with_original": corr, "n_obs": len(valid)})
        if chosen is None and adf_pvalue <= adf_pvalue_max:
            chosen = rows[-1]

    if chosen is None:
        # Nothing in the grid achieved stationarity at the requested confidence;
        # fall back to d=1 (full integer differencing) rather than silently
        # returning a non-stationary feature.
        chosen = rows[-1]

    return {
        "d": chosen["d"],
        "adf_pvalue": chosen["adf_pvalue"],
        "corr_with_original": chosen["corr_with_original"],
        "table": rows,
    }

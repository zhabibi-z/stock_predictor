"""Stage 4c — Deflated Sharpe Ratio (DSR) and Probability of Backtest
Overfitting (PBO).

Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014) and
"The Probability of Backtest Overfitting" (2015).

Both correct for the same underlying problem from different angles: this
project's main backtest table compares 6 models, and Phase 2's ablation
table separately compares 12 configurations of NB/LogReg/MLP (balanced x
threshold). Reporting "the best one" without accounting for how many were
tried overstates how surprising that result is — some model looking best
by chance is expected once you've looked at enough of them.

Deflated Sharpe Ratio
──────────────────────
Standard Sharpe ratio estimation assumes a single trial. With n_trials
candidates evaluated before reporting one, the expected maximum Sharpe
ratio achievable by pure luck (SR0) rises with n_trials. DSR is the
probability that the observed Sharpe ratio exceeds that luck-adjusted
benchmark, using the Probabilistic Sharpe Ratio (PSR) — which itself
corrects the standard Sharpe ratio's confidence interval for skewness and
kurtosis in the return series, since financial returns are rarely normal.

Probability of Backtest Overfitting (via CSCV)
────────────────────────────────────────────────
Given a performance matrix (S blocks x N configurations), split the S
blocks into every possible bipartition. In each split, find which
configuration performs best on one half ("in-sample"), then check its
relative rank on the other half ("out-of-sample"). If the in-sample
winner tends to rank below the out-of-sample median, that is direct
evidence that "best in-sample" and "best out-of-sample" are different
things for this search — i.e. overfitting. PBO is the fraction of splits
where that happens.
"""

import itertools

import numpy as np
from scipy.stats import kurtosis, norm, skew

_EULER_GAMMA = 0.5772156649015329


def sharpe_ratio_stats(returns: np.ndarray) -> dict:
    """Non-annualized per-period Sharpe ratio, skewness, and Pearson
    kurtosis (normal distribution = 3, not excess) of a return series."""
    r = np.asarray(returns, dtype=float)
    sr = float(r.mean() / (r.std(ddof=1) + 1e-12))
    return {
        "sharpe": sr,
        "skew":     float(skew(r, bias=False)),
        "kurtosis": float(kurtosis(r, fisher=False, bias=False)),
        "T": len(r),
    }


def probabilistic_sharpe_ratio(sr_hat: float, sr_benchmark: float, T: int, skewness: float, kurt: float) -> float:
    """PSR(SR*): probability that the true Sharpe ratio exceeds sr_benchmark,
    given an observed sr_hat over T periods with the given skew/kurtosis."""
    denom = np.sqrt(max(1 - skewness * sr_hat + (kurt - 1) / 4 * sr_hat ** 2, 1e-12))
    z = (sr_hat - sr_benchmark) * np.sqrt(max(T - 1, 1)) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """E[max SR_n] over n_trials independent trials, each with the given
    Sharpe-ratio-estimator variance (the luck-adjusted benchmark, SR0)."""
    if n_trials <= 1:
        return 0.0
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(max(sr_variance, 0.0)) * ((1 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int) -> dict:
    """
    DSR = PSR(SR0), where SR0 is the expected maximum Sharpe ratio
    achievable by n_trials trials under pure luck (no real skill), and
    PSR corrects the observed Sharpe's confidence interval for the
    return series' actual skew/kurtosis rather than assuming normality.
    """
    stats = sharpe_ratio_stats(returns)
    sr_hat, T, g3, g4 = stats["sharpe"], stats["T"], stats["skew"], stats["kurtosis"]
    sr_variance = max(1 - g3 * sr_hat + (g4 - 1) / 4 * sr_hat ** 2, 1e-12) / max(T - 1, 1)
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    dsr = probabilistic_sharpe_ratio(sr_hat, sr0, T, g3, g4)
    return {**stats, "n_trials": n_trials, "sr0": sr0, "dsr": dsr}


def probability_of_backtest_overfitting(performance_matrix: np.ndarray) -> dict:
    """
    performance_matrix: (S, N) array — S blocks (folds / CPCV combinations)
    by N configurations, each cell a performance metric for that config on
    that block. Returns PBO (fraction of bipartitions where the
    in-sample-best configuration performs below the out-of-sample median)
    and the number of bipartitions evaluated.
    """
    perf = np.asarray(performance_matrix, dtype=float)
    S, N = perf.shape
    half = S // 2
    logits = []
    for is_combo in itertools.combinations(range(S), half):
        is_idx  = list(is_combo)
        oos_idx = [i for i in range(S) if i not in is_combo]
        is_perf  = perf[is_idx].mean(axis=0)
        oos_perf = perf[oos_idx].mean(axis=0)

        best_is = int(np.argmax(is_perf))
        rank = int((oos_perf < oos_perf[best_is]).sum()) + 1  # 1-indexed, worst=1
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))

    logits = np.array(logits)
    return {
        "pbo": float((logits < 0).mean()),
        "n_splits": len(logits),
        "logits_mean": float(logits.mean()),
    }

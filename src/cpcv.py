"""Stage 4b — Combinatorial Purged Cross-Validation (CPCV).

The existing purged walk-forward CV (src/backtester.purged_walk_forward_splits)
produces exactly one path through the data: each row is tested exactly once,
using only the data before it. That's the right choice for the walk-forward
summary and the holdout — it's what a real deployment would actually see.

But it also means "is 53% real or noise" rests on a single realized path.
CPCV (Lopez de Prado, "Advances in Financial Machine Learning", ch. 12)
partitions the data into N groups and evaluates every C(N, k) combination of
k held-out test groups (with purging at every boundary adjacent to a test
group), producing many train/test splits from the same data. The result is a
DISTRIBUTION of accuracy/Sharpe across combinations instead of one number —
the same idea as the walk-forward summary's mean ± std across folds, applied
combinatorially instead of sequentially.

This trades chronological realism (test groups are not required to come
strictly after their training data — some combinations train on "future"
groups relative to a test group) for statistical power: with N=6, k=2, a
single dataset yields 15 train/test splits instead of 1. It is a
complementary diagnostic, not a replacement for the walk-forward summary or
the holdout, and is never used to select a model or a hyperparameter.
"""

import itertools

import numpy as np


def _make_groups(n_samples: int, n_groups: int) -> list:
    edges = np.linspace(0, n_samples, n_groups + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def combinatorial_purged_splits(
    n_samples:     int,
    n_groups:      int,
    n_test_groups: int,
    purge_gap:     int = 5,
) -> list:
    """
    Partition range(n_samples) into n_groups contiguous, near-equal blocks,
    then return (train_idx, test_idx) for every C(n_groups, n_test_groups)
    combination of which blocks serve as the test set.

    Purging removes training rows within `purge_gap` of either boundary of
    any test block, on both sides — the training set never includes a row
    close enough to a test block for the two to be serially correlated,
    regardless of which side of the test block it falls on.
    """
    groups = _make_groups(n_samples, n_groups)
    splits = []
    for test_combo in itertools.combinations(range(n_groups), n_test_groups):
        test_ranges = [groups[i] for i in test_combo]
        test_idx = np.concatenate([np.arange(s, e) for s, e in test_ranges])

        exclude = np.zeros(n_samples, dtype=bool)
        for s, e in test_ranges:
            lo = max(s - purge_gap, 0)
            hi = min(e + purge_gap, n_samples)
            exclude[lo:hi] = True

        train_idx = np.where(~exclude)[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, np.sort(test_idx)))
    return splits


def summarize_distribution(values: list) -> dict:
    """Mean/std/min/max for a list of per-combination metric values."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "n":    len(arr),
    }

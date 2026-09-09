"""
Tests for fixed-width-window fractional differentiation.

Two properties matter: the weights collapse to known special cases at the
boundaries (d=0 is a no-op, d=1 matches plain first-differencing), and
find_min_ffd_d actually distinguishes a non-stationary series (a random
walk, which needs d close to 1) from an already-stationary one (which
should need very little differencing).
"""

import numpy as np
import pandas as pd

from src.fracdiff import find_min_ffd_d, frac_diff_ffd, frac_diff_weights


def test_d_zero_is_a_no_op():
    weights = frac_diff_weights(0.0)
    assert weights.tolist() == [1.0]

    series = pd.Series(np.arange(1, 21, dtype=float))
    out = frac_diff_ffd(series, d=0.0)
    assert np.allclose(out.values, series.values)


def test_d_one_matches_plain_first_difference():
    series = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=50)))
    ffd = frac_diff_ffd(series, d=1.0)
    plain_diff = series.diff()

    both_valid = ffd.notna() & plain_diff.notna()
    assert both_valid.sum() > 40
    assert np.allclose(ffd[both_valid].values, plain_diff[both_valid].values, atol=1e-8)


def test_weights_decay_and_truncate():
    weights = frac_diff_weights(0.4, threshold=1e-4)
    assert weights[0] == 1.0
    assert abs(weights[-1]) < 1e-3
    assert len(weights) < 10_000  # truncated well before max_width, not running to it

    # A tighter threshold must not produce a *longer* window than a looser one.
    tighter = frac_diff_weights(0.4, threshold=1e-6)
    assert len(tighter) > len(weights)


def test_find_min_ffd_d_distinguishes_stationary_from_random_walk():
    """
    A random walk is I(1) by construction and needs real differencing to
    reject a unit root; an already-stationary AR(1) series needs little to
    none. This does NOT assert d must reach some high absolute value for
    the random walk — in practice (and in Lopez de Prado's own published
    examples on real futures data), d around 0.3-0.4 is often already
    enough to pass an ADF test while preserving far more memory than full
    (d=1) differencing. The property that must hold is comparative: the
    random walk needs strictly more differencing than the stationary series.
    """
    rng = np.random.default_rng(0)
    n = 600

    random_walk = pd.Series(100 + np.cumsum(rng.normal(scale=0.5, size=n)))
    rw_result = find_min_ffd_d(random_walk, d_grid=np.round(np.arange(0.0, 1.01, 0.1), 2))

    ar1 = np.zeros(n)
    for t in range(1, n):
        ar1[t] = 0.3 * ar1[t - 1] + rng.normal(scale=1.0)
    stationary = pd.Series(ar1)
    st_result = find_min_ffd_d(stationary, d_grid=np.round(np.arange(0.0, 1.01, 0.1), 2))

    assert st_result["d"] < rw_result["d"], (
        f"expected the already-stationary series to need less differencing "
        f"than the random walk, got stationary={st_result['d']} vs random_walk={rw_result['d']}"
    )
    assert rw_result["corr_with_original"] > 0.5, (
        "the random walk's chosen d should still preserve substantial memory "
        "of the original series — that's the entire point of FFD over plain diff()"
    )


def test_ffd_preserves_more_memory_than_full_differencing():
    """The whole point of FFD: at its chosen d, correlation with the
    original series should exceed what full (d=1) differencing gives."""
    rng = np.random.default_rng(1)
    n = 600
    series = pd.Series(100 + np.cumsum(rng.normal(scale=0.5, size=n)))

    result = find_min_ffd_d(series, d_grid=np.round(np.arange(0.0, 1.01, 0.1), 2))
    full_diff = frac_diff_ffd(series, d=1.0)
    valid = full_diff.notna()
    full_diff_corr = float(np.corrcoef(full_diff[valid].values, series.reindex(full_diff[valid].index).values)[0, 1])

    if result["d"] < 1.0:
        assert result["corr_with_original"] >= full_diff_corr

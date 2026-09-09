"""
Tests for Combinatorial Purged Cross-Validation splitting.

Three invariants matter: the combination count is exactly C(n_groups,
n_test_groups); train and test never overlap in any split; and no training
row sits within purge_gap of either boundary of any test block in that split.
"""

import math

import numpy as np
import pytest

from src.cpcv import _make_groups, combinatorial_purged_splits, summarize_distribution


@pytest.mark.parametrize("n_samples", [200, 1000, 1750])
@pytest.mark.parametrize("n_groups,n_test_groups", [(6, 2), (5, 1), (8, 3)])
@pytest.mark.parametrize("purge_gap", [1, 5, 10])
def test_split_count_and_no_overlap_and_purge_gap(n_samples, n_groups, n_test_groups, purge_gap):
    splits = combinatorial_purged_splits(n_samples, n_groups, n_test_groups, purge_gap)
    assert len(splits) == math.comb(n_groups, n_test_groups)

    groups = _make_groups(n_samples, n_groups)
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(test_idx)

        # No training row within purge_gap of any test block's boundary.
        test_block_bounds = set()
        # reconstruct which groups are test groups for this split from test_idx
        test_starts = sorted({s for s, e in groups if any(s <= t < e for t in test_idx)})
        for s, e in groups:
            if s in test_starts:
                test_block_bounds.add(s)
                test_block_bounds.add(e)

        train_set = set(train_idx)
        for bound in test_block_bounds:
            near = range(max(bound - purge_gap, 0), min(bound + purge_gap, n_samples))
            # every row in the purge zone that isn't itself a test row must be excluded from train
            for row in near:
                if row not in test_idx:
                    assert row not in train_set, (
                        f"row {row} within purge_gap of test boundary {bound} leaked into train"
                    )


def test_every_row_covered_symmetrically():
    """Each group should appear as part of the test set in exactly
    C(n_groups - 1, n_test_groups - 1) combinations — a basic CPCV symmetry
    property, independent of purge_gap (set to 0 here to isolate it)."""
    n_groups, n_test_groups = 6, 2
    n_samples = 600
    splits = combinatorial_purged_splits(n_samples, n_groups, n_test_groups, purge_gap=0)

    groups = _make_groups(n_samples, n_groups)
    counts = [0] * n_groups
    for _, test_idx in splits:
        test_set = set(test_idx)
        for gi, (s, e) in enumerate(groups):
            if s in test_set:
                counts[gi] += 1

    expected = math.comb(n_groups - 1, n_test_groups - 1)
    assert all(c == expected for c in counts)


def test_summarize_distribution():
    d = summarize_distribution([0.5, 0.6, 0.7])
    assert d["n"] == 3
    assert d["min"] == 0.5
    assert d["max"] == 0.7
    assert np.isclose(d["mean"], 0.6)

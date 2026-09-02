"""
Fold-boundary tests: every walk-forward split, and the train/validation/
holdout split, must leave a real purge gap between what a model trains on
and what it's evaluated against. No boundary may be zero-gap or overlapping.
"""

import pandas as pd
import pytest

from src.backtester import purged_walk_forward_splits, three_way_split


@pytest.mark.parametrize("n_samples", [200, 1000, 2480])
@pytest.mark.parametrize("n_splits", [3, 4, 5])
@pytest.mark.parametrize("purge_gap", [1, 5, 10])
def test_purge_gap_enforced_on_every_fold(n_samples, n_splits, purge_gap):
    splits = purged_walk_forward_splits(n_samples, n_splits, purge_gap)
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert max(train_idx) + purge_gap < min(test_idx)
        # train and test never overlap, and train is a contiguous prefix
        assert set(train_idx).isdisjoint(test_idx)
        assert list(train_idx) == list(range(min(train_idx), max(train_idx) + 1))


def test_three_way_split_disjoint_and_ordered():
    dates = pd.bdate_range("2020-01-01", periods=1000)
    purge_gap = 5
    train_idx, val_idx, holdout_idx = three_way_split(
        dates, holdout_start="2023-06-01", validation_frac=0.15, purge_gap=purge_gap,
    )

    assert len(train_idx) > 0 and len(val_idx) > 0 and len(holdout_idx) > 0
    assert set(train_idx).isdisjoint(val_idx)
    assert set(val_idx).isdisjoint(holdout_idx)
    assert set(train_idx).isdisjoint(holdout_idx)

    # chronological order: train_pool < validation < holdout
    assert max(train_idx) < min(val_idx) < max(val_idx) < min(holdout_idx)

    # a real purge gap at both internal boundaries
    assert max(train_idx) + purge_gap < min(val_idx)
    assert max(val_idx) + purge_gap < min(holdout_idx)


def test_three_way_split_holdout_starts_on_or_after_holdout_start():
    dates = pd.bdate_range("2020-01-01", periods=1000)
    holdout_start = "2023-06-01"
    _, _, holdout_idx = three_way_split(dates, holdout_start=holdout_start, validation_frac=0.15, purge_gap=5)
    assert dates[holdout_idx[0]] >= pd.Timestamp(holdout_start)

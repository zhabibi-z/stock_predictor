"""
Golden-file regression test for engineer_features, plus a no-NaN/no-inf
guard on the real production feature matrix.

The golden fixture is a small, fully deterministic synthetic OHLCV series
(no randomness) run through engineer_features with small windows so a
20-row fixture leaves enough rows post-warm-up. Values were pinned by
running the current, already-fixed implementation once (see
tests/fixtures/golden_features.json) — this test exists to catch any
*future* unintended change to indicator math, including an accidental
regression of SMA_5/SMA_20 back to raw price levels.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src import load_config
from src.data_loader import load_or_download
from src.features import engineer_features

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_tiny_ohlcv(n=20):
    dates = pd.bdate_range("2021-01-04", periods=n)
    t = np.arange(n, dtype=float)
    close = 100 + 0.5 * t + 2 * np.sin(t / 2.0)
    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.2
    volume = 1_000_000 + 10_000 * t
    return pd.DataFrame(
        {"Close": close, "High": high, "Low": low, "Open": open_, "Volume": volume},
        index=dates,
    )


def test_engineer_features_matches_golden_fixture():
    with open(os.path.join(FIXTURES_DIR, "golden_features.json")) as fh:
        golden = json.load(fh)

    df = _make_tiny_ohlcv(20)
    data = engineer_features(
        df,
        sma_short=3, sma_long=5, rsi_window=5,
        bb_window=5, bb_std=2.0,
        macd_fast=3, macd_slow=6, macd_sig=2,
        atr_window=5, vol_window=5,
        lag_periods=[1, 2],
    )

    for col, expected in golden.items():
        actual = data[col].head(3).tolist()
        assert actual == pytest.approx(expected, rel=1e-9), f"{col} drifted from the golden fixture"


def test_sma_ratio_features_are_not_raw_price_levels():
    """Direct regression guard for the F3 fix: SMA_5/SMA_20 must be
    dimensionless ratios centered near 0, not raw dollar price levels."""
    df = _make_tiny_ohlcv(30)
    data = engineer_features(
        df,
        sma_short=3, sma_long=5, rsi_window=5,
        bb_window=5, bb_std=2.0,
        macd_fast=3, macd_slow=6, macd_sig=2,
        atr_window=5, vol_window=5,
        lag_periods=[1],
    )
    assert len(data) > 0, "fixture too small for these windows — test setup bug"
    assert data["SMA_5"].abs().max() < 1.0, "SMA_5 looks like a raw price level, not a Close/SMA-1 ratio"
    assert data["SMA_20"].abs().max() < 1.0, "SMA_20 looks like a raw price level, not a Close/SMA-1 ratio"


def test_no_nan_or_inf_in_production_feature_matrix():
    """The real feature matrix, built from the actual cached data and the
    actual production config, must be entirely finite."""
    cfg = load_config()
    d, f = cfg["data"], cfg["features"]
    raw_df = load_or_download(d["ticker"], d["start"], d["end"])
    data = engineer_features(
        raw_df,
        sma_short=f["sma_short"], sma_long=f["sma_long"],
        rsi_window=f["rsi_window"],
        bb_window=f["bb_window"], bb_std=f["bb_std"],
        macd_fast=f["macd_fast"], macd_slow=f["macd_slow"], macd_sig=f["macd_sig"],
        atr_window=f["atr_window"], vol_window=f["vol_window"],
        lag_periods=f["lag_periods"],
    )
    X = data[f["cols"]].values
    assert np.isfinite(X).all(), "production feature matrix contains NaN or inf"

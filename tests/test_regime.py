"""Tests for src/regime.py — volatility-regime conditional accuracy."""

import numpy as np

from src.regime import beats_always_long, holdout_realized_vol, print_regime_table, regime_report

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_daily_returns(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.015, size=n)


# ── holdout_realized_vol ──────────────────────────────────────────────────────

def test_holdout_realized_vol_shape():
    dr = _make_daily_returns(200)
    holdout_idx = np.arange(150, 200)
    rv = holdout_realized_vol(dr, holdout_idx)
    assert rv.shape == (50,)


def test_holdout_realized_vol_no_nan_after_warmup():
    """All holdout rows are past the 20-day window so no NaN expected."""
    dr = _make_daily_returns(300)
    holdout_idx = np.arange(250, 300)
    rv = holdout_realized_vol(dr, holdout_idx)
    assert not np.any(np.isnan(rv))


def test_holdout_realized_vol_warmup_nans_in_early_rows():
    """First 19 positions of the full series are NaN — not in a real holdout."""
    dr = _make_daily_returns(50)
    rv_full = holdout_realized_vol(dr, np.arange(50))
    # first window-1 = 19 entries must be NaN
    assert np.all(np.isnan(rv_full[:19]))
    assert not np.isnan(rv_full[19])


def test_holdout_realized_vol_custom_window():
    dr = _make_daily_returns(100)
    holdout_idx = np.arange(50, 100)
    rv = holdout_realized_vol(dr, holdout_idx, window=5)
    assert rv.shape == (50,)
    assert not np.any(np.isnan(rv))


# ── regime_report ─────────────────────────────────────────────────────────────

def _make_regime_inputs(n: int = 200, seed: int = 1):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    preds_perfect = y_true.copy()
    preds_random  = rng.integers(0, 2, size=n)
    rv            = np.abs(rng.normal(0.01, 0.005, size=n))
    return y_true, preds_perfect, preds_random, rv


def test_regime_report_keys():
    y_true, preds_perfect, preds_random, rv = _make_regime_inputs()
    predictions = {"perfect": preds_perfect, "random": preds_random}
    rows, edges = regime_report(y_true, predictions, rv)
    assert len(rows) == 2
    assert all(f"Q{q + 1}" in rows[0] for q in range(4))
    assert "Overall" in rows[0]
    assert len(edges) == 3  # 3 inner edges for 4 quartiles


def test_regime_report_perfect_predictor_all_ones():
    y_true, preds_perfect, _, rv = _make_regime_inputs()
    rows, _ = regime_report(y_true, {"perfect": preds_perfect}, rv)
    row = rows[0]
    for q in range(4):
        assert abs(row[f"Q{q + 1}"] - 1.0) < 1e-9, f"Q{q+1} should be 1.0 for perfect predictor"
    assert abs(row["Overall"] - 1.0) < 1e-9


def test_regime_report_n_per_quartile_sums_to_total():
    n = 200
    y_true, preds_perfect, _, rv = _make_regime_inputs(n)
    rows, _ = regime_report(y_true, {"m": preds_perfect}, rv)
    total = sum(rows[0][f"Q{q + 1}_n"] for q in range(4))
    assert total == n


def test_regime_report_nan_rv_handled():
    """NaN rv values must not cause an exception — placed in Q1."""
    n = 100
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=n)
    preds  = rng.integers(0, 2, size=n)
    rv     = np.abs(rng.normal(0.01, 0.005, size=n))
    rv[:5] = np.nan  # simulate warm-up NaNs

    rows, edges = regime_report(y_true, {"m": preds}, rv)
    assert rows[0]["_n_nan_rv"] == 5
    assert not np.isnan(rows[0]["Overall"])


def test_regime_report_quartile_count():
    y_true, preds, _, rv = _make_regime_inputs(400)
    rows, edges = regime_report(y_true, {"m": preds}, rv, n_quartiles=3)
    assert "Q3" in rows[0]
    assert "Q4" not in rows[0]
    assert len(edges) == 2


# ── beats_always_long ─────────────────────────────────────────────────────────

def test_beats_always_long_all_win():
    """Model that is perfect in every quartile beats always_long everywhere."""
    rows = [
        {"Model": "always_long", "Q1": 0.5, "Q2": 0.5, "Q3": 0.5, "Q4": 0.5, "Overall": 0.5},
        {"Model": "model_A",     "Q1": 0.9, "Q2": 0.9, "Q3": 0.9, "Q4": 0.9, "Overall": 0.9},
    ]
    result = beats_always_long(rows)
    assert result["model_A"] == ["Q1", "Q2", "Q3", "Q4"]


def test_beats_always_long_none_win():
    rows = [
        {"Model": "always_long", "Q1": 0.9, "Q2": 0.9, "Q3": 0.9, "Q4": 0.9, "Overall": 0.9},
        {"Model": "model_B",     "Q1": 0.5, "Q2": 0.5, "Q3": 0.5, "Q4": 0.5, "Overall": 0.5},
    ]
    result = beats_always_long(rows)
    assert result["model_B"] == []


def test_beats_always_long_no_always_long_row():
    rows = [{"Model": "model_X", "Q1": 0.8, "Q2": 0.6, "Q3": 0.7, "Q4": 0.5, "Overall": 0.65}]
    result = beats_always_long(rows)
    assert result == {}


def test_beats_always_long_nan_quartile():
    rows = [
        {"Model": "always_long", "Q1": 0.5,         "Q2": 0.5, "Q3": 0.5, "Q4": 0.5, "Overall": 0.5},
        {"Model": "model_C",     "Q1": float("nan"), "Q2": 0.8, "Q3": 0.4, "Q4": 0.6, "Overall": 0.6},
    ]
    result = beats_always_long(rows)
    # Q1 is NaN → skipped; Q2 and Q4 beat always_long; Q3 does not
    assert "Q1" not in result["model_C"]
    assert "Q2" in result["model_C"]
    assert "Q3" not in result["model_C"]
    assert "Q4" in result["model_C"]


# ── print_regime_table (smoke) ────────────────────────────────────────────────

def test_print_regime_table_no_crash(capsys):
    y_true, preds_p, preds_r, rv = _make_regime_inputs(200)
    predictions = {"always_long": preds_p, "random": preds_r}
    rows, edges = regime_report(y_true, predictions, rv)
    print_regime_table(rows, edges)
    out = capsys.readouterr().out
    assert "Q1" in out
    assert "Q4" in out
    assert "Overall" in out

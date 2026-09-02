# Architecture — ML Stock Direction Predictor

## Overview

A purged walk-forward evaluation harness for next-day stock direction (UP / DOWN). See the root `README.md` for what it actually found — this document describes the pipeline mechanics, not the results.

## Pipeline Stages

```
config/config.yaml          ← every tunable constant
        │
        ▼  src/data_loader.py
yfinance API  ──────────────►  data/AAPL_<start>_<end>.csv  (committed, cached after first run)
        │
        ▼  src/features.py
Feature engineering (11 indicators, zero temporal leakage):
  SMA_5, SMA_20             — Close/SMA-1 ratios (NOT raw price levels — see README)
  RSI_14                    — momentum
  BB_Pct                    — mean-reversion signal
  MACD_Hist                 — momentum crossover
  ATR_Norm                  — volatility regime
  Volume_Ratio              — participation relative to its rolling mean
  Return_t1, _t2, _t5       — lagged returns
  Target = shift(-1)        — next-day direction (NEVER used as input)
        │
        ▼  src/backtester.three_way_split
train_pool / validation / holdout — holdout (≥ 2024-03-22) touched exactly
once; validation reserved for threshold/balance tuning only
        │
        ▼  src/backtester.purged_walk_forward_splits  (on train_pool)
Purged walk-forward CV (config-driven fold count, 5-day purge gap)
        │
        ▼  src/baselines.py + src/models.py
Every fold, every model, no exceptions:
  always_long, prev_day_momentum, shuffled_label  — baselines, always reported
  Naive Bayes            — probabilistic, optional balanced sample_weight
  Logistic Regression    — direct classifier, optional balanced class_weight
  Neural Network (MLP)   — 3-layer, trained across N seeds (default 10),
                            reported as mean ± std, not a point estimate
        │
        ▼  src/tuning.py
Decision threshold selected on the validation split by balanced accuracy
(never on a CV fold, never on the holdout)
        │
        ▼  src/backtester.run_backtest
Financial evaluation, on the holdout only:
  1-bar execution lag, per-side cost model with a bps sensitivity curve
  Sharpe, max drawdown, Calmar — for the strategy AND the benchmark
        │
        ▼  src/visualization.py
plots/ directory: equity curves, drawdown chart, confusion matrices
```

## Interfaces

| Entry point | Description |
|---|---|
| `python main.py [--seeds N]` | CLI — full pipeline, the tables in the README |
| `streamlit run app.py` | Interactive dashboard (a separate, simpler pipeline — see README's Engineering section for what it does and doesn't share with `main.py`) |
| `pytest tests/ -v` | 33 tests: leakage check, fold-boundary invariants, golden-file feature regression, no-NaN/no-inf |

## Key Design Decisions

**Zero temporal leakage** — All features at row T use only data available through close of day T. The target variable (`shift(-1)`) is computed separately and never appears as a feature column.

**Purged walk-forward splits** — Standard k-fold would allow future information to bleed into training sets through autocorrelated residuals. A purge gap at each fold boundary (config-driven, default 5 days) eliminates this — enforced by `tests/test_splits.py`, not just asserted in a docstring.

**Three-way split, holdout touched once** — Tuning (thresholds, class balancing) happens against a dedicated validation window; the holdout is evaluated exactly once, after the walk-forward summary's leakage check passes.

**Config-driven** — Ticker, window sizes, fold count, MLP architecture, backtest cost model, and validation split parameters all live in `config/config.yaml`. Some small constants remain as sane function defaults (e.g. a 0.30–0.70 threshold search grid) rather than config entries — see the README's Engineering section for the actual line, not an absolute claim.

**Class balancing, both ways** — A ~54/46 UP/DOWN split is AAPL's secular drift, not class imbalance. `balanced: bool` is a parameter on every model; both settings are run and compared, not one silently chosen.

## Directory Structure

```
.
├── config/
│   └── config.yaml          # every tunable constant
├── data/                    # committed cache (AAPL_<start>_<end>.csv) + gitignored future tickers
├── docs/
│   └── ARCHITECTURE.md
├── notebooks/
│   └── 01_eda.ipynb
├── plots/                   # Auto-generated figures (gitignored)
├── src/
│   ├── data_loader.py       # Stage 1: yfinance download + CSV cache
│   ├── features.py          # Stage 2: 11 technical indicators
│   ├── baselines.py         # Stage 3b: always_long, prev_day_momentum, shuffled_label
│   ├── models.py            # Stage 3: NB, Logistic Regression, MLP definitions
│   ├── tuning.py            # Stage 3d: validation-split threshold selection
│   ├── reporting.py         # Stage 3c: CI-aware metrics tables
│   ├── backtester.py        # Stage 4: splits + lagged, costed backtest engine
│   └── visualization.py     # Stage 5: equity curves, confusion matrices
├── tests/                   # pytest suite
├── app.py                   # Streamlit dashboard
├── main.py                  # CLI pipeline
├── pyproject.toml           # ruff + pytest config
└── requirements.txt         # Pinned runtime dependencies
```

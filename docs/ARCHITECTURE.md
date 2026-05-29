# Architecture — ML Stock Direction Predictor

## Overview

A binary classification pipeline that predicts next-day stock price direction (UP / DOWN) using three ML models, validated with purged walk-forward cross-validation, and evaluated against financial backtesting metrics.

## Pipeline Stages

```
config/config.yaml          ← single source of truth for all hyperparameters
        │
        ▼  src/data_loader.py
yfinance API  ──────────────►  data/AAPL_historical.csv  (cached after first run)
        │
        ▼  src/features.py
Feature engineering (zero temporal leakage):
  SMA_5, SMA_20             — trend direction
  RSI_14                    — momentum
  BB_Pct                    — mean-reversion signal
  MACD_Hist                 — momentum crossover
  ATR_Norm                  — volatility regime
  Return_t1, _t2, _t5       — lagged returns
  Target = shift(-1)        — next-day direction (NEVER used as input)
        │
        ▼  src/backtester.py
Purged walk-forward CV (5 expanding folds, 5-day purge gap)
  Prevents serial-correlation bleed across fold boundaries
        │
        ▼  src/models.py
Three classifiers trained per fold:
  Naive Bayes               — probabilistic baseline
  Ridge (logistic)          — linear baseline with L2 regularisation
  Keras MLP                 — 2-layer neural net with early stopping
  All use balanced class weights to handle UP/DOWN imbalance
        │
        ▼  src/backtester.py  (backtest engine)
Financial evaluation:
  Sharpe ratio              — risk-adjusted return
  Maximum drawdown          — worst peak-to-trough loss
  Cumulative equity curve   — vs buy-and-hold benchmark
        │
        ▼  src/visualization.py
plots/ directory:
  equity curves, drawdown chart, confusion matrices (per model per fold)
```

## Interfaces

| Entry point | Description |
|---|---|
| `python main.py` | CLI — runs all 8 pipeline stages end-to-end |
| `streamlit run app.py` | Interactive dashboard wrapping the same pipeline |

## Key Design Decisions

**Zero temporal leakage** — All features at row T use only data available through close of day T. The target variable (`shift(-1)`) is computed separately and never appears as a feature column.

**Purged walk-forward splits** — Standard k-fold would allow future information to bleed into training sets through autocorrelated residuals. A 5-day purge gap at each fold boundary eliminates this.

**Config-driven** — All hyperparameters (ticker, window sizes, fold count, NN epochs) live in `config/config.yaml`. Source code contains zero magic numbers.

**Balanced class weights** — Markets are not 50/50 UP/DOWN. Balanced weights prevent all models from collapsing to predict the majority class.

## Directory Structure

```
.
├── config/
│   └── config.yaml         # All hyperparameters
├── data/                   # Auto-generated cache (gitignored)
├── docs/
│   └── ARCHITECTURE.md
├── notebooks/
│   └── 01_eda.ipynb
├── plots/                  # Auto-generated figures (gitignored)
├── src/
│   ├── data_loader.py      # Stage 1: yfinance download + CSV cache
│   ├── features.py         # Stage 2: 10 technical indicators
│   ├── models.py           # Stage 3: NB, Ridge, MLP definitions
│   ├── backtester.py       # Stage 4: walk-forward CV + backtest engine
│   └── visualization.py    # Stage 5: equity curves, confusion matrices
├── app.py                  # Streamlit dashboard
├── main.py                 # CLI pipeline
├── pyproject.toml          # ruff linting config
└── requirements.txt        # Pinned runtime dependencies
```

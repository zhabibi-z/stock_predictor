# ML Stock Direction & Backtest Dashboard

> Binary classification of next-day stock price direction (UP / DOWN) using three machine learning models, validated with purged walk-forward cross-validation, and evaluated against financial backtesting metrics including Sharpe ratio and maximum drawdown.

---

## What makes this project production-grade

| Practice | Implementation |
|----------|---------------|
| **Zero temporal leakage** | All features at row T use only data through close of day T; target uses `shift(-1)` and is never a model input |
| **Purged walk-forward CV** | 5 expanding-window folds with a 5-day purge gap at each boundary to prevent serial-correlation bleed |
| **Class imbalance handling** | Balanced `sample_weight` for NB and Ridge; balanced `class_weight` dict for the Keras MLP |
| **Financial evaluation** | Sharpe ratio, max drawdown, cumulative equity curve vs buy-and-hold — not just accuracy |
| **Config-driven** | All hyperparameters live in `config/config.yaml`; source code contains zero magic numbers |
| **Interactive dashboard** | Full Streamlit app wrapping the pipeline without modifying any backend logic |

---

## Directory structure

```
stock-predictor-ml/
│
├── config/
│   └── config.yaml           # All hyperparameters: ticker, windows, folds, NN settings
│
├── data/
│   └── AAPL_historical.csv   # Auto-generated on first run; subsequent runs load from cache
│
├── notebooks/
│   └── 01_eda.ipynb          # EDA: class imbalance, feature correlations, stationarity
│
├── plots/                    # Auto-generated PNGs (equity curves, drawdown, confusion matrices)
│
├── src/
│   ├── __init__.py           # load_config() — parses config/config.yaml
│   ├── data_loader.py        # Stage 1: Data Ingestion — yfinance download + CSV cache
│   ├── features.py           # Stage 2: Feature Engineering — 10 indicators, zero leakage
│   ├── models.py             # Stage 3: NB, Ridge, MLP definitions + classification metrics
│   ├── backtester.py         # Stage 4: Walk-forward splits + vectorized backtest engine
│   └── visualization.py      # Stage 5: Equity curves, confusion matrices, drawdown plots
│
├── app.py                    # Streamlit dashboard (run with: streamlit run app.py)
├── main.py                   # CLI pipeline, all 8 steps end-to-end
└── requirements.txt
```

---

## Setup

```bash
git clone <repo-url>
cd stock-predictor-ml

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Running

**CLI pipeline** (trains all models, prints metrics, saves plots):
```bash
python main.py
```

**Interactive Streamlit dashboard**:
```bash
streamlit run app.py
```

**EDA notebook**:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## Configuration

All parameters are controlled from `config/config.yaml` — no hardcoded values anywhere in `src/`:

```yaml
data:
  ticker: AAPL
  start: "2015-11-11"
  end: "2025-11-11"

features:
  sma_short: 5
  sma_long: 20
  rsi_window: 14
  ...

validation:
  n_splits: 5
  purge_gap: 5
```

To run on a different stock, change `ticker`, `start`, and `end` in `config.yaml` — no code changes required.

---

## Benchmarked Model Results

> **Ticker**: AAPL | **Config range**: 2015-11-11 → 2025-11-11 | **Rows after warm-up**: 2,481
> **Validation**: 5-fold purged walk-forward | **Reported**: final out-of-sample fold

### Classification Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:--------:|:---------:|:------:|:--------:|
| Naive Bayes | 45.99 % | 48.39 % | 13.64 % | 21.28 % |
| Ridge Regression | 50.12 % | 53.30 % | 55.00 % | 54.14 % |
| Neural Network (MLP) | **52.80 %** | 53.44 % | **91.82 %** | **67.56 %** |

### Financial Backtest Metrics (long-only strategy vs buy-and-hold)

| Model | Strategy Return | Benchmark (B&H) | Sharpe Ratio | Max Drawdown | Win Rate |
|:------|:--------------:|:---------------:|:------------:|:------------:|:--------:|
| Naive Bayes | -1.45 % | 33.03 % | -0.5553 | -7.51 % | 48.39 % |
| Ridge Regression | -2.94 % | 33.03 % | -0.1171 | -29.74 % | 53.30 % |
| Neural Network (MLP) | **30.33 %** | 33.03 % | **+0.5799** | -35.04 % | 53.44 % |

**Reading the results**: The MLP strategy nearly matches buy-and-hold return (30.33% vs 33.03%) while trading on only 378 of 411 test days — the expected ceiling for a direction-only long strategy on a secular bull-market stock. Naive Bayes is extremely conservative (only 62 trades, very low recall) — a known limitation of GaussianNB on correlated financial features.

---

## Engineering Design Choices

### 1. Zero Temporal Leakage

The most common mistake in financial ML is lookahead bias. This project enforces a strict contract:

```
Feature[T]    = f(OHLCV[0 .. T])               # strictly historical
Target[T]     = 1{Close[T+1] > Close[T]}        # forward-looking label, never a model input
Fwd_Return[T] = (Close[T+1] - Close[T]) / Close[T]  # backtester only, not a feature
```

The `StandardScaler` is fit exclusively on the training fold and only transformed on the test fold.

### 2. Purged Walk-Forward Validation

Standard `TimeSeriesSplit` leaves no gap between training and test windows. Financial returns exhibit short-term autocorrelation, so the last training row and first test row are statistically dependent. A **5-day purge gap** removes this dependency:

```
Train [0 → N]  |  purge (5 days)  |  Test [N+6 → N+6+fold_size]
```

### 3. Class Imbalance Handling

AAPL has an upward bias over 10 years (~54% UP days). Without correction, models collapse to always predicting UP.

- **GaussianNB**: uses `compute_sample_weight("balanced")` in `fit()`
- **Ridge Regression**: regression target weighted by direction-label balance
- **Keras MLP**: `class_weight` dict passed directly to `model.fit()`

### 4. Feature Engineering — Stationarity First

Raw prices are non-stationary. Every feature is a stationary transformation:

| Feature | Transformation | Property |
|---------|---------------|----------|
| `Daily_Return` | `(Close[T] - Close[T-1]) / Close[T-1]` | ~i.i.d., mean ≈ 0 |
| `RSI` | Wilder EMA of gains / losses | Bounded [0, 100] |
| `BB_Pct` | `(Close - Lower) / (Upper - Lower)` | Bounded [0, 1] |
| `ATR_Norm` | `ATR(14) / Close[T]` | Dimensionless volatility ratio |
| `MACD_Hist` | `EMA(12) - EMA(26) - Signal(9)` | Mean-reverting around 0 |
| `Return_t1/2/5` | `Daily_Return[T-N]` | Stationary by inheritance |

### 5. Neural Network Architecture

```
Input(10)
  Dense(128, ReLU) → BatchNormalization → Dropout(0.30)
  Dense(64,  ReLU) → BatchNormalization → Dropout(0.20)
  Dense(32,  ReLU) → Dropout(0.15)
  Dense(1,   Sigmoid)  →  P(UP)
```

- **BatchNormalization** stabilises training on correlated financial features
- **EarlyStopping** (patience=15) + **ReduceLROnPlateau** (halves LR after 7 stagnant epochs) prevents overfitting

---

## Dependencies

```
yfinance        ≥ 0.2.40    # market data
scikit-learn    ≥ 1.4.0     # NB, Ridge, metrics, preprocessing
tensorflow      ≥ 2.15.0    # Keras MLP
pandas          ≥ 2.1.0
numpy           ≥ 1.26.0
matplotlib      ≥ 3.8.0
seaborn         ≥ 0.13.0
streamlit       ≥ 1.35.0    # dashboard
pyyaml          ≥ 6.0       # config parsing
```

---

## License

MIT

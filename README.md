# ML Stock Direction & Backtest Harness

> **The headline finding: on a genuinely untouched holdout, no trained model beats always being long.** This repository is a rigorous purged walk-forward evaluation harness for next-day stock direction prediction — and the evidence it produces says that next-day direction on a single equity (AAPL) is not predictable from these technical indicators. Naive Bayes, Logistic Regression, and a tuned MLP all score below a model that does nothing but predict UP, every day, unconditionally. That absence of signal is the result. The harness that proves it — baselines in every table, confidence intervals on every figure, a holdout touched exactly once — is the actual deliverable.

An earlier version of this README reported a headline MLP accuracy of 52.80% "nearly matching buy-and-hold," with no baseline row in any table and a single seed on a single fold standing in for a result. An independent review found that number was an artifact of a missing baseline (always-long scores *higher*, at 56.83%), and a four-phase repair (see `git log`) rebuilt the evaluation from the ground up. Everything below is regenerated from `python main.py` against the committed data — every number in this document is reproducible with a command shown next to it.

---

## Reproduce every number in this document

```bash
pip install -r requirements.txt
python main.py --seeds 10          # full run — the tables below
pytest tests/ -v                   # 33 tests, including the leakage check
ruff check src/ app.py main.py     # lint — clean
```

`--seeds 10` trains the MLP 10 times per fold and once more for the holdout (~50-60 fits total, a few minutes on CPU). Use `--seeds 2` for a fast sanity check of the pipeline's mechanics — the point estimates will differ (small-sample MLP variance is part of the finding) but every deterministic model's numbers (`always_long`, `prev_day_momentum`, `shuffled_label`, `Naive Bayes`, `Logistic Regression`) reproduce **exactly**, seed-for-seed, run-for-run.

---

## What changed, and why

| # | Problem found | Fix |
|---|---|---|
| 1 | No baseline row anywhere — "52.80%" looked good only because nothing else was in the table | `src/baselines.py`: `always_long`, `prev_day_momentum`, `shuffled_label`, in every table, always |
| 2 | Ridge regressed the *price*, then thresholded — R²=0.959 vs. 0.973 for a naive "tomorrow=today" forecast; it was a laundered random walk | Deleted. Replaced by `LogisticRegression`, a direct classifier |
| 3 | `SMA_5`/`SMA_20` were raw price levels on a stock that ran \$26→\$270 — zero train/test overlap after scaling, on every fold | Replaced with `Close/SMA − 1` ratios; overlap is asserted in `main.py`, not just claimed |
| 4 | MLP trained on the final fold only; single seed, std reported as 0 | Trains on every fold; `--seeds N` (default 10) reports mean ± std, with an explicit warning whenever seed variance exceeds the model's edge over `always_long` |
| 5 | Reported benchmark return (33.03%) didn't match the committed data (57.67%) | Every backtest number is generated from the committed cache (`data/AAPL_<start>_<end>.csv`) in the same run that prints it |
| 6 | Hyperparameters (`alpha`, `threshold=0.50`, MLP architecture) were picked while looking at the fold reported as out-of-sample | Three-way split: `train_pool` / `validation` / `holdout`. Tuning happens on `validation` only; `holdout` (2024-03-22 onward) is touched exactly once, at the end of the run |
| 7 | Backtest traded on the same close the signal was computed from; no cost model; benchmark had no Sharpe/drawdown to compare against; `n_trades` counted long-*days*, not trades | 1-bar execution lag, per-side cost model with a 0/1/5/10bps sensitivity curve, benchmark Sharpe/drawdown/Calmar in the same table, `days_long` vs. real `n_round_trips` |
| 8 | `tf.random.set_seed` alone doesn't reproduce a Keras run; TensorFlow imported eagerly; no tests; magic numbers scattered through `src/` | `keras.utils.set_random_seed` + `enable_op_determinism` (verified bit-identical across reruns); lazy TF import; `pytest` suite in CI; constants moved to `config.yaml` |

---

## Results

### Walk-Forward Summary — 4 pre-holdout folds, every model, mean ± std

```
+---------------------------------+---------------+---------------+----------------+---------------+
| Model                           | Accuracy      | Precision     | Recall         | F1-Score      |
+---------------------------------+---------------+---------------+----------------+---------------+
| always_long (folds=4)           | 53.04% ± 1.72 | 53.04% ± 1.72 | 100.00% ± 0.00 | 69.30% ± 1.48 |
| prev_day_momentum (folds=4)     | 48.82% ± 3.00 | 51.73% ± 2.59 | 51.74% ± 2.76  | 51.73% ± 2.67 |
| shuffled_label (folds=4)        | 50.90% ± 1.94 | 52.62% ± 1.60 | 74.89% ± 11.46 | 61.45% ± 4.06 |
| Naive Bayes (folds=4)           | 53.39% ± 3.09 | 54.37% ± 2.98 | 72.88% ± 13.20 | 61.88% ± 5.65 |
| Logistic Regression (folds=4)   | 48.89% ± 2.12 | 53.00% ± 3.21 | 27.19% ± 6.15  | 35.73% ± 6.07 |
| Neural Network (MLP) (folds=4)  | 48.81% ± 0.50 | 52.24% ± 1.49 | 38.05% ± 3.63  | 41.87% ± 2.21 |
+---------------------------------+---------------+---------------+----------------+---------------+
```

`shuffled_label` — a classifier trained on a random permutation of the training labels — never beats `always_long`'s CI on any fold. That's the leakage check passing: if it had, the pipeline halts before touching the holdout and reports it as a bug, not a result.

### Holdout Report — touched exactly once, at 2024-03-22

```
+----------------------+---------------------+---------------------+------------------------+---------------------+
| Model                | Accuracy            | Precision           | Recall                 | F1-Score            |
+----------------------+---------------------+---------------------+------------------------+---------------------+
| always_long          | 56.83% [52.0, 61.5] | 56.83% [52.2, 62.0] | 100.00% [100.0, 100.0] | 72.47% [68.6, 76.1] |
| shuffled_label       | 55.61% [50.8, 60.3] | 58.20% [52.6, 63.8] | 77.68% [72.3, 82.9]    | 66.54% [61.6, 71.2] |
| prev_day_momentum    | 52.20% [47.4, 57.0] | 57.94% [51.5, 64.1] | 57.94% [51.4, 64.3]    | 57.94% [52.2, 63.1] |
| Naive Bayes          | 53.90% [49.1, 58.7] | 56.67% [51.6, 62.1] | 80.26% [75.1, 85.2]    | 66.43% [61.8, 70.9] |
| Logistic Regression  | 44.39% [39.7, 49.2] | 60.87% [40.0, 80.0] |  6.01% [3.1, 9.3]      | 10.94% [5.8, 16.6]  |
+----------------------+---------------------+---------------------+------------------------+---------------------+
| Neural Network (MLP), mean ± std across 10 seeds:  48.39% ± 4.76                                                |
+-------------------------------------------------------------------------------------------------------------------+
```

**Every trained model loses to `always_long` on the holdout.** Logistic Regression's collapse (44.39%, 6% recall) is not a bug — it's a real finding: its decision threshold, selected on the validation split by balanced accuracy (never by peeking at the holdout), simply doesn't generalize to this specific holdout window. See "What threshold tuning revealed" below.

### Backtest — holdout only, 1-bar execution lag, 5bps/side cost

```
+----------------------+-----------+-----------+-----------+-----------+
| Model                | Return    | Sharpe    | Max DD    | Calmar    |
+----------------------+-----------+-----------+-----------+-----------+
| always_long          |   58.93%  |   0.974   |  -33.36%  |   0.988   |
| Benchmark (buy&hold) |   57.67%  |   0.957   |  -33.36%  |   0.968   |
| shuffled_label       |   23.73%  |   0.501   |  -28.10%  |   0.498   |
| prev_day_momentum    |   27.52%  |   0.643   |  -16.97%  |   0.950   |
| Naive Bayes          |   17.41%  |   0.367   |  -37.16%  |   0.279   |
| Logistic Regression  |   10.02%  |   0.199   |   -9.55%  |   0.633   |
| MLP (seed 42)        |    3.12%  |  -0.327   |   -8.29%  |   0.230   |
+----------------------+-----------+-----------+-----------+-----------+
```

Every trained model underperforms simple buy-and-hold on every risk-adjusted metric. `prev_day_momentum`'s apparent edge is the most cost-fragile: its Sharpe roughly halves (0.94 → 0.35) between 0 and 10bps/side because it round-trips 98 times; `python main.py` prints the full 0/1/5/10bps sensitivity curve for every model.

---

## What threshold tuning revealed

Phase 2 tunes each classifier's decision threshold on the validation split (2022-12-19 → 2024-03-14) — never on a CV test fold, never on the holdout. The first attempt selected by raw accuracy and immediately broke: it drove Logistic Regression and the MLP to ~100% recall (predict UP almost unconditionally), because on a mildly imbalanced split with little real signal, that degenerate point *is* the accuracy-maximizing threshold. Switching the selection criterion to **balanced accuracy** fixed that specific collapse — but the resulting threshold (0.58 for Logistic Regression) still doesn't transfer to the holdout: recall there collapses to 6%, a pattern already visible across every walk-forward fold, not just the holdout. A 310-row validation window is not enough to select a threshold that generalizes here. This wasn't caused by looking at the holdout — the selection was blind to it, exactly as required — it's a real finding about the limits of small-sample threshold tuning on this dataset, not a bug in the harness. The full ablation (`balanced` × `threshold`, 12 rows) is in `main.py`'s Step 7 output and shows the naive `threshold=0.50` default actually *beating* the tuned threshold for two of three models on this holdout.

---

## Methodology

```
                    config/config.yaml
                            │
                            ▼
              src/data_loader.py  (cached CSV, no network after first run)
                            │
                            ▼
              src/features.py  (11 indicators, zero temporal leakage;
                                 SMA_5/SMA_20 are Close/SMA-1 ratios)
                            │
                            ▼
              src/backtester.three_way_split
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   train_pool          validation            holdout
 (purged walk-forward   (threshold/balance    (≥ 2024-03-22,
  CV folds drawn         tuning only)          touched once,
  from here)                                   at the end)
```

- **Zero temporal leakage.** `Feature[T] = f(OHLCV[0..T])`; `Target[T] = 1{Close[T+1] > Close[T]}`, never a model input.
- **Purged walk-forward CV.** Expanding-window folds with a purge gap (default 5 days) between train and test — enforced by `tests/test_splits.py` across a grid of fold counts and gap sizes, not just claimed.
- **Baselines in every table.** `always_long` (the null hypothesis for a long-only strategy with positive drift), `prev_day_momentum` (yesterday's realized move continues), and `shuffled_label` (a real classifier trained on permuted labels — the leakage check).
- **Confidence intervals on every figure.** Wilson interval for accuracy (exact for a binomial proportion); paired bootstrap for precision/recall/F1. The MLP's stochastic training is reported as mean ± std across seeds instead — with an explicit warning whenever that std exceeds the model's edge over `always_long`.
- **Class balancing, both ways.** A ~54/46 UP/DOWN split is AAPL's secular drift, not class imbalance — forcing balanced weights bets against that drift. `balanced: bool` is a parameter on every model, not a hardcoded default; both settings are run and reported (`main.py` Step 7's ablation table).
- **The holdout is touched exactly once.** 410 rows from 2024-03-22 onward, evaluated in Step 7 of `main.py`, after the walk-forward summary and its leakage check both pass. A detected leak halts the run before the holdout is ever touched.

---

## Engineering

```bash
pytest tests/ -v                      # 33 tests
ruff check src/ app.py main.py        # lint
streamlit run app.py                  # interactive dashboard
```

- **Tests** (`tests/`): a shuffled-label leakage test on perfectly-separable synthetic data; a fold-boundary test asserting `max(train) + purge_gap < min(test)` across a grid of sample counts, fold counts, and purge gaps, for both `purged_walk_forward_splits` and `three_way_split`; a golden-file regression fixture pinning indicator values against the current (fixed) implementation, plus a direct regression guard that `SMA_5`/`SMA_20` stay ratios; a no-NaN/no-inf assertion on the real production feature matrix. Wired into `.github/workflows/ci.yml` as a `test` job next to `lint`.
- **Determinism.** `_build_mlp` calls `keras.utils.set_random_seed()` + `tf.config.experimental.enable_op_determinism()`, not `tf.random.set_seed()` alone — verified by running the identical pipeline twice and diffing (bit-identical outside TensorFlow's own timestamped log lines).
- **Config-driven.** MLP architecture (layer sizes, dropout rates, learning rate, LR-plateau factor/patience), backtest parameters (execution lag, cost model, trading days per year), and validation split parameters (holdout start, validation fraction, purge gap) all live in `config/config.yaml`. Nothing here is asserted to be "zero magic numbers" — some small constants (e.g. a 0.30–0.70 threshold search grid, a `min_lr` floor) remain as sane function defaults, documented where they appear.
- **Lazy TensorFlow.** `src/models.py` imports TensorFlow inside the MLP functions only; `import src.models` alone does not pull in the ~600MB dependency, and Naive Bayes / Logistic Regression work without it installed.
- **Lint is clean.** `ruff check src/ app.py main.py` reports zero findings. `main.py`, `app.py`, and `src/models.py` share an aligned-column import style rather than `ruff --fix`'s reformatting; `pyproject.toml`'s per-file-ignores name each one and why.
- **`app.py`.** `_run_pipeline` is `@st.cache_data`-keyed on `(ticker, start, end)`, so re-running with unchanged inputs returns the stored result instead of retraining. The dashboard's own pipeline still has Phase 1's `is_final` gate and no baseline rows — it wraps an earlier, simpler version of the CLI pipeline and hasn't yet had the same audit applied to it.

---

## Directory structure

```
stock_predictor/
├── config/config.yaml         # every tunable constant
├── data/AAPL_2015-11-11_2025-11-11.csv   # committed cache; no network needed to reproduce
├── docs/ARCHITECTURE.md
├── notebooks/01_eda.ipynb
├── src/
│   ├── data_loader.py         # cached OHLCV download
│   ├── features.py            # 11 indicators, zero leakage
│   ├── baselines.py           # always_long, prev_day_momentum, shuffled_label
│   ├── models.py              # NB, Logistic Regression, MLP
│   ├── tuning.py              # validation-split threshold selection
│   ├── reporting.py           # CI-aware metrics tables
│   ├── backtester.py          # splits + lagged, costed backtest engine
│   └── visualization.py       # equity curves, drawdown, confusion matrices
├── tests/                     # pytest suite (33 tests)
├── app.py                     # Streamlit dashboard
├── main.py                    # CLI pipeline — the tables in this README
└── requirements.txt
```

## Setup

```bash
git clone <repo-url> && cd stock-predictor-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --seeds 10
```

To run on a different ticker or date range, edit `config/config.yaml` — no code changes required. On this machine, `python main.py` deadlocked under TensorFlow's default multi-threaded op execution (a known TF/macOS Eigen-threadpool issue, unrelated to this repository's own code); if that happens, force single-threaded ops before importing TensorFlow, or run with `TF_NUM_INTEROP_THREADS=1` set at the OS level (Python-side `tf.config.threading` calls must run before any TF op executes).

## License

MIT

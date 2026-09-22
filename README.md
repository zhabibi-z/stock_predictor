# ML Stock Direction & Backtest Harness

> **The headline finding: on a genuinely untouched holdout, no trained model beats always being long.** This repository is a rigorous purged walk-forward evaluation harness for next-day stock direction prediction — and the evidence it produces says that next-day direction on a single equity (AAPL) is not predictable from these technical indicators. Naive Bayes, Logistic Regression, and a tuned MLP all score below a model that does nothing but predict UP, every day, unconditionally. That absence of signal is the result. The harness that proves it — baselines in every table, confidence intervals on every figure, a holdout touched exactly once — is the actual deliverable.

An earlier version of this README reported a headline MLP accuracy of 52.80% "nearly matching buy-and-hold," with no baseline row in any table and a single seed on a single fold standing in for a result. An independent review found that number was an artifact of a missing baseline (always-long scores *higher*, at 56.83%), and a four-phase repair (see `git log`) rebuilt the evaluation from the ground up. Everything below is regenerated from `python main.py` against the committed data — every number in this document is reproducible with a command shown next to it.

---

## Reproduce every number in this document

```bash
pip install -r requirements.txt
python main.py --seeds 10          # full run — the tables below
pytest tests/ -v                    # 86 tests, including the leakage check
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

## Tier 2 — statistical rigor beyond the original audit

Everything above was the original independent-review repair. Three further additions push the same "is this real or luck" question harder, using standard quantitative-finance tools (Lopez de Prado, *Advances in Financial Machine Learning* and the Deflated Sharpe Ratio papers) rather than inventing new ones. None of them were needed to reach the headline finding — they make the case for it considerably harder to argue with.

### Combinatorial Purged CV — is 53% real, or one lucky split?

The walk-forward summary above is one path through `train_pool`: each row tested exactly once. `src/cpcv.py` partitions the same `train_pool` into 6 groups and evaluates all `C(6,2)=15` combinations of 2 held-out test groups (purging training rows within the purge gap of any test boundary), giving a *distribution* instead of one number:

```
+---------------------------------+---------------+------------------+
| Model                           | Mean ± Std    | Range            |
+---------------------------------+---------------+------------------+
| always_long (15 combos)         | 53.26% ± 1.50 | [50.68%, 55.92%] |
| Naive Bayes (15 combos)         | 52.93% ± 1.53 | [49.66%, 55.50%] |
| Logistic Regression (15 combos) | 47.56% ± 1.97 | [43.74%, 50.68%] |
+---------------------------------+---------------+------------------+
```

Logistic Regression's entire 15-combination range sits at or below `always_long`'s entire range — this isn't one unlucky walk-forward fold, it holds across 15 independently-purged resamples of the same data. MLP is omitted (15x more MLP fits per run); `main.py` prints exactly why.

### Deflated Sharpe Ratio — even the winner's Sharpe isn't real

The backtest table's Sharpe ratios are single point estimates. `src/dsr.py` implements the Deflated Sharpe Ratio (DSR): the probability that an observed Sharpe exceeds what pure luck would produce, given how many candidates were actually compared (more candidates → higher bar). Run twice — once counting only the 6-model headline table, once honestly counting Step 7b's full 12-config balance × threshold ablation:

```
+----------------------+---------------+----------------+
| Model                | DSR (n=6)     | DSR (n=15)     |
+----------------------+---------------+----------------+
| always_long          | 0.559         | 0.374          |
| prev_day_momentum    | 0.401         | 0.235          |
| shuffled_label       | 0.340         | 0.188          |
| Naive Bayes          | 0.267         | 0.137          |
| Logistic Regression  | 0.284         | 0.149          |
| Neural Network (MLP) | 0.189         | 0.088          |
+----------------------+---------------+----------------+
```

**Every model's DSR is well below any normal significance bar (typically ~0.95), including `always_long` — the outright winner.** Its observed Sharpe (0.974) is not statistically distinguishable from luck once the number of models actually compared is honestly counted, and drops further once the ablation search counts too. This is a stronger statement than "it loses on average": even the best-performing strategy here doesn't clear the bar for a real result.

### Probability of Backtest Overfitting — was the ablation table itself overfit?

Phase 2's balance × threshold ablation tried 12 configurations before Phase 2 picked one canonical path. `src/dsr.py`'s `probability_of_backtest_overfitting` (CSCV) checks whether picking "the best-looking config" from a search like that is usually overfitting: split the CPCV combinations into every possible in-sample/out-of-sample bipartition, and see whether the in-sample winner tends to rank below the out-of-sample median.

```
PBO = 0.003, over 6435 bipartitions, on the NB/LogReg half of that ablation (8 of its 12 configs)
```

PBO this low means the in-sample-best of those 8 configs is *not* an overfitting artifact — its relative ranking is stable across resamples. That is a narrower claim than "the model has skill" (DSR above already answers that question, negatively): it means the balance/threshold choice's effect on ranking is a structural property of the modeling choice, not resampling noise.

### Fractional differentiation — the general fix for the SMA_5/SMA_20 problem

Phase 2 fixed `SMA_5`/`SMA_20`'s non-stationarity with a hand-picked `Close/SMA − 1` ratio. `src/fracdiff.py` implements Lopez de Prado's Fixed-Width Window Fractional Differentiation: the *general* version of that fix — grid-search the minimum differencing order d that passes an Augmented Dickey-Fuller stationarity test, rather than hand-picking a transform that happens to work. Run on `train_pool`'s actual `Close` series (`main.py` Step 3b, train-only — same rule as every other tuned parameter here):

```
d = 0.25 passes ADF (p = 0.0155) at 0.940 correlation with the original price.
Full differencing (d = 1.00, what Daily_Return already amounts to) passes ADF trivially
(p ≈ 0) but preserves almost none of it (0.023 correlation).
```

This is a real, measured version of the stationarity-vs-memory tradeoff the technique exists to improve — not a synthetic example. It's reported as a diagnostic only, not wired into `FEATURE_COLS`: at this d the FFD window is ~445 rows wide, which would eat deeply into the three-way split's warm-up budget and requires reconciling against split boundaries every other phase depends on. A genuine follow-up, left undone rather than rushed.

---

## Tier 3 — volatility-regime conditional accuracy

All the evidence above is unconditional: no model beats `always_long` on average, across the full holdout. A natural follow-up is whether any model shows **conditional** skill in a specific market regime — even if it loses on average, does it have edge when volatility is elevated (or suppressed)?

`src/regime.py` splits the holdout into four equal-sized groups by 20-day realized volatility (rolling std of daily returns, computed on the full series and extracted at holdout dates). Regimes are defined purely from price data — no model information used to choose them. `python main.py` runs this as Step 8b, immediately after all holdout predictions are in hand.

```
Volatility-Regime Conditional Accuracy — holdout, 4 quartiles by 20-day realized vol
+------------------------------------------+-------------+-------------+-------------+-------------+----------+
|                  Model                   | Q1 (Low vol)|      Q2     |      Q3     |Q4 (High vol)| Overall  |
+------------------------------------------+-------------+-------------+-------------+-------------+----------+
| always_long                              |    56.31%   |    56.86%   |    55.88%   |    58.25%   |  56.83%  |
| prev_day_momentum                        |    50.49%   |    44.12%   |    56.86%   |    57.28%   |  52.20%  |
| Naive Bayes                              |    56.31%   |    56.86%   |    50.00%   |    52.43%   |  53.90%  |
| Logistic Regression                      |    43.69%   |    47.06%   |    49.02%   |    37.86%   |  44.39%  |
+------------------------------------------+-------------+-------------+-------------+-------------+----------+
N per quartile: 103 | 102 | 102 | 103
Vol quartile edges (20-day realized vol, annualized): 1.285%, 1.551%, 1.860%
MLP omitted — TF/macOS Eigen-threadpool deadlock on this machine; NB/LogReg/baselines are deterministic.
```

**Findings:**

- **No model beats `always_long` consistently across regimes.** `prev_day_momentum` edges it in Q3 (56.86% vs 55.88%), but that single-quartile result has no statistical significance on 102 rows.
- **Naive Bayes matches `always_long` exactly in Q1 and Q2** (56.31% / 56.86%). At its tuned threshold (0.43), NB predicts UP on essentially every low-volatility day — its accuracy in calm regimes is `always_long`'s accuracy by construction.
- **Logistic Regression collapses in high-vol regimes** (37.86% in Q4 — *worse* than random). Its tuned threshold (0.58) turns it into a de facto bear-day predictor; when high-vol periods are up-heavy (Q4: 58.25% UP days), that directional bet backfires badly.
- **The `always_long` baseline is stable** (55.88%–58.25% across all four quartiles, tightest range of any model). This is what a null hypothesis looks like: uniform drift across volatility regimes.

The regime analysis makes the Tier 1/Tier 2 finding harder to argue with: the absence of skill is not concentrated in one volatility state, and it does not disappear in the high-vol regime where mean-reversion or momentum effects are most often claimed.

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
pytest tests/ -v                      # 74 tests
ruff check src/ app.py main.py        # lint
streamlit run app.py                  # interactive dashboard
```

- **Tests** (`tests/`): a shuffled-label leakage test on perfectly-separable synthetic data; a fold-boundary test asserting `max(train) + purge_gap < min(test)` across a grid of sample counts, fold counts, and purge gaps, for both `purged_walk_forward_splits` and `three_way_split`; a golden-file regression fixture pinning indicator values against the current (fixed) implementation, plus a direct regression guard that `SMA_5`/`SMA_20` stay ratios; a no-NaN/no-inf assertion on the real production feature matrix. Wired into `.github/workflows/ci.yml` as a `test` job next to `lint`.
- **Determinism.** `_build_mlp` calls `keras.utils.set_random_seed()` + `tf.config.experimental.enable_op_determinism()`, not `tf.random.set_seed()` alone — verified by running the identical pipeline twice and diffing (bit-identical outside TensorFlow's own timestamped log lines).
- **Config-driven.** MLP architecture (layer sizes, dropout rates, learning rate, LR-plateau factor/patience), backtest parameters (execution lag, cost model, trading days per year), and validation split parameters (holdout start, validation fraction, purge gap) all live in `config/config.yaml`. Nothing here is asserted to be "zero magic numbers" — some small constants (e.g. a 0.30–0.70 threshold search grid, a `min_lr` floor) remain as sane function defaults, documented where they appear.
- **Lazy TensorFlow.** `src/models.py` imports TensorFlow inside the MLP functions only; `import src.models` alone does not pull in the ~600MB dependency, and Naive Bayes / Logistic Regression work without it installed.
- **Lint is clean.** `ruff check src/ app.py main.py` reports zero findings. `main.py`, `app.py`, and `src/models.py` share an aligned-column import style rather than `ruff --fix`'s reformatting; `pyproject.toml`'s per-file-ignores name each one and why.
- **`app.py`.** `_run_pipeline` is `@st.cache_data`-keyed on `(ticker, start, end)`, so re-running with unchanged inputs returns the stored result instead of retraining. It now shares the CLI's honesty baseline — `always_long`/`prev_day_momentum`/`shuffled_label` in every table, a leak-check warning banner, Wilson/bootstrap CIs on the final-fold report, and a multi-seed MLP (`neural_network.app_seeds`, default 3 — smaller than the CLI's 10, trading rigor for interactive speed; a caption says so and points to `python main.py --seeds 10`). It still doesn't attempt the CLI's three-way holdout split, validation-tuned thresholds, CPCV, or DSR/PBO — those require real wall-clock time no interactive dashboard should ask a user to wait through.
- **A headless smoke test** (`tests/test_app_smoke.py`, via Streamlit's own `AppTest`) actually loads `app.py` and clicks "Run Quant Pipeline" — the test that would have caught a real bug that shipped invisibly for the app's entire history: `st.image(..., use_container_width=True)` doesn't exist on `streamlit==1.35.0` (the pin in `requirements.txt`), only on a newer version that had drifted into local dev environments. Nothing had ever launched the app against the actually-pinned dependency until this test did.
- **Pinned-dependency verification is real, not assumed.** Local development on this repo drifted significantly from `requirements.txt` (newer numpy/pandas/scikit-learn/TensorFlow/Streamlit than pinned) — which is *how* the `st.image` bug above went unnoticed. Every Tier 1/Tier 2 change was re-verified end-to-end (full test suite, and for Tier 1, the full CLI pipeline) in a fresh venv built from the exact pinned `requirements.txt`, not just locally. It caught real, otherwise-invisible bugs twice: the `st.image` crash, and a newer `statsmodels` keyword (`result_object=True`) that doesn't exist in the pinned `statsmodels==0.14.2`.
- **Two new pinned dependencies, both justified and verified.** `scipy==1.13.1` (already an indirect dependency of scikit-learn; used directly now for the normal distribution's CDF/PPF in the Deflated Sharpe Ratio) and `statsmodels==0.14.2` (the Augmented Dickey-Fuller test for fractional-differentiation order selection — the standard tool for this, not reimplemented by hand).

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
│   ├── fracdiff.py            # fractional differentiation (stationarity, Tier 2)
│   ├── baselines.py           # always_long, prev_day_momentum, shuffled_label
│   ├── models.py              # NB, Logistic Regression, MLP
│   ├── tuning.py              # validation-split threshold selection
│   ├── reporting.py           # CI-aware metrics tables
│   ├── backtester.py          # splits + lagged, costed backtest engine
│   ├── cpcv.py                # Combinatorial Purged CV (Tier 2)
│   ├── dsr.py                 # Deflated Sharpe Ratio + PBO (Tier 2)
│   └── visualization.py       # equity curves, drawdown, confusion matrices
├── tests/                     # pytest suite (74 tests)
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

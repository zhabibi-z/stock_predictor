"""
app.py — Streamlit dashboard for the ML Stock Direction Predictor.

Wraps the src/ pipeline without modifying any underlying module logic.
All heavy computation runs in the main thread; intermediate state is
persisted via st.session_state.

Run with:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="ML Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src                import load_config
from src.data_loader    import load_or_download
from src.features       import engineer_features
from src.models         import (
    train_naive_bayes, predict_naive_bayes,
    train_logistic, predict_logistic,
    train_mlp, predict_mlp,
    compute_metrics, aggregate_walk_forward,
)
from src.baselines      import (
    train_always_long, predict_always_long,
    train_prev_day_momentum, predict_prev_day_momentum,
    train_shuffled_label, predict_shuffled_label,
)
from src.reporting      import compute_metrics_ci, aggregate_mean_std, fmt_row_ci, fmt_row_mean_std
from src.backtester     import purged_walk_forward_splits, run_backtest
from src.visualization  import plot_equity_curves, plot_confusion_matrix, plot_drawdown

# ── Load config once at module level ─────────────────────────────────────────
_cfg          = load_config()
FEATURE_COLS  = _cfg["features"]["cols"]
SMA_SHORT     = _cfg["features"]["sma_short"]
SMA_LONG      = _cfg["features"]["sma_long"]
RSI_WINDOW    = _cfg["features"]["rsi_window"]
BB_WINDOW     = _cfg["features"]["bb_window"]
BB_STD        = _cfg["features"]["bb_std"]
MACD_FAST     = _cfg["features"]["macd_fast"]
MACD_SLOW     = _cfg["features"]["macd_slow"]
MACD_SIG      = _cfg["features"]["macd_sig"]
ATR_WINDOW    = _cfg["features"]["atr_window"]
VOL_WINDOW    = _cfg["features"]["vol_window"]
LAG_PERIODS   = _cfg["features"]["lag_periods"]
N_SPLITS      = _cfg["validation"]["n_splits"]
PURGE_GAP     = _cfg["validation"]["purge_gap"]
NN_EPOCHS     = _cfg["neural_network"]["epochs"]
NN_BATCH_SIZE = _cfg["neural_network"]["batch_size"]
NN_PATIENCE   = _cfg["neural_network"]["patience"]
NN_APP_SEEDS  = _cfg["neural_network"]["app_seeds"]
RISK_FREE_RATE   = _cfg["backtest"]["risk_free_rate"]
BT_EXECUTION_LAG = _cfg["backtest"]["execution_lag"]
BT_COSTS_BPS     = _cfg["backtest"]["costs_bps"]
RANDOM_SEED   = _cfg["general"]["random_seed"]
PLOTS_DIR     = _cfg["general"]["plots_dir"]
DATA_START    = _cfg["data"]["start"]
DATA_END      = _cfg["data"]["end"]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline utilities
# ─────────────────────────────────────────────────────────────────────────────

def _scale_fold(X_all, train_idx, test_idx):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_all[train_idx])
    X_test  = scaler.transform(X_all[test_idx])
    return X_train, X_test


@st.cache_data(show_spinner=False)
def _fetch_and_engineer(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download / load cached OHLCV and compute all features."""
    raw_df = load_or_download(ticker, start, end)
    return engineer_features(
        raw_df,
        sma_short=SMA_SHORT,   sma_long=SMA_LONG,
        rsi_window=RSI_WINDOW,
        bb_window=BB_WINDOW,   bb_std=BB_STD,
        macd_fast=MACD_FAST,   macd_slow=MACD_SLOW,  macd_sig=MACD_SIG,
        atr_window=ATR_WINDOW, vol_window=VOL_WINDOW, lag_periods=LAG_PERIODS,
    )


@st.cache_data(show_spinner="Running pipeline…")
def _run_pipeline(ticker: str, start: str, end: str) -> dict:
    """
    Execute the full ML pipeline and return a structured results dict.

    Baselines (always_long, prev_day_momentum, shuffled_label) and
    Naive Bayes / Logistic Regression are trained on all folds. The Neural
    Network is trained on the final fold only, across NN_APP_SEEDS seeds
    (reported as mean ± std), to keep interactive runtime reasonable — this
    trades the full rigor of `python main.py` (a real three-way holdout
    split, validation-tuned thresholds, 10+ seeds) for a fast dashboard.
    Run `python main.py --seeds 10` for the fully-audited result.

    Cached on (ticker, start, end): re-running with the same inputs (e.g.
    clicking "Run" again without changing anything) returns the stored
    result instantly instead of retraining everything from scratch. Only a
    new ticker or date range triggers real work.
    """
    np.random.seed(RANDOM_SEED)

    data          = _fetch_and_engineer(ticker, start, end)
    X_all         = data[FEATURE_COLS].values
    y_all         = data["Target"].values
    fwd_ret_all   = data["Fwd_Return"].values.ravel()
    daily_ret_all = data["Daily_Return"].values.ravel()

    splits  = purged_walk_forward_splits(len(X_all), N_SPLITS, PURGE_GAP)
    n_folds = len(splits)

    DET_MODELS = ["always_long", "prev_day_momentum", "shuffled_label", "nb", "lr"]
    LABELS = {
        "always_long": "always_long", "prev_day_momentum": "prev_day_momentum",
        "shuffled_label": "shuffled_label", "nb": "Naive Bayes", "lr": "Logistic Regression",
        "nn": "Neural Network (MLP)",
    }

    fold_metrics = {k: [] for k in DET_MODELS}
    final_y_test = final_fwd_ret = None
    final_preds  = {}
    final_ci_rows = {}
    nn_seed_rows = None
    nn_agg = None

    prog = st.progress(0, text="Initialising…")

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        is_final = fold_idx == n_folds - 1
        tag      = f"Fold {fold_idx + 1}/{n_folds}{'  —  final fold' if is_final else ''}"
        prog.progress((fold_idx + 0.5) / n_folds, text=f"Training {tag}…")

        X_train, X_test = _scale_fold(X_all, train_idx, test_idx)
        y_train  = y_all[train_idx]
        y_test   = y_all[test_idx]
        dr_test  = daily_ret_all[test_idx]

        preds = {
            "always_long":       predict_always_long(train_always_long(X_train, y_train), X_test),
            "prev_day_momentum": predict_prev_day_momentum(train_prev_day_momentum(X_train, y_train), dr_test),
            "shuffled_label":    predict_shuffled_label(train_shuffled_label(X_train, y_train, seed=fold_idx), X_test),
            "nb":                predict_naive_bayes(train_naive_bayes(X_train, y_train), X_test),
            "lr":                predict_logistic(train_logistic(X_train, y_train), X_test),
        }
        for key in DET_MODELS:
            fold_metrics[key].append(compute_metrics(LABELS[key], y_test, preds[key]))

        if is_final:
            prog.progress(0.75, text=f"Training Neural Network  ({NN_APP_SEEDS} seeds, final fold)…")
            nn_seed_rows = []
            nn_preds_by_seed = {}
            for seed in range(RANDOM_SEED, RANDOM_SEED + NN_APP_SEEDS):
                nn_model = train_mlp(X_train, y_train, epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE,
                                      patience=NN_PATIENCE, seed=seed)
                nn_preds_by_seed[seed] = predict_mlp(nn_model, X_test)
                nn_seed_rows.append(compute_metrics_ci("Neural Network (MLP)", y_test, nn_preds_by_seed[seed], seed=seed))
            nn_agg = aggregate_mean_std(nn_seed_rows, "n_seeds")
            preds["nn"] = nn_preds_by_seed[RANDOM_SEED]  # representative seed, for backtest + plots

            final_y_test  = y_test
            final_fwd_ret = fwd_ret_all[test_idx]
            final_preds   = preds
            for key in DET_MODELS:
                final_ci_rows[key] = compute_metrics_ci(LABELS[key], y_test, preds[key], seed=fold_idx)

    prog.progress(0.95, text="Running backtests and saving plots…")

    model_keys   = DET_MODELS + ["nn"]
    model_labels = [LABELS[k] for k in model_keys]
    backtests    = {
        k: run_backtest(final_preds[k], final_fwd_ret, RISK_FREE_RATE,
                         execution_lag=BT_EXECUTION_LAG, costs_bps=BT_COSTS_BPS)
        for k in model_keys
    }

    plots_dir = os.path.join(ROOT_DIR, PLOTS_DIR)
    bt_list   = [backtests[k] for k in model_keys]

    plot_equity_curves(bt_list, model_labels, plots_dir)
    plot_drawdown(bt_list, model_labels, plots_dir)
    for key, name in zip(model_keys, model_labels):
        plot_confusion_matrix(final_y_test, final_preds[key], name, plots_dir)

    prog.progress(1.0, text="Pipeline complete.")
    prog.empty()

    return {
        "ticker":         ticker,
        "start":          start,
        "end":            end,
        "data":           data,
        "fold_metrics":   fold_metrics,
        "final_ci_rows":  final_ci_rows,
        "nn_agg":         nn_agg,
        "backtests":      backtests,
        "y_test":         final_y_test,
        "preds":          final_preds,
        "plots_dir":      plots_dir,
        "model_keys":     model_keys,
        "model_labels":   model_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ci_classification_df(final_ci_rows: dict, nn_agg: dict) -> pd.DataFrame:
    rows = [
        {"Model": row["Model"], **dict(zip(
            ["Accuracy", "Precision", "Recall", "F1-Score"], fmt_row_ci(row)[1:],
        ))}
        for row in final_ci_rows.values()
    ]
    if nn_agg is not None:
        mean_std_row = fmt_row_mean_std(nn_agg, "n_seeds", "seeds")
        rows.append(dict(zip(["Model", "Accuracy", "Precision", "Recall", "F1-Score"], mean_std_row)))
    return pd.DataFrame(rows)


def _walk_forward_df(fold_metrics: dict) -> pd.DataFrame:
    agg = aggregate_walk_forward(fold_metrics)
    return pd.DataFrame([
        {
            "Model":     v["Model"],
            "Accuracy":  v["Accuracy"],
            "Precision": v["Precision"],
            "Recall":    v["Recall"],
            "F1-Score":  v["F1-Score"],
        }
        for v in agg.values()
    ])


def _backtest_row(bt: dict, model_name: str) -> None:
    alpha = bt["total_return"] - bt["benchmark_return"]
    st.caption(f"**{model_name}**")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Strategy Return",  f"{bt['total_return']:.2f} %",
              delta=f"{alpha:+.2f} % vs B&H", delta_color="normal")
    c2.metric("Benchmark (B&H)", f"{bt['benchmark_return']:.2f} %")
    c3.metric("Sharpe Ratio",    f"{bt['sharpe']:.4f}")
    c4.metric("Max Drawdown",    f"{bt['max_drawdown']:.2f} %")
    c5.metric("Win Rate",        f"{bt['win_rate']:.2f} %")
    c6.metric("Days Long",       f"{bt['days_long']:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Pipeline Controls")
    st.divider()

    ticker_input = st.text_input(
        "Stock Ticker", value="AAPL", max_chars=10,
        help="Any ticker supported by Yahoo Finance  (e.g. MSFT, GOOG, SPY, QQQ).",
    )
    ticker = ticker_input.upper().strip()

    start_date = st.date_input(
        "Start Date", value=pd.Timestamp(DATA_START),
        help="Start of historical data window (inclusive).",
    )
    end_date = st.date_input(
        "End Date", value=pd.Timestamp(DATA_END),
        help="End of historical data window (exclusive).",
    )
    start_str = str(start_date)
    end_str   = str(end_date)

    st.divider()
    run_clicked = st.button("Run Quant Pipeline", type="primary", use_container_width=True)
    st.divider()

    with st.expander("Feature Set", expanded=False):
        for feat in FEATURE_COLS:
            st.caption(f"• {feat}")

    with st.expander("Validation Config", expanded=False):
        st.caption(f"Walk-forward folds : {N_SPLITS}")
        st.caption(f"Purge gap          : {PURGE_GAP} trading days")
        st.caption(f"MLP seeds          : {NN_APP_SEEDS}  (CLI default: {_cfg['neural_network']['default_seeds']})")
        st.caption(f"Risk-free rate     : {RISK_FREE_RATE * 100:.1f} %")
        st.caption(f"Execution lag      : {BT_EXECUTION_LAG} bar")
        st.caption(f"Cost per side      : {BT_COSTS_BPS} bps")

    with st.expander("Model Architecture", expanded=False):
        st.caption("**always_long / prev_day_momentum / shuffled_label** — baselines, in every table")
        st.caption("**Naive Bayes** — GaussianNB + balanced sample_weight")
        st.caption("**Logistic Regression** — direction classifier + balanced class weights")
        st.caption("**MLP** — 128 → 64 → 32, BatchNorm, Dropout, ReduceLROnPlateau")


# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 ML Stock Direction & Backtest Dashboard")
st.caption(
    "Walk-forward validated ML pipeline  ·  Naive Bayes  ·  Logistic Regression  ·  Neural Network  "
    "| Backtested against buy-and-hold with Sharpe ratio, max drawdown, and equity-curve analysis."
)
st.divider()

if run_clicked:
    if not ticker:
        st.error("Please enter a valid ticker symbol in the sidebar.")
    else:
        try:
            results = _run_pipeline(ticker, start_str, end_str)
            st.session_state["results"] = results
        except ValueError as exc:
            st.error(f"Data error — {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Pipeline failed — {exc}")
            st.exception(exc)
            st.stop()

if "results" not in st.session_state:
    st.info(
        "Select a ticker and date range in the sidebar, "
        "then click **Run Quant Pipeline** to begin."
    )
    st.stop()

res  = st.session_state["results"]
data = res["data"]

up_pct   = data["Target"].mean() * 100
dn_pct   = 100.0 - up_pct
date_rng = (
    f"{data.index[0].strftime('%Y-%m-%d')}  →  {data.index[-1].strftime('%Y-%m-%d')}"
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ticker",      res["ticker"])
m2.metric("Date Range",  date_rng)
m3.metric("Data Points", f"{len(data):,}")
m4.metric("UP / DOWN",   f"{up_pct:.1f}% / {dn_pct:.1f}%")
m5.metric("Features",    len(FEATURE_COLS))
st.divider()

tab_eval, tab_bt = st.tabs(["📊 Model Evaluation", "💰 Backtest Performance"])

with tab_eval:
    st.caption(
        "This dashboard trades rigor for speed: a real held-out period, validation-tuned "
        "thresholds, and 10-seed MLP variance live in the CLI (`python main.py --seeds 10`), "
        "not here. What's below is fast enough to click through interactively."
    )

    shuffled_row = res["final_ci_rows"].get("shuffled_label")
    always_long_acc = res["final_ci_rows"]["always_long"]["Accuracy"]
    if shuffled_row is not None and shuffled_row["Accuracy_CI"][0] > always_long_acc:
        st.warning(
            "⚠️ Leak check failed: shuffled_label's accuracy CI sits entirely above "
            "always_long's accuracy on this fold. A classifier trained on permuted labels "
            "should not beat the trivial baseline — treat these results with suspicion."
        )

    st.subheader("Walk-Forward Summary")
    st.caption(
        f"{N_SPLITS}-fold expanding-window cross-validation with a {PURGE_GAP}-day purge gap.  "
        "Baselines (always_long, prev_day_momentum, shuffled_label) are included in every table "
        "— never compare a model's accuracy without them.  Values show **mean ± std** across all folds."
    )
    st.dataframe(_walk_forward_df(res["fold_metrics"]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Final-Fold Classification Report")
    st.caption(
        "Detailed per-class metrics on the most recent out-of-sample test window, with a 95% "
        f"confidence interval on each figure. Neural Network shows mean ± std across {NN_APP_SEEDS} "
        "training seeds — its training is stochastic, so a single seed is not a result."
    )
    st.dataframe(_ci_classification_df(res["final_ci_rows"], res["nn_agg"]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Confusion Matrices  —  Final Fold")
    st.caption(
        "Row-normalised rates shown with raw counts.  Rows = true label, Columns = predicted.  "
        "Neural Network uses its representative seed's predictions (the same one backtested below)."
    )
    # use_column_width, not use_container_width: streamlit==1.35.0 (the pin in
    # requirements.txt) doesn't have use_container_width on st.image. If that pin
    # ever moves, note that newer Streamlit deprecates use_column_width in favor
    # of a width= parameter — re-check both call sites below when it does.
    labels = res["model_labels"]
    for row_start in range(0, len(labels), 3):
        cm_cols = st.columns(3)
        for col, label in zip(cm_cols, labels[row_start:row_start + 3]):
            safe     = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            img_path = os.path.join(res["plots_dir"], f"confusion_matrix_{safe}.png")
            with col:
                if os.path.exists(img_path):
                    st.image(img_path, caption=label, use_column_width=True)
                else:
                    st.warning(f"Plot not found: {img_path}")

with tab_bt:
    st.subheader("Financial Performance Metrics  —  Final Fold")
    st.caption(
        "Long-only strategy: enter long when model predicts UP; hold cash otherwise.  "
        f"Benchmark: buy-and-hold over the same test window.  "
        f"Risk-free rate: {RISK_FREE_RATE * 100:.1f} % annualised.  "
        "Baselines included — always_long is the null hypothesis this strategy has to beat."
    )
    st.write("")
    for key, name in zip(res["model_keys"], res["model_labels"]):
        _backtest_row(res["backtests"][key], name)
        st.write("")

    st.divider()
    st.subheader("Cumulative Equity Curve")
    eq_path = os.path.join(res["plots_dir"], "equity_curves.png")
    if os.path.exists(eq_path):
        st.image(eq_path, use_column_width=True)
    else:
        st.warning(f"Equity curve plot not found at {eq_path}")

    st.subheader("Rolling Drawdown")
    dd_path = os.path.join(res["plots_dir"], "drawdown.png")
    if os.path.exists(dd_path):
        st.image(dd_path, use_column_width=True)
    else:
        st.warning(f"Drawdown plot not found at {dd_path}")

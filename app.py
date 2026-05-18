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
    train_ridge, predict_ridge_direction,
    train_mlp, predict_mlp,
    compute_metrics, aggregate_walk_forward,
)
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
LAG_PERIODS   = _cfg["features"]["lag_periods"]
N_SPLITS      = _cfg["validation"]["n_splits"]
PURGE_GAP     = _cfg["validation"]["purge_gap"]
NN_EPOCHS     = _cfg["neural_network"]["epochs"]
NN_BATCH_SIZE = _cfg["neural_network"]["batch_size"]
NN_PATIENCE   = _cfg["neural_network"]["patience"]
RISK_FREE_RATE = _cfg["backtest"]["risk_free_rate"]
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
        atr_window=ATR_WINDOW, lag_periods=LAG_PERIODS,
    )


def _run_pipeline(ticker: str, start: str, end: str) -> dict:
    """
    Execute the full ML pipeline and return a structured results dict.

    NB and Ridge are trained on all folds.  The Neural Network is trained
    on the final fold only to keep interactive runtime reasonable.
    """
    np.random.seed(RANDOM_SEED)

    data        = _fetch_and_engineer(ticker, start, end)
    X_all       = data[FEATURE_COLS].values
    y_all       = data["Target"].values
    close_all   = data["Close"].values.ravel()
    nxt_cls_all = data["Next_Close"].values.ravel()
    fwd_ret_all = data["Fwd_Return"].values.ravel()

    splits  = purged_walk_forward_splits(len(X_all), N_SPLITS, PURGE_GAP)
    n_folds = len(splits)

    fold_metrics                                     = {"nb": [], "lr": [], "nn": []}
    final_y_test = final_fwd_ret                     = None
    final_nb_preds = final_lr_preds = final_nn_preds = None

    prog = st.progress(0, text="Initialising…")

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        is_final = fold_idx == n_folds - 1
        tag      = f"Fold {fold_idx + 1}/{n_folds}{'  —  final fold' if is_final else ''}"
        prog.progress((fold_idx + 0.5) / n_folds, text=f"Training {tag}  (Naive Bayes + Ridge)…")

        X_train, X_test = _scale_fold(X_all, train_idx, test_idx)
        y_train  = y_all[train_idx]
        y_test   = y_all[test_idx]
        y_reg_tr = nxt_cls_all[train_idx]
        tc_test  = close_all[test_idx]

        nb_preds = predict_naive_bayes(train_naive_bayes(X_train, y_train), X_test)
        lr_preds = predict_ridge_direction(
            train_ridge(X_train, y_reg_tr, y_train, alpha=1.0), X_test, tc_test,
        )

        fold_metrics["nb"].append(compute_metrics("Naive Bayes",              y_test, nb_preds))
        fold_metrics["lr"].append(compute_metrics("Linear Regression (Ridge)", y_test, lr_preds))

        if is_final:
            prog.progress(0.85, text="Training Neural Network  (final fold)…")
            nn_preds = predict_mlp(
                train_mlp(
                    X_train, y_train,
                    epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE,
                    patience=NN_PATIENCE, seed=RANDOM_SEED,
                ),
                X_test,
            )
            fold_metrics["nn"].append(compute_metrics("Neural Network (MLP)", y_test, nn_preds))

            final_y_test   = y_test
            final_fwd_ret  = fwd_ret_all[test_idx]
            final_nb_preds = nb_preds
            final_lr_preds = lr_preds
            final_nn_preds = nn_preds

    prog.progress(0.95, text="Running backtests and saving plots…")

    bt_nb = run_backtest(final_nb_preds, final_fwd_ret, RISK_FREE_RATE)
    bt_lr = run_backtest(final_lr_preds, final_fwd_ret, RISK_FREE_RATE)
    bt_nn = run_backtest(final_nn_preds, final_fwd_ret, RISK_FREE_RATE)

    plots_dir    = os.path.join(ROOT_DIR, PLOTS_DIR)
    model_labels = ["Naive Bayes", "Ridge", "Neural Network"]
    bt_list      = [bt_nb, bt_lr, bt_nn]

    plot_equity_curves(bt_list, model_labels, plots_dir)
    plot_drawdown(bt_list, model_labels, plots_dir)
    for preds, name in zip([final_nb_preds, final_lr_preds, final_nn_preds], model_labels):
        plot_confusion_matrix(final_y_test, preds, name, plots_dir)

    prog.progress(1.0, text="Pipeline complete.")
    prog.empty()

    return {
        "ticker":        ticker,
        "start":         start,
        "end":           end,
        "data":          data,
        "fold_metrics":  fold_metrics,
        "final_results": [
            fold_metrics["nb"][-1],
            fold_metrics["lr"][-1],
            fold_metrics["nn"][-1],
        ],
        "backtests":     {"nb": bt_nb, "lr": bt_lr, "nn": bt_nn},
        "y_test":        final_y_test,
        "preds":         {"nb": final_nb_preds, "lr": final_lr_preds, "nn": final_nn_preds},
        "plots_dir":     plots_dir,
        "model_labels":  model_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classification_df(results: list) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Model":     r["Model"],
            "Accuracy":  f"{r['Accuracy']:.2f} %",
            "Precision": f"{r['Precision']:.2f} %",
            "Recall":    f"{r['Recall']:.2f} %",
            "F1-Score":  f"{r['F1-Score']:.2f} %",
        }
        for r in results
    ])


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
    c6.metric("Trades",          f"{bt['n_trades']:,}")


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
        st.caption(f"Risk-free rate     : {RISK_FREE_RATE * 100:.1f} %")

    with st.expander("Model Architecture", expanded=False):
        st.caption("**Naive Bayes** — GaussianNB + balanced sample_weight")
        st.caption("**Ridge** — price forecast → direction + balanced weights")
        st.caption("**MLP** — 128 → 64 → 32, BatchNorm, Dropout, ReduceLROnPlateau")


# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 ML Stock Direction & Backtest Dashboard")
st.caption(
    "Walk-forward validated ML pipeline  ·  Naive Bayes  ·  Ridge Regression  ·  Neural Network  "
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
    st.subheader("Walk-Forward Summary")
    st.caption(
        f"{N_SPLITS}-fold expanding-window cross-validation with a {PURGE_GAP}-day purge gap.  "
        "Values show **mean ± std** across all folds.  "
        "Neural Network is evaluated on the final fold only."
    )
    st.dataframe(_walk_forward_df(res["fold_metrics"]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Final-Fold Classification Report")
    st.caption("Detailed per-class metrics on the most recent out-of-sample test window.")
    st.dataframe(_classification_df(res["final_results"]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Confusion Matrices  —  Final Fold")
    st.caption("Row-normalised rates shown with raw counts.  Rows = true label, Columns = predicted.")
    cm_cols = st.columns(3)
    for col, label in zip(cm_cols, res["model_labels"]):
        safe     = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        img_path = os.path.join(res["plots_dir"], f"confusion_matrix_{safe}.png")
        with col:
            if os.path.exists(img_path):
                st.image(img_path, caption=label, use_container_width=True)
            else:
                st.warning(f"Plot not found: {img_path}")

with tab_bt:
    st.subheader("Financial Performance Metrics  —  Final Fold")
    st.caption(
        "Long-only strategy: enter long when model predicts UP; hold cash otherwise.  "
        f"Benchmark: buy-and-hold over the same test window.  "
        f"Risk-free rate: {RISK_FREE_RATE * 100:.1f} % annualised."
    )
    st.write("")
    for key, name in zip(["nb", "lr", "nn"], res["model_labels"]):
        _backtest_row(res["backtests"][key], name)
        st.write("")

    st.divider()
    st.subheader("Cumulative Equity Curve")
    eq_path = os.path.join(res["plots_dir"], "equity_curves.png")
    if os.path.exists(eq_path):
        st.image(eq_path, use_container_width=True)
    else:
        st.warning(f"Equity curve plot not found at {eq_path}")

    st.subheader("Rolling Drawdown")
    dd_path = os.path.join(res["plots_dir"], "drawdown.png")
    if os.path.exists(dd_path):
        st.image(dd_path, use_container_width=True)
    else:
        st.warning(f"Drawdown plot not found at {dd_path}")

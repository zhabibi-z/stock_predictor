"""
Stock Market Direction Predictor
═════════════════════════════════
Pipeline orchestration script.  Configuration is read from
config/config.yaml; all logic lives in src/.

Stages
──────
1. Data Acquisition        — download / load cached OHLCV bars
2. Feature Engineering     — 10 indicators, zero temporal leakage
3. Purged Walk-Forward CV  — 5 expanding-window folds, 5-day purge gap
4. Walk-Forward Training   — NB + Ridge on all folds; NN on final fold
5. Walk-Forward Summary    — mean ± std across folds
6. Final-Fold Report       — detailed classification metrics
7. Backtesting             — Sharpe, max drawdown, equity curves
8. Visualisation           — plots saved to plots/

Usage
─────
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from sklearn.preprocessing import StandardScaler

from src                import load_config
from src.data_loader    import load_or_download
from src.features       import engineer_features
from src.models         import (
    train_naive_bayes, predict_naive_bayes,
    train_ridge, predict_ridge_direction,
    train_mlp, predict_mlp,
    compute_metrics, print_report, print_walk_forward_report,
)
from src.backtester     import (
    purged_walk_forward_splits, run_backtest, print_backtest_report,
)
from src.visualization  import (
    plot_equity_curves, plot_confusion_matrix, plot_drawdown,
)


def _banner(step: int, title: str) -> None:
    print(f"\n{'═' * 66}")
    print(f"  STEP {step}  │  {title}")
    print(f"{'═' * 66}")


def _scale_fold(X_all, train_idx, test_idx):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_all[train_idx])
    X_test  = scaler.transform(X_all[test_idx])
    return X_train, X_test


def main() -> None:
    cfg = load_config()
    d   = cfg["data"]
    f   = cfg["features"]
    v   = cfg["validation"]
    nn  = cfg["neural_network"]
    bt  = cfg["backtest"]
    g   = cfg["general"]

    TICKER       = d["ticker"]
    DATA_START   = d["start"]
    DATA_END     = d["end"]
    FEATURE_COLS = f["cols"]
    N_SPLITS     = v["n_splits"]
    PURGE_GAP    = v["purge_gap"]
    RANDOM_SEED  = g["random_seed"]
    PLOTS_DIR    = g["plots_dir"]

    np.random.seed(RANDOM_SEED)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── STEP 1: Data Acquisition ──────────────────────────────────────────────
    _banner(1, f"Data Acquisition  ({TICKER},  {DATA_START} → {DATA_END})")
    raw_df = load_or_download(TICKER, DATA_START, DATA_END)

    # ── STEP 2: Feature Engineering ───────────────────────────────────────────
    _banner(2, "Feature Engineering  (10 indicators, zero data leakage)")
    data = engineer_features(
        raw_df,
        sma_short=f["sma_short"],   sma_long=f["sma_long"],
        rsi_window=f["rsi_window"],
        bb_window=f["bb_window"],   bb_std=f["bb_std"],
        macd_fast=f["macd_fast"],   macd_slow=f["macd_slow"],  macd_sig=f["macd_sig"],
        atr_window=f["atr_window"],
        lag_periods=f["lag_periods"],
    )
    up_pct = data["Target"].mean() * 100
    print(f"      Rows : {len(data):,}   Features : {len(FEATURE_COLS)}")
    print(f"      Class distribution : UP={up_pct:.1f}%  DOWN={100 - up_pct:.1f}%")

    # ── STEP 3: Purged Walk-Forward Splits ────────────────────────────────────
    _banner(3, f"Purged Walk-Forward CV  (n_splits={N_SPLITS}, purge_gap={PURGE_GAP} days)")

    X_all       = data[FEATURE_COLS].values
    y_all       = data["Target"].values
    close_all   = data["Close"].values.ravel()
    nxt_cls_all = data["Next_Close"].values.ravel()
    fwd_ret_all = data["Fwd_Return"].values.ravel()

    splits = purged_walk_forward_splits(len(X_all), N_SPLITS, PURGE_GAP)
    print(f"      Total rows : {len(X_all):,}   Folds : {len(splits)}")
    for k, (tr, te) in enumerate(splits):
        print(f"        Fold {k + 1} : train {len(tr):,}  │  test {len(te):,}")

    # ── STEP 4: Walk-Forward Training ─────────────────────────────────────────
    _banner(4, "Walk-Forward Training  (NB + Ridge all folds; NN final fold)")

    fold_metrics = {"nb": [], "lr": [], "nn": []}

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        is_final = fold_idx == len(splits) - 1
        label    = "final" if is_final else f"{fold_idx + 1}/{len(splits)}"
        print(f"\n    ── Fold {fold_idx + 1}/{len(splits)}  ({label}) ──")

        X_train, X_test = _scale_fold(X_all, train_idx, test_idx)
        y_train  = y_all[train_idx]
        y_test   = y_all[test_idx]
        y_reg_tr = nxt_cls_all[train_idx]
        tc_test  = close_all[test_idx]

        nb_preds = predict_naive_bayes(train_naive_bayes(X_train, y_train), X_test)
        lr_preds = predict_ridge_direction(
            train_ridge(X_train, y_reg_tr, y_train, alpha=1.0), X_test, tc_test,
        )

        nb_m = compute_metrics("Naive Bayes",              y_test, nb_preds)
        lr_m = compute_metrics("Linear Regression (Ridge)", y_test, lr_preds)
        fold_metrics["nb"].append(nb_m)
        fold_metrics["lr"].append(lr_m)
        print(f"      NB  acc={nb_m['Accuracy']}%   Ridge acc={lr_m['Accuracy']}%")

        if is_final:
            nn_model = train_mlp(
                X_train, y_train,
                epochs=nn["epochs"], batch_size=nn["batch_size"],
                patience=nn["patience"], seed=RANDOM_SEED,
            )
            nn_preds = predict_mlp(nn_model, X_test)
            nn_m     = compute_metrics("Neural Network (MLP)", y_test, nn_preds)
            fold_metrics["nn"].append(nn_m)
            print(f"      NN  acc={nn_m['Accuracy']}%")

            final_y_test   = y_test
            final_fwd_ret  = fwd_ret_all[test_idx]
            final_nb_preds = nb_preds
            final_lr_preds = lr_preds
            final_nn_preds = nn_preds

    # ── STEP 5: Walk-Forward Summary ──────────────────────────────────────────
    _banner(5, "Walk-Forward Summary  (mean ± std across all folds)")
    print_walk_forward_report(fold_metrics)

    # ── STEP 6: Final-Fold Report ─────────────────────────────────────────────
    _banner(6, "Final-Fold Classification Report")
    print_report([fold_metrics["nb"][-1], fold_metrics["lr"][-1], fold_metrics["nn"][-1]])

    # ── STEP 7: Vectorized Backtesting ────────────────────────────────────────
    _banner(7, "Vectorized Backtesting  (long-only, frictionless)")
    bt_nb = run_backtest(final_nb_preds, final_fwd_ret, bt["risk_free_rate"])
    bt_lr = run_backtest(final_lr_preds, final_fwd_ret, bt["risk_free_rate"])
    bt_nn = run_backtest(final_nn_preds, final_fwd_ret, bt["risk_free_rate"])

    print_backtest_report("Naive Bayes",              bt_nb)
    print_backtest_report("Linear Regression (Ridge)", bt_lr)
    print_backtest_report("Neural Network (MLP)",      bt_nn)

    # ── STEP 8: Visualisation ─────────────────────────────────────────────────
    _banner(8, f"Visualisation  (saving plots to '{PLOTS_DIR}/')")
    bt_list = [bt_nb, bt_lr, bt_nn]
    labels  = ["Naive Bayes", "Ridge", "Neural Network"]

    plot_equity_curves(bt_list, labels, PLOTS_DIR)
    plot_drawdown(bt_list, labels, PLOTS_DIR)
    for preds, name in zip([final_nb_preds, final_lr_preds, final_nn_preds], labels):
        plot_confusion_matrix(final_y_test, preds, name, PLOTS_DIR)

    print(f"\n{'═' * 66}")
    print("  PIPELINE COMPLETE")
    print(f"{'═' * 66}\n")


if __name__ == "__main__":
    main()

"""
Stock Market Direction Predictor
═════════════════════════════════
Pipeline orchestration script.  Configuration is read from
config/config.yaml; all logic lives in src/.

Stages
──────
1. Data Acquisition        — download / load cached OHLCV bars
2. Feature Engineering     — 11 indicators (SMA_5/SMA_20 as Close/SMA-1
                              ratios, not raw price levels), zero leakage
3. Three-Way Split         — train_pool / validation / holdout (holdout
                              touched exactly once, at Step 7); asserts
                              SMA_5/SMA_20 train/test z-ranges overlap
4. Threshold Tuning        — NB / LogReg / MLP, balanced and unbalanced,
                              threshold selected on the validation split
                              (never the holdout, never a CV test fold)
5. Walk-Forward Training   — 3 baselines + NB + LogReg + MLP(multi-seed),
                              ALL folds, ALL models, unbalanced + tuned
                              threshold (see Step 6's ablation for why)
6. Walk-Forward Summary    — mean ± std across folds, CI on every figure
7. Holdout Report          — the only evaluation on unseen data, once,
                              plus the balance/threshold ablation in full
8. Backtesting             — Sharpe, max drawdown, equity curves
9. Visualisation           — plots saved to plots/

Usage
─────
    python main.py [--seeds N]

    --seeds N   Number of MLP training seeds per fold and for the holdout
                report (default: neural_network.default_seeds in config).
                Mean ± std across seeds is reported alongside every MLP
                accuracy figure — a point estimate from one seed is not a
                result.
"""

import argparse
import os
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler

from src                import load_config
from src.data_loader    import load_or_download
from src.features       import engineer_features
from src.models         import (
    train_naive_bayes, predict_proba_naive_bayes,
    train_logistic, predict_logistic, predict_proba_logistic,
    train_mlp, predict_mlp, predict_proba_mlp,
)
from src.baselines      import (
    train_always_long, predict_always_long,
    train_prev_day_momentum, predict_prev_day_momentum,
    train_shuffled_label, predict_shuffled_label,
)
from src.reporting      import (
    compute_metrics_ci, aggregate_mean_std,
    fmt_row_ci, fmt_row_mean_std, print_table, check_leakage,
)
from src.tuning         import tune_threshold, threshold_report
from src.backtester     import (
    purged_walk_forward_splits, three_way_split, run_backtest, print_backtest_report,
)
from src.visualization  import (
    plot_equity_curves, plot_confusion_matrix, plot_drawdown,
)

warnings.filterwarnings("ignore")

HEADER = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]

# Canonical config for the walk-forward CV and headline holdout report. The
# full balanced-vs-unbalanced, 0.50-vs-tuned comparison lives in Step 7's
# ablation table on the holdout — this default isn't asserted as "correct,"
# it's just the one path repeated across every fold for a stable narrative.
CANONICAL_BALANCED = False


def _banner(step: int, title: str) -> None:
    print(f"\n{'═' * 66}")
    print(f"  STEP {step}  │  {title}")
    print(f"{'═' * 66}")


def _scale(X_all, fit_idx, apply_idx):
    scaler = StandardScaler()
    X_fit    = scaler.fit_transform(X_all[fit_idx])
    X_applied = scaler.transform(X_all[apply_idx])
    return X_fit, X_applied


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds", type=int, default=None,
        help="MLP training seeds per evaluation (default: neural_network.default_seeds)",
    )
    return p.parse_args()


def _assert_feature_overlap(X_all, feature_cols, train_idx, test_idx, names=("SMA_5", "SMA_20")):
    """Re-measure the train/test z-range for each name and assert it overlaps.
    This is the direct, re-run diagnostic behind the fix in src/features.py —
    it must pass on the holdout, the fold most exposed to a decade of price
    drift, or the ratio transform didn't do its job."""
    scaler = StandardScaler().fit(X_all[train_idx])
    X_tr = scaler.transform(X_all[train_idx])
    X_te = scaler.transform(X_all[test_idx])
    for name in names:
        idx = feature_cols.index(name)
        tr_lo, tr_hi = X_tr[:, idx].min(), X_tr[:, idx].max()
        te_lo, te_hi = X_te[:, idx].min(), X_te[:, idx].max()
        overlap = not (te_lo > tr_hi or te_hi < tr_lo)
        print(f"      {name:8s} train z [{tr_lo:6.2f}, {tr_hi:6.2f}]   "
              f"test z [{te_lo:6.2f}, {te_hi:6.2f}]   overlap={overlap}")
        assert overlap, (
            f"{name}: train/test z-ranges do not overlap on the holdout "
            f"(train=[{tr_lo:.2f},{tr_hi:.2f}] test=[{te_lo:.2f},{te_hi:.2f}]) — "
            "the ratio transform did not fix the out-of-distribution problem."
        )


def _eval_deterministic_models(X_train, y_train, X_test, y_test, daily_ret_test, fold_seed,
                                nb_threshold=0.50, lr_threshold=0.50, balanced=CANONICAL_BALANCED):
    """Train + evaluate the three baselines, Naive Bayes, and Logistic Regression.
    Returns (all rows, shuffled_label row, always_long row)."""
    rows = []

    al_model = train_always_long(X_train, y_train)
    al_preds = predict_always_long(al_model, X_test)
    rows.append(compute_metrics_ci("always_long", y_test, al_preds, seed=fold_seed))

    pm_model = train_prev_day_momentum(X_train, y_train)
    pm_preds = predict_prev_day_momentum(pm_model, daily_ret_test)
    rows.append(compute_metrics_ci("prev_day_momentum", y_test, pm_preds, seed=fold_seed + 1))

    sh_model = train_shuffled_label(X_train, y_train, seed=fold_seed)
    sh_preds = predict_shuffled_label(sh_model, X_test)
    shuffled_row = compute_metrics_ci("shuffled_label", y_test, sh_preds, seed=fold_seed + 2)
    rows.append(shuffled_row)

    nb_model = train_naive_bayes(X_train, y_train, balanced=balanced)
    nb_preds = (predict_proba_naive_bayes(nb_model, X_test) >= nb_threshold).astype(int)
    rows.append(compute_metrics_ci("Naive Bayes", y_test, nb_preds, seed=fold_seed + 3))

    lr_model = train_logistic(X_train, y_train, balanced=balanced)
    lr_preds = predict_logistic(lr_model, X_test, threshold=lr_threshold)
    rows.append(compute_metrics_ci("Logistic Regression", y_test, lr_preds, seed=fold_seed + 4))

    return rows, shuffled_row, rows[0]  # (all rows, shuffled row, always_long row)


def _eval_mlp_multiseed(X_train, y_train, X_test, y_test, seeds, nn_cfg,
                         threshold=0.50, balanced=CANONICAL_BALANCED):
    """Train the MLP once per seed; return (per-seed CI rows, mean±std row, {seed: preds})."""
    seed_rows = []
    preds_by_seed = {}
    for seed in seeds:
        model = train_mlp(
            X_train, y_train,
            epochs=nn_cfg["epochs"], batch_size=nn_cfg["batch_size"],
            patience=nn_cfg["patience"], seed=seed, balanced=balanced,
        )
        preds = predict_mlp(model, X_test, threshold=threshold)
        preds_by_seed[seed] = preds
        seed_rows.append(compute_metrics_ci("Neural Network (MLP)", y_test, preds, seed=seed))
    agg = aggregate_mean_std(seed_rows, "n_seeds")
    return seed_rows, agg, preds_by_seed


def _tune_all_thresholds(X_tp, y_tp, X_val, y_val, nn_cfg, random_seed):
    """
    Fit NB / Logistic Regression / MLP(representative seed) on the train
    pool, balanced and unbalanced, and pick the threshold that maximizes
    validation BALANCED accuracy for each (see src/tuning.py — raw accuracy
    degenerates to a near-constant majority-class predictor on this data).
    Returns {(model_name, balanced): threshold}. Never touches a CV test
    fold or the holdout.
    """
    tuned = {}

    def _report(name, balanced, t, probs_val):
        at_t   = threshold_report(y_val, probs_val, t)
        at_050 = threshold_report(y_val, probs_val, 0.50)
        print(f"      {name:22s} balanced={str(balanced):5s} -> tuned threshold={t:.2f}  "
              f"(val bal.acc @tuned={at_t['balanced_accuracy']*100:.2f}% [acc={at_t['accuracy']*100:.2f}%], "
              f"@0.50 bal.acc={at_050['balanced_accuracy']*100:.2f}% [acc={at_050['accuracy']*100:.2f}%])")

    specs = [
        ("Naive Bayes",         lambda b: train_naive_bayes(X_tp, y_tp, balanced=b), predict_proba_naive_bayes),
        ("Logistic Regression", lambda b: train_logistic(X_tp, y_tp, balanced=b),    predict_proba_logistic),
    ]
    for name, train_fn, proba_fn in specs:
        for balanced in (False, True):
            model = train_fn(balanced)
            probs_val = proba_fn(model, X_val)
            t, _ = tune_threshold(y_val, probs_val)
            tuned[(name, balanced)] = t
            _report(name, balanced, t, probs_val)

    for balanced in (False, True):
        model = train_mlp(X_tp, y_tp, epochs=nn_cfg["epochs"], batch_size=nn_cfg["batch_size"],
                           patience=nn_cfg["patience"], seed=random_seed, balanced=balanced)
        probs_val = predict_proba_mlp(model, X_val)
        t, _ = tune_threshold(y_val, probs_val)
        tuned[("Neural Network (MLP)", balanced)] = t
        _report("Neural Network (MLP)", balanced, t, probs_val)
    return tuned


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    d   = cfg["data"]
    f   = cfg["features"]
    v   = cfg["validation"]
    nn  = cfg["neural_network"]
    bt  = cfg["backtest"]
    g   = cfg["general"]

    TICKER        = d["ticker"]
    DATA_START    = d["start"]
    DATA_END      = d["end"]
    FEATURE_COLS  = f["cols"]
    N_SPLITS      = v["n_splits"]
    PURGE_GAP     = v["purge_gap"]
    HOLDOUT_START = v["holdout_start"]
    VAL_FRAC      = v["validation_frac"]
    RANDOM_SEED   = g["random_seed"]
    PLOTS_DIR     = g["plots_dir"]
    N_SEEDS       = args.seeds if args.seeds is not None else nn["default_seeds"]

    np.random.seed(RANDOM_SEED)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── STEP 1: Data Acquisition ──────────────────────────────────────────────
    _banner(1, f"Data Acquisition  ({TICKER},  {DATA_START} → {DATA_END})")
    raw_df = load_or_download(TICKER, DATA_START, DATA_END)

    # ── STEP 2: Feature Engineering ───────────────────────────────────────────
    _banner(2, "Feature Engineering  (11 indicators, zero data leakage)")
    data = engineer_features(
        raw_df,
        sma_short=f["sma_short"],   sma_long=f["sma_long"],
        rsi_window=f["rsi_window"],
        bb_window=f["bb_window"],   bb_std=f["bb_std"],
        macd_fast=f["macd_fast"],   macd_slow=f["macd_slow"],  macd_sig=f["macd_sig"],
        atr_window=f["atr_window"], vol_window=f["vol_window"],
        lag_periods=f["lag_periods"],
    )
    up_pct = data["Target"].mean() * 100
    print(f"      Rows : {len(data):,}   Features : {len(FEATURE_COLS)}")
    print(f"      Class distribution : UP={up_pct:.1f}%  DOWN={100 - up_pct:.1f}%")

    X_all          = data[FEATURE_COLS].values
    y_all          = data["Target"].values
    fwd_ret_all    = data["Fwd_Return"].values.ravel()
    daily_ret_all  = data["Daily_Return"].values.ravel()

    # ── STEP 3: Three-Way Split + feature-overlap assertion ──────────────────
    _banner(3, f"Three-Way Split  (holdout ≥ {HOLDOUT_START}, touched once, at Step 7)")
    train_pool_idx, validation_idx, holdout_idx = three_way_split(
        data.index, HOLDOUT_START, VAL_FRAC, PURGE_GAP,
    )
    print(f"      Train pool : {len(train_pool_idx):,} rows "
          f"({data.index[train_pool_idx[0]].date()} → {data.index[train_pool_idx[-1]].date()})")
    print(f"      Validation : {len(validation_idx):,} rows "
          f"({data.index[validation_idx[0]].date()} → {data.index[validation_idx[-1]].date()})  "
          f"[reserved for tuning]")
    print(f"      Holdout    : {len(holdout_idx):,} rows "
          f"({data.index[holdout_idx[0]].date()} → {data.index[holdout_idx[-1]].date()})  "
          f"[NOT TOUCHED until Step 7]")

    train_for_holdout_idx = np.concatenate([train_pool_idx, validation_idx])
    print("\n      SMA_5 / SMA_20 train/test z-range overlap (train_pool+validation vs holdout):")
    _assert_feature_overlap(X_all, FEATURE_COLS, train_for_holdout_idx, holdout_idx)

    splits = purged_walk_forward_splits(len(train_pool_idx), N_SPLITS, PURGE_GAP)
    print(f"\n      Walk-forward folds within train pool : {len(splits)}")
    for k, (tr, te) in enumerate(splits):
        print(f"        Fold {k + 1} : train {len(tr):,}  │  test {len(te):,}")

    # ── STEP 4: Threshold Tuning  (validation split, holdout untouched) ──────
    _banner(4, "Threshold Tuning  (validation split — never a CV fold, never the holdout)")
    X_tp, X_val = _scale(X_all, train_pool_idx, validation_idx)
    y_tp, y_val = y_all[train_pool_idx], y_all[validation_idx]
    tuned = _tune_all_thresholds(X_tp, y_tp, X_val, y_val, nn, RANDOM_SEED)

    nb_t = tuned[("Naive Bayes", CANONICAL_BALANCED)]
    lr_t = tuned[("Logistic Regression", CANONICAL_BALANCED)]
    mlp_t = tuned[("Neural Network (MLP)", CANONICAL_BALANCED)]
    print(f"\n      Canonical config for Steps 5-8: balanced={CANONICAL_BALANCED}  "
          f"(full balanced-vs-unbalanced comparison in Step 7's ablation)")
    print(f"      Thresholds carried forward — NB={nb_t:.2f}  LogReg={lr_t:.2f}  MLP={mlp_t:.2f}")

    # ── STEP 5: Walk-Forward Training  (every model, every fold) ─────────────
    _banner(5, f"Walk-Forward Training  (3 baselines + NB + LogReg + MLP×{N_SEEDS} seeds — ALL folds)")

    per_fold_rows = {}   # model label -> list of per-fold point-estimate dicts, for cross-fold aggregation
    any_leak = False

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n    ── Fold {fold_idx + 1}/{len(splits)} "
              f"({data.index[test_idx[0]].date()} → {data.index[test_idx[-1]].date()}) ──")

        X_train, X_test = _scale(X_all, train_idx, test_idx)
        y_train  = y_all[train_idx]
        y_test   = y_all[test_idx]
        dr_test  = daily_ret_all[test_idx]

        det_rows, shuffled_row, always_long_row = _eval_deterministic_models(
            X_train, y_train, X_test, y_test, dr_test, fold_seed=RANDOM_SEED + fold_idx * 100,
            nb_threshold=nb_t, lr_threshold=lr_t,
        )

        seeds = list(range(RANDOM_SEED, RANDOM_SEED + N_SEEDS))
        mlp_seed_rows, mlp_fold_agg, _ = _eval_mlp_multiseed(
            X_train, y_train, X_test, y_test, seeds, nn, threshold=mlp_t,
        )

        print_table(f"Fold {fold_idx + 1} — deterministic models (95% CI)", HEADER,
                    [fmt_row_ci(r) for r in det_rows])
        print_table(f"Fold {fold_idx + 1} — MLP across {N_SEEDS} seeds (mean ± std)", HEADER,
                    [fmt_row_mean_std(mlp_fold_agg, "n_seeds", "seeds")])

        gap = mlp_fold_agg["Accuracy"] - always_long_row["Accuracy"]
        seed_std = mlp_fold_agg["Accuracy_std"]
        if seed_std > abs(gap):
            print(f"      ⚠ MLP seed-to-seed std ({seed_std*100:.2f}pp) exceeds its edge over "
                  f"always_long ({gap*100:+.2f}pp) — this fold's MLP result is not distinguishable from noise.")

        if check_leakage(shuffled_row, always_long_row["Accuracy"]):
            any_leak = True

        for row in det_rows:
            per_fold_rows.setdefault(row["Model"], []).append(row)
        per_fold_rows.setdefault("Neural Network (MLP)", []).append(mlp_fold_agg)

    # ── STEP 6: Walk-Forward Summary  (mean ± std across all folds) ──────────
    _banner(6, "Walk-Forward Summary  (mean ± std across all folds)")
    summary_rows = []
    for label, rows in per_fold_rows.items():
        agg = aggregate_mean_std(rows, "n_folds")
        summary_rows.append(fmt_row_mean_std(agg, "n_folds", "folds"))
    print_table("Walk-Forward Summary — every model, all folds", HEADER, summary_rows)

    if any_leak:
        print("\n  >>> A LEAK WAS DETECTED IN AT LEAST ONE FOLD ABOVE. <<<")
        print("  >>> Stopping before the holdout is touched. Investigate before proceeding. <<<\n")
        return

    # ── STEP 7: Holdout Report  (touched exactly once) ────────────────────────
    _banner(7, f"Holdout Report  (≥ {HOLDOUT_START})  —  TOUCHED EXACTLY ONCE, RIGHT HERE")
    X_train_h, X_holdout = _scale(X_all, train_for_holdout_idx, holdout_idx)
    y_train_h = y_all[train_for_holdout_idx]
    y_holdout = y_all[holdout_idx]
    dr_h      = daily_ret_all[holdout_idx]

    det_rows_h, shuffled_row_h, always_long_row_h = _eval_deterministic_models(
        X_train_h, y_train_h, X_holdout, y_holdout, dr_h, fold_seed=RANDOM_SEED + 9000,
        nb_threshold=nb_t, lr_threshold=lr_t,
    )

    holdout_seeds = list(range(RANDOM_SEED, RANDOM_SEED + N_SEEDS))
    mlp_seed_rows_h, mlp_agg_h, mlp_preds_by_seed_h = _eval_mlp_multiseed(
        X_train_h, y_train_h, X_holdout, y_holdout, holdout_seeds, nn, threshold=mlp_t,
    )

    print_table("HOLDOUT — deterministic models (95% CI)", HEADER, [fmt_row_ci(r) for r in det_rows_h])
    print_table(f"HOLDOUT — MLP across {N_SEEDS} seeds (mean ± std)", HEADER,
                [fmt_row_mean_std(mlp_agg_h, "n_seeds", "seeds")])

    gap_h = mlp_agg_h["Accuracy"] - always_long_row_h["Accuracy"]
    seed_std_h = mlp_agg_h["Accuracy_std"]
    if seed_std_h > abs(gap_h):
        print(f"      ⚠ HOLDOUT: MLP seed-to-seed std ({seed_std_h*100:.2f}pp) exceeds its edge over "
              f"always_long ({gap_h*100:+.2f}pp) — the holdout result is not distinguishable from noise.")

    holdout_leak = check_leakage(shuffled_row_h, always_long_row_h["Accuracy"], model_label="shuffled-label (holdout)")

    # Reuse the seed=RANDOM_SEED model's predictions as the backtest's representative
    # seed, rather than training another MLP (Phase 3 reworks backtest realism further).
    mlp_preds_h = mlp_preds_by_seed_h[RANDOM_SEED]

    # ── STEP 7b: Class Balancing × Threshold Ablation  (HOLDOUT, for context) ─
    print("\n  " + "─" * 60)
    print("  Class Balancing × Threshold Ablation — HOLDOUT (context, not the headline)")
    print("  " + "─" * 60)
    ablation_rows = []
    det_specs = [
        ("Naive Bayes",         lambda b: train_naive_bayes(X_train_h, y_train_h, balanced=b), predict_proba_naive_bayes),
        ("Logistic Regression", lambda b: train_logistic(X_train_h, y_train_h, balanced=b),    predict_proba_logistic),
    ]
    for name, train_fn, proba_fn in det_specs:
        for balanced in (False, True):
            model = train_fn(balanced)
            probs = proba_fn(model, X_holdout)
            for thr_label, thr in (("thr=0.50", 0.50), (f"thr=tuned({tuned[(name, balanced)]:.2f})", tuned[(name, balanced)])):
                preds = (probs >= thr).astype(int)
                label = f"{name} (balanced={balanced}, {thr_label})"
                ablation_rows.append(compute_metrics_ci(label, y_holdout, preds, seed=RANDOM_SEED))

    for balanced in (False, True):
        probs = predict_proba_mlp(
            train_mlp(X_train_h, y_train_h, epochs=nn["epochs"], batch_size=nn["batch_size"],
                      patience=nn["patience"], seed=RANDOM_SEED, balanced=balanced),
            X_holdout,
        )
        for thr_label, thr in (("thr=0.50", 0.50),
                                (f"thr=tuned({tuned[('Neural Network (MLP)', balanced)]:.2f})",
                                 tuned[("Neural Network (MLP)", balanced)])):
            preds = (probs >= thr).astype(int)
            label = f"Neural Network (MLP, rep. seed) (balanced={balanced}, {thr_label})"
            ablation_rows.append(compute_metrics_ci(label, y_holdout, preds, seed=RANDOM_SEED))

    print_table("HOLDOUT — class balancing x threshold ablation", HEADER,
                [fmt_row_ci(r) for r in ablation_rows])

    # ── STEP 8: Vectorized Backtesting  (long-only, frictionless, on the HOLDOUT) ─
    _banner(8, "Vectorized Backtesting  (long-only, frictionless, HOLDOUT only)")
    print("      NOTE: backtest realism (execution lag, costs, benchmark risk metrics) is Phase 3 —")
    print("      this step only makes the SIGNAL SOURCE honest (holdout, canonical config).")

    fwd_ret_h = fwd_ret_all[holdout_idx]
    al_model_h = train_always_long(X_train_h, y_train_h)
    al_preds_h = predict_always_long(al_model_h, X_holdout)
    pm_model_h = train_prev_day_momentum(X_train_h, y_train_h)
    pm_preds_h = predict_prev_day_momentum(pm_model_h, dr_h)
    sh_preds_h = predict_shuffled_label(train_shuffled_label(X_train_h, y_train_h, seed=RANDOM_SEED), X_holdout)
    nb_preds_h = (predict_proba_naive_bayes(
        train_naive_bayes(X_train_h, y_train_h, balanced=CANONICAL_BALANCED), X_holdout,
    ) >= nb_t).astype(int)
    lr_preds_h = predict_logistic(
        train_logistic(X_train_h, y_train_h, balanced=CANONICAL_BALANCED), X_holdout, threshold=lr_t,
    )

    backtest_models = [
        ("always_long",               al_preds_h),
        ("prev_day_momentum",         pm_preds_h),
        ("shuffled_label",            sh_preds_h),
        ("Naive Bayes",               nb_preds_h),
        ("Logistic Regression",       lr_preds_h),
        ("Neural Network (MLP, representative seed)", mlp_preds_h),
    ]
    for name, preds in backtest_models:
        result = run_backtest(preds, fwd_ret_h, bt["risk_free_rate"])
        print_backtest_report(name, result)

    # ── STEP 9: Visualisation ─────────────────────────────────────────────────
    _banner(9, f"Visualisation  (saving plots to '{PLOTS_DIR}/', HOLDOUT predictions)")
    bt_list = [run_backtest(nb_preds_h, fwd_ret_h, bt["risk_free_rate"]),
               run_backtest(lr_preds_h, fwd_ret_h, bt["risk_free_rate"]),
               run_backtest(mlp_preds_h, fwd_ret_h, bt["risk_free_rate"])]
    labels = ["Naive Bayes", "Logistic Regression", "Neural Network"]

    plot_equity_curves(bt_list, labels, PLOTS_DIR)
    plot_drawdown(bt_list, labels, PLOTS_DIR)
    for preds, name in zip([nb_preds_h, lr_preds_h, mlp_preds_h], labels):
        plot_confusion_matrix(y_holdout, preds, name, PLOTS_DIR)

    if holdout_leak:
        print("\n  >>> A LEAK WAS DETECTED ON THE HOLDOUT ITSELF. <<<")
        print("  >>> This is the most serious possible finding — report it before anything else. <<<\n")

    print(f"\n{'═' * 66}")
    print("  PIPELINE COMPLETE")
    print(f"{'═' * 66}\n")


if __name__ == "__main__":
    main()

"""Stage 5 — Reporting & Plot Generation.

Generates publication-quality Matplotlib / Seaborn plots and saves them
to the configured plots/ directory.  Uses the "Agg" non-interactive
backend so the pipeline runs headlessly (no display required).

Plots produced
──────────────
  equity_curves.png        — cumulative strategy returns vs. buy-and-hold
  confusion_matrix_*.png   — per-model heatmap of TP / FP / TN / FN
  drawdown.png             — rolling peak-to-trough drawdown by strategy
"""

import os

import matplotlib

matplotlib.use("Agg")   # must be set before importing pyplot

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.05)
STRATEGY_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#9b5de5"]
BENCHMARK_COLOR = "#457b9d"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_equity_curves(
    backtest_results: list,
    model_names:      list,
    plots_dir:        str = "plots",
) -> None:
    _ensure_dir(plots_dir)
    fig, ax = plt.subplots(figsize=(13, 6))

    bm = backtest_results[0]["cum_benchmark"]
    ax.plot(bm.values, label="Buy & Hold", color=BENCHMARK_COLOR,
            linewidth=2.5, linestyle="--", alpha=0.85)

    for i, (result, name) in enumerate(zip(backtest_results, model_names)):
        strat = result["cum_strategy"]
        ax.plot(
            strat.values,
            label=f"{name}  (ret={result['total_return']:+.1f}%  SR={result['sharpe']:.2f})",
            color=STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
            linewidth=2,
        )

    ax.set_title("Equity Curve — ML Long-Only Strategy vs. Buy & Hold",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Test Day", fontsize=11)
    ax.set_ylabel("Portfolio Value  (normalised to 1.0)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}×"))
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(plots_dir, "equity_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {path}")


def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str = "Model",
    plots_dir:  str = "plots",
) -> None:
    _ensure_dir(plots_dir)
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=np.array([[f"{v}\n({p:.1%})" for v, p in zip(row_c, row_n)]
                        for row_c, row_n in zip(cm, cm_norm)]),
        fmt="", cmap="Blues", vmin=0, vmax=1, linewidths=0.5,
        xticklabels=["Pred DOWN", "Pred UP"],
        yticklabels=["True DOWN", "True UP"],
        ax=ax, cbar_kws={"label": "Row-normalised rate"},
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label",      fontsize=10)

    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(plots_dir, f"confusion_matrix_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {path}")


def plot_drawdown(
    backtest_results: list,
    model_names:      list,
    plots_dir:        str = "plots",
) -> None:
    _ensure_dir(plots_dir)
    fig, ax = plt.subplots(figsize=(13, 4))

    for i, (result, name) in enumerate(zip(backtest_results, model_names)):
        cum   = result["cum_strategy"].values
        peak  = np.maximum.accumulate(cum)
        dd    = (cum - peak) / (peak + 1e-10) * 100.0
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        ax.fill_between(range(len(dd)), dd, 0, alpha=0.25, color=color)
        ax.plot(dd, label=f"{name}  (max={result['max_drawdown']:.1f}%)",
                color=color, linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Rolling Drawdown by Strategy", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Test Day",     fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(plots_dir, "drawdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {path}")

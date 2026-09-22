"""Stage 7c — Volatility-Regime Conditional Accuracy.

Tests whether any model's accuracy varies systematically with the market's
realized volatility level. Uses 20-day realized volatility (rolling std of
daily returns) computed on the full series, then extracted at holdout dates
only, and split into quartiles. Regimes are defined entirely from price data
— no model information is used to choose them.

The key question: even if no model beats always_long on average, does any
model show conditional skill in a specific volatility regime?
"""

import numpy as np
import pandas as pd


def holdout_realized_vol(
    daily_returns: np.ndarray,
    holdout_idx:   np.ndarray,
    window:        int = 20,
) -> np.ndarray:
    """
    20-day rolling std of daily returns, extracted at holdout indices.

    Computed on the full series so the rolling window is fully warm at the
    first holdout row. The holdout typically begins 8+ years into the data,
    so NaN warm-up rows are never in the holdout window in practice, but
    the caller should handle them defensively anyway.
    """
    rv = pd.Series(daily_returns).rolling(window=window, min_periods=window).std()
    return rv.iloc[holdout_idx].values


def regime_report(
    y_true:      np.ndarray,
    predictions: dict,
    rv:          np.ndarray,
    n_quartiles: int = 4,
) -> tuple:
    """
    Split holdout rows by realized vol quartile and compute per-regime accuracy.

    NaN rv values (warm-up rows that somehow appear in the holdout) are placed
    in Q1 rather than silently dropped — conservative and transparent.

    Returns
    -------
    rows          : list of dicts, one per model
    quartile_edges: inner bin boundaries (n_quartiles - 1 values)
    """
    rv_s     = pd.Series(rv)
    n_nan    = int(rv_s.isna().sum())
    rv_clean = rv_s.fillna(rv_s.min())  # NaN → Q1 (lowest vol)

    labels, bins = pd.qcut(rv_clean, q=n_quartiles, labels=False, retbins=True, duplicates="drop")
    labels        = np.asarray(labels, dtype=int)
    quartile_edges = bins[1:-1]

    rows = []
    for name, preds in predictions.items():
        row = {"Model": name, "_n_nan_rv": n_nan}
        for q in range(n_quartiles):
            mask = labels == q
            row[f"Q{q + 1}"]   = float((preds[mask] == y_true[mask]).mean()) if mask.sum() else float("nan")
            row[f"Q{q + 1}_n"] = int(mask.sum())
        row["Overall"] = float((preds == y_true).mean())
        rows.append(row)

    return rows, quartile_edges


def beats_always_long(rows: list, n_quartiles: int = 4) -> dict:
    """
    For each non-baseline model, return the quartiles where it beats always_long.
    Returns {model_name: [quartile_labels_where_it_beats]}.
    """
    al_row = next((r for r in rows if r["Model"] == "always_long"), None)
    if al_row is None:
        return {}

    result = {}
    for row in rows:
        if row["Model"] == "always_long":
            continue
        winning_qs = []
        for q in range(n_quartiles):
            key = f"Q{q + 1}"
            v, al_v = row.get(key, float("nan")), al_row.get(key, float("nan"))
            if not (np.isnan(v) or np.isnan(al_v)) and v > al_v:
                winning_qs.append(f"Q{q + 1}")
        result[row["Model"]] = winning_qs
    return result


def print_regime_table(
    rows:           list,
    quartile_edges: np.ndarray,
    n_quartiles:    int = 4,
) -> None:
    q_labels = [
        "Q1 (Low vol)" if q == 0
        else f"Q{n_quartiles} (High vol)" if q == n_quartiles - 1
        else f"Q{q + 1}"
        for q in range(n_quartiles)
    ]
    header   = ["Model"] + q_labels + ["Overall"]
    col_w    = [42] + [13] * n_quartiles + [10]

    def _pct(v: float) -> str:
        return "N/A" if np.isnan(v) else f"{v * 100:.2f}%"

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    hdr = "|" + "|".join(h.center(w) for h, w in zip(header, col_w)) + "|"
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        cells = [(" " + r["Model"]).ljust(col_w[0])]
        for q in range(n_quartiles):
            cells.append(_pct(r[f"Q{q + 1}"]).center(col_w[q + 1]))
        cells.append(_pct(r["Overall"]).center(col_w[-1]))
        print("|" + "|".join(cells) + "|")
    print(sep)

    n_per_q    = [rows[0][f"Q{q + 1}_n"] for q in range(n_quartiles)]
    edge_strs  = ", ".join(f"{e * 100:.3f}%" for e in quartile_edges)
    print(f"      N per quartile : {' | '.join(str(n) for n in n_per_q)}")
    print(f"      Vol quartile edges (20-day realized vol, annualized) : {edge_strs}")
    if rows and rows[0]["_n_nan_rv"] > 0:
        print(f"      Note: {rows[0]['_n_nan_rv']} NaN realized-vol rows placed in Q1.")

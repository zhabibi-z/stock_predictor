"""Stage 4 — Financial Vector Backtesting.

Maps binary UP/DOWN signals to a long-only trading strategy and computes
true portfolio performance metrics against a buy-and-hold benchmark.

Strategy definition
───────────────────
  signal[T] is computed from data available at the close of day T. It
  cannot be acted on until T+1 at the earliest — `execution_lag` (default
  1 bar) enforces that:

    executed[T] = signal[T - execution_lag]
    strategy_return[T] = executed[T] * Fwd_Return[T]
                         + (1 - executed[T]) * daily_rf     (cash earns rf)
                         - cost_drag[T]                     (per side, on
                                                              position changes)

`costs_bps` charges a fixed drag each time the position changes (an entry
or an exit — "per side"), not on every day held. With `costs_bps=0` and
`execution_lag=0` this collapses to the original frictionless same-bar
model; neither is the default.

Metrics computed (for both the strategy AND the benchmark, so they are
comparable in the same table)
────────────────
  Total Return   — total compounded portfolio growth
  Sharpe Ratio   — annualised excess-return / std-dev, excess over cash
  Max Drawdown   — worst peak-to-trough equity drop (%), measured from an
                   equity curve that starts at 1.0 (inception), not 1+r0
  Calmar Ratio   — CAGR / |max drawdown|

Also reported: win rate (fraction of long days that beat what cash would
have earned), days_long (days actually holding a position, post-lag), and
n_round_trips (entries counted from signal transitions — not the same
number as days_long, which double-counts every day of a multi-day hold).
"""

import numpy as np
import pandas as pd


def purged_walk_forward_splits(
    n_samples: int,
    n_splits:  int = 5,
    purge_gap: int = 5,
) -> list:
    """
    Expanding-window walk-forward splits with a purge-gap buffer.

    A `purge_gap`-day buffer between the end of training and the start of
    testing prevents serial-autocorrelation from bleeding across the boundary.
    Returns list of (train_idx, test_idx) numpy arrays.
    """
    fold_size = n_samples // (n_splits + 1)
    splits    = []
    for i in range(1, n_splits + 1):
        train_end  = i * fold_size
        test_start = train_end + purge_gap
        test_end   = min(test_start + fold_size, n_samples)
        if test_end > test_start:
            splits.append((
                np.arange(0, train_end),
                np.arange(test_start, test_end),
            ))
    return splits


def three_way_split(
    dates,
    holdout_start:    str,
    validation_frac:  float,
    purge_gap:        int = 5,
) -> tuple:
    """
    Partition a date-sorted index into (train_pool_idx, validation_idx, holdout_idx).

    holdout_idx    — every row on/after `holdout_start`. Touched exactly once, at
                     the end of the pipeline. Never used for tuning of any kind.
    validation_idx — the tail `validation_frac` of the rows strictly before the
                     holdout, reserved for threshold/hyperparameter tuning.
                     Excluded from the walk-forward CV pool below.
    train_pool_idx — everything else. The purged walk-forward CV folds used for
                     the walk-forward summary are drawn only from this pool.

    A `purge_gap`-row buffer separates train_pool from validation, and separates
    validation from the holdout, matching the buffer already enforced between
    walk-forward train/test folds. When train_pool and validation are later
    pooled to fit the final holdout model, no internal purge is needed between
    them — purge gaps only matter at a boundary into data that will be
    evaluated out-of-sample.
    """
    dates = pd.DatetimeIndex(dates)
    holdout_start_pos = int(np.searchsorted(dates.values, pd.Timestamp(holdout_start).to_datetime64()))

    pre_holdout_end = max(holdout_start_pos - purge_gap, 0)
    val_n           = int(round(pre_holdout_end * validation_frac))
    train_pool_end  = max(pre_holdout_end - val_n - purge_gap, 0)

    train_pool_idx = np.arange(0, train_pool_end)
    validation_idx = np.arange(train_pool_end + purge_gap, pre_holdout_end)
    holdout_idx    = np.arange(holdout_start_pos, len(dates))
    return train_pool_idx, validation_idx, holdout_idx


def _curve_stats(cum_curve: np.ndarray, period_returns: np.ndarray,
                  risk_free_rate: float, trading_days: int) -> dict:
    """Sharpe, max drawdown, and Calmar for one equity curve that starts at 1.0."""
    daily_rf = risk_free_rate / trading_days
    excess   = period_returns - daily_rf
    sharpe   = (excess.mean() / (excess.std() + 1e-10)) * np.sqrt(trading_days)

    rolling_peak = np.maximum.accumulate(cum_curve)
    drawdown     = (cum_curve - rolling_peak) / (rolling_peak + 1e-10)
    max_dd       = float(drawdown.min())

    n_periods = len(period_returns)
    years     = n_periods / trading_days
    end_value = float(cum_curve[-1])
    cagr      = end_value ** (1.0 / years) - 1.0 if years > 0 and end_value > 0 else float("nan")
    calmar    = cagr / abs(max_dd) if max_dd < 0 and cagr == cagr else float("nan")

    return {
        "total_return": round((end_value - 1.0) * 100, 2),
        "sharpe":       round(float(sharpe), 4),
        "max_drawdown": round(max_dd * 100, 2),
        "calmar":       round(float(calmar), 4) if calmar == calmar else None,
    }


def run_backtest(
    signals:        np.ndarray,
    fwd_returns:    np.ndarray,
    risk_free_rate: float = 0.04,
    trading_days:   int   = 252,
    execution_lag:  int   = 1,
    costs_bps:      float = 0.0,
) -> dict:
    """
    Compute portfolio performance metrics for a long-only binary strategy.
    See module docstring for the strategy definition.

    Parameters
    ----------
    signals        : binary array  (1 = long, 0 = cash), computed from data
                     available at the close of each day
    fwd_returns    : actual one-day forward returns = (Close[T+1] − Close[T]) / Close[T]
    risk_free_rate : annualised risk-free rate; cash earns this, not 0
    trading_days   : trading days per year for annualisation
    execution_lag  : bars between signal and execution (default 1 — cannot
                     trade on the same close the signal was computed from)
    costs_bps      : cost per side (entry or exit) in basis points
    """
    signals     = np.asarray(signals,     dtype=float)
    fwd_returns = np.asarray(fwd_returns, dtype=float)
    n           = len(signals)
    daily_rf    = risk_free_rate / trading_days

    lag = max(int(execution_lag), 0)
    executed = np.concatenate([np.zeros(lag), signals])[:n] if lag > 0 else signals.copy()

    transitions = np.abs(np.diff(np.concatenate([[0.0], executed])))
    cost_drag   = (costs_bps / 10000.0) * transitions

    strategy_returns  = executed * fwd_returns + (1.0 - executed) * daily_rf - cost_drag
    benchmark_returns = fwd_returns  # buy-and-hold: always in, no lag, no cost, no cash leg

    cum_strategy  = np.concatenate([[1.0], np.cumprod(1.0 + strategy_returns)])
    cum_benchmark = np.concatenate([[1.0], np.cumprod(1.0 + benchmark_returns)])

    strat_stats = _curve_stats(cum_strategy,  strategy_returns,  risk_free_rate, trading_days)
    bench_stats = _curve_stats(cum_benchmark, benchmark_returns, risk_free_rate, trading_days)

    active_returns = strategy_returns[executed == 1]
    win_rate = float((active_returns > daily_rf).mean()) * 100.0 if len(active_returns) else 0.0

    days_long     = int(executed.sum())
    n_round_trips = int(np.sum(np.diff(np.concatenate([[0.0], executed])) == 1))

    return {
        "cum_strategy":  pd.Series(cum_strategy),
        "cum_benchmark": pd.Series(cum_benchmark),
        "strategy":      strat_stats,
        "benchmark":     bench_stats,
        "win_rate":      round(win_rate, 2),
        "days_long":     days_long,
        "n_round_trips": n_round_trips,
        "execution_lag": lag,
        "costs_bps":     costs_bps,
        # Flat aliases for existing callers (app.py, visualization.py) that
        # read the strategy's own numbers at the top level.
        "sharpe":           strat_stats["sharpe"],
        "max_drawdown":     strat_stats["max_drawdown"],
        "total_return":     strat_stats["total_return"],
        "benchmark_return": bench_stats["total_return"],
        "n_trades":         days_long,
    }


def cost_sensitivity(
    signals:        np.ndarray,
    fwd_returns:    np.ndarray,
    risk_free_rate: float = 0.04,
    trading_days:   int   = 252,
    execution_lag:  int   = 1,
    cost_grid:      tuple = (0, 1, 5, 10),
) -> list:
    """Re-run the backtest across a grid of per-side costs (bps) to show how
    fast the edge (if any) is eaten by realistic transaction costs."""
    rows = []
    for bps in cost_grid:
        r = run_backtest(signals, fwd_returns, risk_free_rate, trading_days,
                          execution_lag=execution_lag, costs_bps=bps)
        rows.append({"costs_bps": bps, "total_return": r["strategy"]["total_return"],
                     "sharpe": r["strategy"]["sharpe"]})
    return rows


def print_backtest_report(model_name: str, result: dict) -> None:
    s, b = result["strategy"], result["benchmark"]
    alpha = s["total_return"] - b["total_return"]
    print(f"\n  {'─' * 68}")
    print(f"  Backtest  │  {model_name}  "
          f"(lag={result['execution_lag']} bar, costs={result['costs_bps']:.0f}bps/side)")
    print(f"  {'─' * 68}")
    print(f"  {'Metric':<20}{'Strategy':>14}{'Benchmark (B&H)':>20}")
    print(f"  {'Total Return':<20}{s['total_return']:>13.2f}%{b['total_return']:>19.2f}%")
    print(f"  {'Sharpe Ratio':<20}{s['sharpe']:>14.4f}{b['sharpe']:>20.4f}")
    print(f"  {'Max Drawdown':<20}{s['max_drawdown']:>13.2f}%{b['max_drawdown']:>19.2f}%")
    calmar_s = f"{s['calmar']:.4f}" if s["calmar"] is not None else "n/a"
    calmar_b = f"{b['calmar']:.4f}" if b["calmar"] is not None else "n/a"
    print(f"  {'Calmar Ratio':<20}{calmar_s:>14}{calmar_b:>20}")
    print(f"  Alpha vs Benchmark : {alpha:>8.2f} %")
    print(f"  Win Rate (> cash)  : {result['win_rate']:>8.2f} %")
    print(f"  Days Long          : {result['days_long']:>8d}")
    print(f"  Round Trips        : {result['n_round_trips']:>8d}")


def print_cost_sensitivity(model_name: str, rows: list) -> None:
    print(f"\n  Cost sensitivity — {model_name}")
    print(f"  {'costs (bps/side)':<20}{'Total Return':>16}{'Sharpe':>12}")
    for r in rows:
        print(f"  {r['costs_bps']:<20}{r['total_return']:>15.2f}%{r['sharpe']:>12.4f}")

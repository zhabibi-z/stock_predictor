"""Stage 4 — Financial Vector Backtesting.

Maps binary UP/DOWN signals to a long-only trading strategy and computes
true portfolio performance metrics against a buy-and-hold benchmark.

Strategy definition
───────────────────
  signal[T] = 1  →  go long from close of T to close of T+1
  signal[T] = 0  →  hold cash (earn 0 %)

  strategy_return[T] = signal[T] × Fwd_Return[T]

This is a frictionless model (no slippage, no commission) so the ML
edge is the only variable under test.

Metrics computed
────────────────
  Cumulative Strategy Return  — total compounded portfolio growth
  Benchmark Return            — equivalent buy-and-hold over the same period
  Sharpe Ratio                — annualised excess-return / std-dev
  Max Drawdown                — worst peak-to-trough equity drop (%)
  Win Rate                    — fraction of long-days that were profitable
  Trades Taken                — number of long-days entered
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


def run_backtest(
    signals:        np.ndarray,
    fwd_returns:    np.ndarray,
    risk_free_rate: float = 0.04,
    trading_days:   int   = 252,
) -> dict:
    """
    Compute portfolio performance metrics for a long-only binary strategy.

    Parameters
    ----------
    signals        : binary array  (1 = long, 0 = cash)
    fwd_returns    : actual one-day forward returns = (Close[T+1] − Close[T]) / Close[T]
    risk_free_rate : annualised risk-free rate
    trading_days   : trading days per year for annualisation
    """
    signals     = np.asarray(signals,     dtype=float)
    fwd_returns = np.asarray(fwd_returns, dtype=float)

    strategy_returns  = signals * fwd_returns
    benchmark_returns = fwd_returns

    cum_strategy  = (1.0 + strategy_returns).cumprod()
    cum_benchmark = (1.0 + benchmark_returns).cumprod()

    daily_rf = risk_free_rate / trading_days
    excess   = strategy_returns - daily_rf
    sharpe   = (excess.mean() / (excess.std() + 1e-10)) * np.sqrt(trading_days)

    rolling_peak = np.maximum.accumulate(cum_strategy)
    drawdown     = (cum_strategy - rolling_peak) / (rolling_peak + 1e-10)
    max_drawdown = float(drawdown.min()) * 100.0

    active_returns = strategy_returns[signals == 1]
    win_rate = float((active_returns > 0).mean()) * 100.0 if len(active_returns) else 0.0

    return {
        "cum_strategy":      pd.Series(cum_strategy),
        "cum_benchmark":     pd.Series(cum_benchmark),
        "sharpe":            round(float(sharpe), 4),
        "max_drawdown":      round(max_drawdown, 2),
        "total_return":      round(float(cum_strategy[-1] - 1) * 100, 2),
        "benchmark_return":  round(float(cum_benchmark[-1] - 1) * 100, 2),
        "win_rate":          round(win_rate, 2),
        "n_trades":          int(signals.sum()),
    }


def print_backtest_report(model_name: str, result: dict) -> None:
    alpha = result["total_return"] - result["benchmark_return"]
    print(f"\n  {'─' * 56}")
    print(f"  Backtest  │  {model_name}")
    print(f"  {'─' * 56}")
    print(f"  Strategy Return    : {result['total_return']:>8.2f} %")
    print(f"  Benchmark Return   : {result['benchmark_return']:>8.2f} %")
    print(f"  Alpha vs Benchmark : {alpha:>8.2f} %")
    print(f"  Sharpe Ratio       : {result['sharpe']:>8.4f}")
    print(f"  Max Drawdown       : {result['max_drawdown']:>8.2f} %")
    print(f"  Win Rate           : {result['win_rate']:>8.2f} %")
    print(f"  Trades Taken       : {result['n_trades']:>8d}")

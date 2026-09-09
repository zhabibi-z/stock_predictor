"""
Tests for Deflated Sharpe Ratio and Probability of Backtest Overfitting.

DSR: the key property is monotonicity — the same return series must score
a LOWER deflated Sharpe as n_trials grows, since more trials raise the
luck-adjusted bar the observed Sharpe has to clear.

PBO: a configuration that is genuinely best on every block must score
PBO ~ 0 (no overfitting signature); performance matrices with no real
skill differences (pure block-level noise) must score PBO close to 0.5 —
the in-sample winner is then no better than a coin flip out-of-sample.
"""

import numpy as np

from src.dsr import deflated_sharpe_ratio, probability_of_backtest_overfitting, sharpe_ratio_stats


def test_sharpe_ratio_stats_on_normal_returns():
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.001, scale=0.01, size=2000)
    stats = sharpe_ratio_stats(returns)
    assert stats["T"] == 2000
    assert abs(stats["skew"]) < 0.3
    assert abs(stats["kurtosis"] - 3.0) < 0.5  # Pearson kurtosis of a normal ~ 3


def test_dsr_decreases_as_n_trials_grows():
    rng = np.random.default_rng(1)
    returns = rng.normal(loc=0.0008, scale=0.01, size=410)  # a modest positive-Sharpe series

    dsr_1   = deflated_sharpe_ratio(returns, n_trials=1)
    dsr_6   = deflated_sharpe_ratio(returns, n_trials=6)
    dsr_20  = deflated_sharpe_ratio(returns, n_trials=20)

    assert dsr_1["sr0"] == 0.0  # no correction with a single trial
    assert dsr_6["sr0"] > dsr_1["sr0"]
    assert dsr_20["sr0"] > dsr_6["sr0"]
    assert dsr_1["dsr"] >= dsr_6["dsr"] >= dsr_20["dsr"], (
        "DSR must not increase as more trials are counted against the same return series"
    )


def test_dsr_same_sharpe_higher_n_trials_can_flip_significance():
    """The whole point of DSR: a Sharpe that looks significant at n_trials=1
    can stop looking significant once the real number of trials is counted."""
    rng = np.random.default_rng(2)
    returns = rng.normal(loc=0.0015, scale=0.01, size=410)  # a clear, realized positive edge
    stats = sharpe_ratio_stats(returns)
    assert stats["sharpe"] > 0.1, f"test setup expects a clearly positive realized Sharpe, got {stats['sharpe']}"

    dsr_1  = deflated_sharpe_ratio(returns, n_trials=1)
    dsr_50 = deflated_sharpe_ratio(returns, n_trials=50)
    assert dsr_1["dsr"] > 0.9
    assert dsr_50["dsr"] < dsr_1["dsr"]


def test_pbo_zero_when_one_config_dominates_every_block():
    rng = np.random.default_rng(0)
    S, N = 10, 5
    base = rng.normal(size=(S, N)) * 0.01
    base[:, 0] += 1.0  # config 0 is always far ahead, on every block
    result = probability_of_backtest_overfitting(base)
    assert result["pbo"] < 0.05


def test_pbo_near_half_when_performance_is_pure_noise():
    """
    With no real config differences, the in-sample winner should be no
    better than a coin flip out-of-sample. CSCV's bipartitions overlap
    heavily (they are not independent draws), so a single moderate-S
    matrix can land anywhere in a fairly wide band around 0.5 by chance —
    the qualitative property that matters is "not near 0 or 1" (which is
    what a real, exploitable performance difference would produce, per
    the dominant-config test above). Average several seeds to keep this
    from being a coin flip itself.
    """
    S, N = 14, 6
    pbos = []
    for seed in range(8):
        perf = np.random.default_rng(seed).normal(size=(S, N))
        pbos.append(probability_of_backtest_overfitting(perf)["pbo"])
    mean_pbo = float(np.mean(pbos))
    assert 0.3 < mean_pbo < 0.7, f"expected pure noise to average near 0.5 PBO, got {mean_pbo}"

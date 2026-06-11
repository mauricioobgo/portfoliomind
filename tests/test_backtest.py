"""Hermetic tests for :mod:`portfoliomind.backtest`.

All series are synthetic; ``fetch`` is injected. No yfinance, no network.
"""

from __future__ import annotations

import math

import pytest

from portfoliomind.backtest import (
    BacktestResult,
    backtest_closes,
    backtest_ticker,
    backtest_universe,
)
from portfoliomind.backtest.engine import UniverseBacktest


# --- Synthetic series ---------------------------------------------------------


def oscillating(n: int = 400, drift: float = 0.0008, amp: float = 0.02, period: float = 9.0) -> list[float]:
    """A drifting sine wave — generates many pattern setups that resolve
    both ways, so win rate and calibration are non-trivial."""
    base = 100.0
    out = []
    for i in range(n):
        base *= 1.0 + drift + amp * math.sin(i / period)
        out.append(base)
    return out


def steady_rise(n: int = 300) -> list[float]:
    return [100.0 + 0.5 * i for i in range(n)]


# --- backtest_closes ----------------------------------------------------------


def test_insufficient_history_returns_zero_trades():
    res = backtest_closes("TEST", [100.0] * 20)
    assert isinstance(res, BacktestResult)
    assert res.n_trades == 0
    assert "insufficient" in res.note


def test_empty_series_never_raises():
    res = backtest_closes("TEST", [])
    assert res.n_trades == 0


def test_oscillating_series_produces_trades_and_stats():
    res = backtest_closes("TEST", oscillating())
    assert res.n_trades > 0
    assert 0.0 <= res.win_rate <= 1.0
    assert res.n_wins <= res.n_trades
    # Calibration gap is claimed p minus realized win rate.
    assert res.calibration_gap == pytest.approx(res.avg_p_bullish - res.win_rate, abs=1e-9)
    # Each trade carries provenance.
    for t in res.trades:
        assert t.exit_reason in ("tp", "sl", "timeout")
        assert t.patterns
        assert t.exit_index > t.entry_index


def test_positions_do_not_overlap():
    res = backtest_closes("TEST", oscillating())
    prev_exit = -1
    for t in res.trades:
        assert t.entry_index > prev_exit
        prev_exit = t.exit_index


def test_per_pattern_breakdown_present():
    res = backtest_closes("TEST", oscillating())
    assert res.per_pattern
    for name, (cnt, rate) in res.per_pattern.items():
        assert cnt >= 1
        assert 0.0 <= rate <= 1.0


def test_steady_rise_mostly_hits_targets():
    """In a clean uptrend, exits should skew to take-profit and the
    strategy should make money."""
    res = backtest_closes("TEST", steady_rise(), max_hold=40)
    assert res.n_trades > 0
    assert res.total_return > 0
    tp_exits = sum(1 for t in res.trades if t.exit_reason == "tp")
    assert tp_exits >= 1


def test_max_drawdown_is_bounded():
    res = backtest_closes("TEST", oscillating())
    assert 0.0 <= res.max_drawdown <= 1.0


def test_profit_factor_none_when_no_losses():
    res = backtest_closes("TEST", steady_rise(), max_hold=60)
    # A pure uptrend may produce zero losing trades → profit_factor None.
    if all(t.ret >= 0 for t in res.trades):
        assert res.profit_factor is None


def test_supported_predicate():
    res = backtest_closes("TEST", steady_rise(), max_hold=40)
    assert res.supported(min_trades=1) is (res.n_trades >= 1 and res.expectancy > 0)


def test_summary_line_is_readable():
    res = backtest_closes("AAPL", oscillating())
    line = res.summary_line()
    assert "AAPL" in line
    assert "win_rate" in line


# --- backtest_ticker (fetch injection) ----------------------------------------


def test_backtest_ticker_with_injected_fetch():
    series = oscillating()

    def fetch(ticker, period="2y"):
        return series

    res = backtest_ticker("NVDA", fetch=fetch)
    assert res.ticker == "NVDA"
    assert res.n_trades > 0


def test_backtest_ticker_fetch_without_period_kwarg():
    """An injected fetch that doesn't accept ``period`` still works."""
    series = oscillating()

    def fetch(ticker):  # no period kwarg
        return series

    res = backtest_ticker("NVDA", fetch=fetch)
    assert res.n_trades > 0


def test_backtest_ticker_fetch_failure_is_noted():
    def fetch(ticker, period="2y"):
        raise RuntimeError("yfinance down")

    res = backtest_ticker("NVDA", fetch=fetch)
    assert res.n_trades == 0
    assert "fetch failed" in res.note


def test_backtest_ticker_empty_fetch():
    res = backtest_ticker("NVDA", fetch=lambda t, period="2y": [])
    assert res.n_trades == 0


# --- backtest_universe --------------------------------------------------------


def test_backtest_universe_pools_trades():
    series = {"AAA": oscillating(), "BBB": steady_rise(), "CCC": oscillating(drift=0.001)}

    def fetch(ticker, period="2y"):
        return series.get(ticker, [])

    sweep = backtest_universe(("AAA", "BBB", "CCC"), fetch=fetch)
    assert isinstance(sweep, UniverseBacktest)
    assert set(sweep.results) == {"AAA", "BBB", "CCC"}
    pooled = sum(r.n_trades for r in sweep.results.values())
    assert sweep.n_trades == pooled
    assert 0.0 <= sweep.win_rate <= 1.0
    d = sweep.to_dict()
    assert d["tickers"] == 3
    assert "per_ticker" in d


def test_backtest_universe_all_empty():
    sweep = backtest_universe(("AAA", "BBB"), fetch=lambda t, period="2y": [])
    assert sweep.n_trades == 0
    assert sweep.win_rate == 0.0

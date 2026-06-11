"""Backtesting (card 10): walk-forward validation of the bullish strategy.

Public surface::

    from portfoliomind.backtest import (
        BacktestResult,
        BacktestTrade,
        UniverseBacktest,
        backtest_closes,    # pure: takes a list of closes
        backtest_ticker,    # fetches then backtests one ticker
        backtest_universe,  # sweeps the whole universe
    )

The headline metric is ``calibration_gap`` — how far the model's
claimed ``p_bullish`` sat above the realized win rate. It is the
empirical check on the pattern hit-rate priors in
:mod:`portfoliomind.signals.patterns`, and the independent validator
(:mod:`portfoliomind.validation`) uses it as a hard gate before any
trade reaches the user.
"""

from __future__ import annotations

from .engine import (
    DEFAULT_ENTRY_P,
    DEFAULT_MAX_HOLD,
    DEFAULT_PERIOD,
    BacktestResult,
    BacktestTrade,
    UniverseBacktest,
    backtest_closes,
    backtest_ticker,
    backtest_universe,
)

__all__ = [
    "DEFAULT_ENTRY_P",
    "DEFAULT_MAX_HOLD",
    "DEFAULT_PERIOD",
    "BacktestResult",
    "BacktestTrade",
    "UniverseBacktest",
    "backtest_closes",
    "backtest_ticker",
    "backtest_universe",
]

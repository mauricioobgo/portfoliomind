"""Strategy signals — technical + news sentiment → single score (card 6).

Public surface:

* :func:`score_ticker` — single-ticker entry point. **Never raises**;
  failures are returned as :class:`Signal` with ``error`` set.
* :func:`score_universe` — full 45-ticker universe, sorted by
  ``combined`` descending.
* :class:`Signal` / :class:`TechnicalScore` — the dataclasses cards
  7/8 will import.
* :func:`compute_technical_score` — pure-math technical score from a
  list of closes. Useful for backtests / unit tests.
* :class:`TechnicalCache` — same-day idempotency layer.

The technical layer is yfinance (6 months of daily bars); the
sentiment layer is card 5's :func:`score_ticker_sentiment`. The two
are combined with weights 0.6 / 0.4 (technicals dominate) and the
result is bounded to [-1, +1]. Confidence is the magnitude of the
combined signal times the agreement of the two sub-scores — a
signal where the components contradict each other has near-zero
confidence and should be filtered out before any operator-facing
ping.

Idempotency: a re-run in the same Bogota day returns identical
:class:`Signal` objects for every ticker (cache hit). A run after
midnight Bogota triggers a fresh yfinance fetch + (if not cached)
a fresh LLM sentiment call.

Design constraints (from the card 6 spec):

* No raw OHLCV at INFO log level; DEBUG only.
* **Never raise** from :func:`score_ticker` / :func:`score_universe`.
* yfinance is the only network call; one batched fetch per ticker
  per day (cache hit on re-runs).
* yfinance returning insufficient history → ``Signal(error="insufficient
  history")`` with ``combined=0.0``.
"""

from __future__ import annotations

from .cache import TechnicalCache
from .combined import (
    Candidate,
    score_universe as score_bullish_universe,
)
from .combiner import (
    MIN_HISTORY_BARS,
    WEIGHT_SENTIMENT,
    WEIGHT_TECHNICAL,
    Signal,
    SignalError,
    score_ticker,
    score_universe,
)
from .patterns import (
    BullishPatterns,
    PatternHit,
    detect_bullish_patterns,
)
from .sizer import (
    PositionSizer,
    SizingError,
    TradeOrder,
)
from .technicals import (
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    RSI_PERIOD,
    SMA_FAST,
    SMA_SLOW,
    VOL_EXPANSION_RATIO,
    VOL_LONG,
    VOL_SHORT,
    WEIGHT_MOMENTUM,
    WEIGHT_TREND,
    WEIGHT_VOLATILITY,
    TechnicalScore,
    TechnicalsError,
    compute_technical_score,
    fetch_ohlcv,
    realized_vol,
    rsi,
    sma,
)

__all__ = [
    # Public dataclasses
    "Signal",
    "SignalError",
    "TechnicalScore",
    "TechnicalsError",
    "Candidate",
    "BullishPatterns",
    "PatternHit",
    "TradeOrder",
    "PositionSizer",
    "SizingError",
    # Public functions
    "score_ticker",
    "score_universe",
    "score_bullish_universe",
    "detect_bullish_patterns",
    "compute_technical_score",
    "fetch_ohlcv",
    # Cache
    "TechnicalCache",
    # Pure helpers (re-exported for backtests / unit tests)
    "sma",
    "rsi",
    "realized_vol",
    # Weights + windows (so card 7 can introspect)
    "WEIGHT_TECHNICAL",
    "WEIGHT_SENTIMENT",
    "WEIGHT_TREND",
    "WEIGHT_MOMENTUM",
    "WEIGHT_VOLATILITY",
    "SMA_FAST",
    "SMA_SLOW",
    "RSI_PERIOD",
    "RSI_OVERSOLD",
    "RSI_OVERBOUGHT",
    "VOL_SHORT",
    "VOL_LONG",
    "VOL_EXPANSION_RATIO",
    "MIN_HISTORY_BARS",
]

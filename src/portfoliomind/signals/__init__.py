"""Technical indicators + combined bullish+news signal ranking (card 6).

Public surface:

* :func:`compute_technical_signal` — pull OHLCV for a ticker, compute the
  4 indicator booleans, return a :class:`TechnicalSignal`. Cached on
  disk for 1 hour via :class:`PriceCache`.
* :func:`score_universe` — iterate the universe, keep only tickers that
  pass the AND-of-two gate (>=2 of 4 technical bullish AND news
  sentiment > +0.2), return the ranked top-N :class:`Candidate` list.
* :class:`TechnicalSignal` — the 4 bools + their underlying numbers.
* :class:`Candidate` — the public contract for card 7 (sizer + Discord
  approval). Fields: ticker, strategy, timeframe, entry_price,
  technical_score (0-1), news_score (-1, +1), combined_score,
  top_signal_reason.

Design constraints (from the card 6 spec):

* **yfinance is slow and rate-limited.** Every price pull is cached on
  disk in the same SQLite pattern as ``portfoliomind.news.store``; a
  re-run in the same hour skips the network.
* **Fail-soft on a single ticker.** If yfinance errors out for one
  ticker, log at WARNING and skip — never break the whole universe.
* **No look-ahead bias.** Indicators at time ``T`` use only data up
  to ``T``; the cache key is ``(ticker, as_of_date)`` so a re-run with
  a different ``as_of_date`` re-computes.
* **The "at least 2 of 4" technical gate is intentionally loose.** It
  is the COMBINED signal that is the gate, not the technical pattern
  alone. 3+ technical bullish + positive news is high conviction;
  2/4 + positive news is moderate; 1/4 + positive news is dropped.
* **Daily bars only.** No 1-minute or intraday. The morning run fires
  at 8:30 Colombia = 9:30 ET — the open is just starting, so we use
  yesterday's close as the "entry price" reference.

Weighting (operator-preference):

* Technicals carry 0.6 of the combined score.
* News sentiment carries 0.4 of the combined score.

This is the operator's stated preference: technicals (price action) are
the primary signal, news is a confirmation filter. The weights are
documented in :mod:`portfoliomind.signals.combined`.
"""

from __future__ import annotations

from .combined import (
    Candidate,
    MIN_TECHNICAL_BULLISH,
    MIN_NEWS_SENTIMENT,
    STRATEGY,
    TIMEFRAME,
    WEIGHT_TECHNICAL,
    WEIGHT_NEWS,
    score_universe,
)
from .price_cache import DEFAULT_CACHE_PATH, PriceCache, PriceCacheError
from .technical import (
    TechnicalSignal,
    compute_technical_signal,
    indicator_buy,
    indicator_macd_bullish,
    indicator_rsi_not_overbought,
    indicator_sma_golden_cross,
    indicator_twenty_day_breakout,
)

__all__ = [
    # Public dataclasses
    "TechnicalSignal",
    "Candidate",
    # Public functions
    "compute_technical_signal",
    "score_universe",
    # Cache
    "PriceCache",
    "PriceCacheError",
    "DEFAULT_CACHE_PATH",
    # Indicator helpers (re-exported for testability)
    "indicator_sma_golden_cross",
    "indicator_twenty_day_breakout",
    "indicator_macd_bullish",
    "indicator_rsi_not_overbought",
    "indicator_buy",
    # Constants
    "MIN_TECHNICAL_BULLISH",
    "MIN_NEWS_SENTIMENT",
    "WEIGHT_TECHNICAL",
    "WEIGHT_NEWS",
    "STRATEGY",
    "TIMEFRAME",
]

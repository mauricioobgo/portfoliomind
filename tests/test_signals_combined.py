"""Hermetic tests for :mod:`portfoliomind.signals.combined`.

``fetch`` and ``sentiment_fn`` are always injected — no yfinance, no
OpenAI, no network.
"""

from __future__ import annotations

import pytest

from portfoliomind.signals.combined import (
    MIN_COMBINED,
    MIN_HISTORY_BARS,
    MIN_P_BULLISH,
    SENTIMENT_FLOOR,
    Candidate,
    score_universe,
)


# --- Synthetic series ---------------------------------------------------------


def bullish_series() -> list[float]:
    """Recent golden cross + uptrend stack — passes every gate."""
    return [160.0 - i for i in range(60)] + [101.0 + 2.0 * (i + 1) for i in range(20)]


def bearish_series() -> list[float]:
    return [200.0 - 0.5 * i for i in range(120)]


def make_fetch(series_by_ticker: dict[str, list[float]]):
    def fetch(ticker: str) -> list[float]:
        return series_by_ticker.get(ticker, [])

    return fetch


# --- Gates ------------------------------------------------------------------------


def test_bullish_ticker_qualifies():
    out = score_universe(
        top_n=5,
        tickers=("AAPL",),
        fetch=make_fetch({"AAPL": bullish_series()}),
        sentiment_fn=lambda t: 0.4,
    )
    assert len(out) == 1
    c = out[0]
    assert isinstance(c, Candidate)
    assert c.ticker == "AAPL"
    assert c.technical > 0
    assert c.p_bullish >= MIN_P_BULLISH
    assert c.sentiment == pytest.approx(0.4)
    assert c.combined >= MIN_COMBINED
    assert c.last_close == pytest.approx(bullish_series()[-1])
    assert c.vol_20d > 0
    assert c.patterns  # at least one named pattern
    assert c.reasons


def test_bearish_ticker_filtered_out():
    out = score_universe(
        top_n=5,
        tickers=("XYZ",),
        fetch=make_fetch({"XYZ": bearish_series()}),
        sentiment_fn=lambda t: 0.4,
    )
    assert out == []


def test_negative_news_vetoes_a_bullish_chart():
    out = score_universe(
        top_n=5,
        tickers=("AAPL",),
        fetch=make_fetch({"AAPL": bullish_series()}),
        sentiment_fn=lambda t: SENTIMENT_FLOOR - 0.2,
    )
    assert out == []


def test_neutral_news_still_passes():
    """sentiment exactly at the floor (0.0 = no news) must pass."""
    out = score_universe(
        top_n=5,
        tickers=("AAPL",),
        fetch=make_fetch({"AAPL": bullish_series()}),
        sentiment_fn=lambda t: 0.0,
    )
    assert len(out) == 1


def test_insufficient_history_is_skipped():
    out = score_universe(
        top_n=5,
        tickers=("NEW",),
        fetch=make_fetch({"NEW": [100.0] * (MIN_HISTORY_BARS - 1)}),
        sentiment_fn=lambda t: 0.5,
    )
    assert out == []


# --- Robustness ----------------------------------------------------------------------


def test_fetch_raising_never_propagates():
    def bad_fetch(ticker: str) -> list[float]:
        raise RuntimeError("yfinance down")

    out = score_universe(top_n=5, tickers=("AAPL",), fetch=bad_fetch, sentiment_fn=lambda t: 0.0)
    assert out == []


def test_sentiment_raising_defaults_to_zero():
    def bad_sentiment(ticker: str) -> float:
        raise RuntimeError("LLM down")

    out = score_universe(
        top_n=5,
        tickers=("AAPL",),
        fetch=make_fetch({"AAPL": bullish_series()}),
        sentiment_fn=bad_sentiment,
    )
    assert len(out) == 1
    assert out[0].sentiment == 0.0


def test_sentiment_is_clamped():
    out = score_universe(
        top_n=5,
        tickers=("AAPL",),
        fetch=make_fetch({"AAPL": bullish_series()}),
        sentiment_fn=lambda t: 7.0,
    )
    assert out[0].sentiment == 1.0


# --- Ordering + top_n -------------------------------------------------------------------


def test_sorted_by_combined_and_top_n_respected():
    series = {f"T{i}": bullish_series() for i in range(4)}
    sentiments = {"T0": 0.1, "T1": 0.9, "T2": 0.5, "T3": 0.3}
    out = score_universe(
        top_n=2,
        tickers=tuple(series),
        fetch=make_fetch(series),
        sentiment_fn=lambda t: sentiments[t],
    )
    assert len(out) == 2
    assert [c.ticker for c in out] == ["T1", "T2"]
    assert out[0].combined >= out[1].combined


def test_generator_tickers_are_materialized():
    """A generator input must not be silently exhausted mid-scan."""
    out = score_universe(
        top_n=5,
        tickers=(t for t in ("AAPL", "MSFT")),
        fetch=make_fetch({"AAPL": bullish_series(), "MSFT": bullish_series()}),
        sentiment_fn=lambda t: 0.2,
    )
    assert len(out) == 2


def test_strategy_runner_call_shape():
    """The strategy runner calls score_universe(top_n=N) with no other
    args — that exact call shape must not raise (network paths are
    injected away here by an empty universe)."""
    out = score_universe(top_n=3, tickers=(), fetch=make_fetch({}), sentiment_fn=lambda t: 0.0)
    assert out == []

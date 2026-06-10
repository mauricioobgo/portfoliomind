"""Unit tests for :mod:`portfoliomind.signals.combined` (card 6).

These tests are hermetic — they build fake :class:`TechnicalSignal`
objects and a fake sentiment map, then exercise the gate + ranking
logic in :func:`score_universe` without ever hitting yfinance, the
news feeds, or the OpenAI API.

The full ``score_universe`` path is exercised by mocking BOTH the
price-cache pull and the LLM sentiment call, so the test stays
fast and deterministic.
"""

from __future__ import annotations

import math
import os
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from portfoliomind.signals.combined import (
    MIN_NEWS_SENTIMENT,
    MIN_TECHNICAL_BULLISH,
    STRATEGY,
    TIMEFRAME,
    WEIGHT_NEWS,
    WEIGHT_TECHNICAL,
    Candidate,
    _build_candidate,
    _normalize_tickers,
    _top_signal_reason,
    score_universe,
)
from portfoliomind.signals.technical import TechnicalSignal
from portfoliomind.signals.price_cache import PriceCache


# --- Synthetic TechnicalSignal builder -------------------------------------


def _make_signal(
    ticker: str,
    *,
    sma_golden_cross: bool = False,
    twenty_day_breakout: bool = False,
    macd_bullish: bool = False,
    rsi_not_overbought: bool = True,
    close: float = 100.0,
    as_of_date: str = "2026-06-10",
) -> TechnicalSignal:
    bullish_count = int(sma_golden_cross) + int(twenty_day_breakout) + int(macd_bullish) + int(rsi_not_overbought)
    return TechnicalSignal(
        ticker=ticker,
        as_of_date=as_of_date,
        sma_golden_cross=sma_golden_cross,
        twenty_day_breakout=twenty_day_breakout,
        macd_bullish=macd_bullish,
        rsi_not_overbought=rsi_not_overbought,
        bullish_count=bullish_count,
        sma_50=100.0,
        sma_200=99.0,
        rsi_14=55.0,
        macd=0.5,
        macd_signal=0.3,
        close=close,
    )


# --- _normalize_tickers -----------------------------------------------------


class TestNormalizeTickers:
    def test_none_returns_full_universe(self):
        from portfoliomind.universe import UNIVERSE

        result = _normalize_tickers(None)
        assert result == UNIVERSE
        assert len(result) >= 40  # 15 ETFs + 30 stocks

    def test_uppercases_and_strips(self):
        result = _normalize_tickers(["  aapl ", "MSFT", "qqq"])
        assert result == ("AAPL", "MSFT", "QQQ")

    def test_dedupes_preserving_order(self):
        result = _normalize_tickers(["AAPL", "aapl", "MSFT", "AAPL"])
        assert result == ("AAPL", "MSFT")

    def test_empty_iterable(self):
        assert _normalize_tickers([]) == ()

    def test_empty_strings_dropped(self):
        result = _normalize_tickers(["", "  ", "AAPL"])
        assert result == ("AAPL",)


# --- _top_signal_reason -----------------------------------------------------


class TestTopSignalReason:
    def test_no_flags(self):
        sig = _make_signal("AAPL", rsi_not_overbought=False)
        # Even with 0 flags, RSI is False so bullish_count is 0.
        sig = _make_signal("AAPL")  # default has only RSI True → count=1
        # Actually default has rsi_not_overbought=True → bullish_count=1.
        # Force 0:
        sig = TechnicalSignal(
            ticker="AAPL", as_of_date="2026-06-10",
            sma_golden_cross=False, twenty_day_breakout=False,
            macd_bullish=False, rsi_not_overbought=False,
            bullish_count=0, sma_50=0, sma_200=0, rsi_14=0, macd=0, macd_signal=0, close=0,
        )
        assert _top_signal_reason(sig) == "no technical signal"

    def test_single_breakout(self):
        sig = _make_signal("AAPL", twenty_day_breakout=True, rsi_not_overbought=False)
        sig = TechnicalSignal(
            ticker="AAPL", as_of_date="2026-06-10",
            sma_golden_cross=False, twenty_day_breakout=True,
            macd_bullish=False, rsi_not_overbought=False,
            bullish_count=1, sma_50=0, sma_200=0, rsi_14=0, macd=0, macd_signal=0, close=0,
        )
        assert _top_signal_reason(sig) == "20-day high breakout"

    def test_three_flags_uses_confluence(self):
        sig = TechnicalSignal(
            ticker="AAPL", as_of_date="2026-06-10",
            sma_golden_cross=True, twenty_day_breakout=True,
            macd_bullish=True, rsi_not_overbought=False,
            bullish_count=3, sma_50=0, sma_200=0, rsi_14=0, macd=0, macd_signal=0, close=0,
        )
        assert "3/4 technical confluence" in _top_signal_reason(sig)

    def test_four_flags_uses_confluence(self):
        sig = TechnicalSignal(
            ticker="AAPL", as_of_date="2026-06-10",
            sma_golden_cross=True, twenty_day_breakout=True,
            macd_bullish=True, rsi_not_overbought=True,
            bullish_count=4, sma_50=0, sma_200=0, rsi_14=0, macd=0, macd_signal=0, close=0,
        )
        # bullish_count is 4.
        assert "4/4 technical confluence" in _top_signal_reason(sig)

    def test_two_flags_highlights_primary(self):
        sig = _make_signal(
            "AAPL",
            sma_golden_cross=True,
            twenty_day_breakout=True,
            rsi_not_overbought=False,
        )
        # Breakout is the priority; expect the breakout label + 1 other.
        msg = _top_signal_reason(sig)
        assert "20-day high breakout" in msg
        assert "1 other" in msg


# --- _build_candidate (the AND-of-two gate) --------------------------------


class TestBuildCandidate:
    def test_two_tech_and_positive_news_passes(self):
        sig = _make_signal(
            "AAPL",
            sma_golden_cross=True,
            twenty_day_breakout=True,
            rsi_not_overbought=True,
        )
        cand = _build_candidate(sig, news_score=0.5)
        assert cand is not None
        assert cand.ticker == "AAPL"
        assert cand.entry_price == 100.0
        assert cand.technical_score == pytest.approx(0.75)  # 3/4
        assert cand.news_score == 0.5
        # combined = 0.75*0.6 + 0.5*0.4 = 0.45 + 0.20 = 0.65
        assert cand.combined_score == pytest.approx(0.65)
        assert cand.strategy == STRATEGY
        assert cand.timeframe == TIMEFRAME

    def test_only_one_technical_fails_gate(self):
        sig = _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True)
        # bullish_count = 2 (SMA + RSI). Actually that's 2; let me make it 1.
        sig = _make_signal("AAPL", rsi_not_overbought=True)  # bullish_count=1
        assert _build_candidate(sig, news_score=0.5) is None

    def test_news_at_threshold_fails(self):
        # Threshold is STRICTLY greater than MIN_NEWS_SENTIMENT.
        sig = _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True)
        assert _build_candidate(sig, news_score=MIN_NEWS_SENTIMENT) is None

    def test_news_below_threshold_fails(self):
        sig = _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True)
        assert _build_candidate(sig, news_score=0.1) is None

    def test_negative_news_fails(self):
        sig = _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True)
        assert _build_candidate(sig, news_score=-0.5) is None

    def test_zero_news_fails(self):
        sig = _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True)
        assert _build_candidate(sig, news_score=0.0) is None

    def test_just_above_threshold_passes(self):
        sig = _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True)
        cand = _build_candidate(sig, news_score=MIN_NEWS_SENTIMENT + 0.01)
        assert cand is not None

    def test_two_tech_passes_with_zero_techs_below_threshold(self):
        # Two technical flags (e.g. breakout + RSI) passes the
        # technical floor, but if news is below threshold, it must
        # still fail.
        sig = _make_signal("AAPL", twenty_day_breakout=True, rsi_not_overbought=True)
        # bullish_count = 2, which meets the floor.
        assert sig.bullish_count == MIN_TECHNICAL_BULLISH
        # News is below threshold → fail.
        assert _build_candidate(sig, news_score=0.0) is None
        # News above threshold → pass.
        cand = _build_candidate(sig, news_score=0.3)
        assert cand is not None

    def test_combined_score_weights(self):
        # combined_score = technical_score * 0.6 + news_score * 0.4
        sig = _make_signal(
            "AAPL",
            sma_golden_cross=True,
            twenty_day_breakout=True,
            macd_bullish=True,
            rsi_not_overbought=True,
        )
        # 4/4 technical, score=1.0; news=+1.0; combined = 1.0.
        cand = _build_candidate(sig, news_score=1.0)
        assert cand is not None
        assert cand.combined_score == pytest.approx(1.0)

        # 2/4 technical, score=0.5; news=+1.0; combined = 0.5*0.6 + 1.0*0.4 = 0.70
        sig2 = _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True)
        cand2 = _build_candidate(sig2, news_score=1.0)
        assert cand2 is not None
        assert cand2.combined_score == pytest.approx(0.70)

        # 2/4 technical, score=0.5; news=+0.2 (just over threshold);
        # combined = 0.5*0.6 + 0.2*0.4 = 0.38
        cand3 = _build_candidate(sig2, news_score=0.21)
        assert cand3 is not None
        assert cand3.combined_score == pytest.approx(0.384)


# --- score_universe (full path with mocks) ----------------------------------


def _build_price_cache_with_signals(
    signals_by_ticker: dict[str, TechnicalSignal],
) -> PriceCache:
    """Pre-populate a PriceCache with bars derived from each signal.

    The bars are not actually consulted by the indicator math (we
    override ``compute_technical_signal`` to return the signal
    directly) — but the cache must NOT return ``None`` for
    ``fetch_bars`` or ``compute_technical_signal`` will try to
    re-pull from yfinance.
    """
    import tempfile

    cache = PriceCache(db_path=os.path.join(tempfile.mkdtemp(), "pc.sqlite"))
    today = date.today()
    for ticker, sig in signals_by_ticker.items():
        # 250 days of placeholder data; the real series doesn't
        # matter because we mock compute_technical_signal.
        dates = pd.bdate_range(end=today, periods=250)
        bars = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000,
            }
            for d in dates
        ]
        cache.store_bars(ticker=ticker, as_of_date=sig.as_of_date, bars=bars)
    return cache


class TestScoreUniverse:
    def test_filters_out_below_threshold_tech(self):
        sigs = {
            "AAPL": _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True),
            # Only 1 technical flag (RSI) → below MIN_TECHNICAL_BULLISH.
            "MSFT": _make_signal("MSFT", rsi_not_overbought=True),
        }
        cache = _build_price_cache_with_signals(sigs)
        news_scores = {"AAPL": 0.5, "MSFT": 0.5}

        with (
            patch(
                "portfoliomind.signals.combined.compute_technical_signal",
                side_effect=lambda t, **kw: sigs[t.upper()],
            ),
            patch(
                "portfoliomind.news.sentiment.score_universe_sentiment",
                return_value=news_scores,
            ),
        ):
            candidates = score_universe(
                tickers=("AAPL", "MSFT"),
                top_n=10,
                price_cache=cache,
                as_of_date="2026-06-10",
            )

        tickers = [c.ticker for c in candidates]
        assert "AAPL" in tickers
        assert "MSFT" not in tickers

    def test_filters_out_below_threshold_news(self):
        sigs = {
            "AAPL": _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True),
        }
        cache = _build_price_cache_with_signals(sigs)
        news_scores = {"AAPL": 0.0}  # exactly at threshold → fails

        with (
            patch(
                "portfoliomind.signals.combined.compute_technical_signal",
                side_effect=lambda t, **kw: sigs[t.upper()],
            ),
            patch(
                "portfoliomind.news.sentiment.score_universe_sentiment",
                return_value=news_scores,
            ),
        ):
            candidates = score_universe(
                tickers=("AAPL",),
                top_n=10,
                price_cache=cache,
                as_of_date="2026-06-10",
            )

        assert candidates == []

    def test_sorted_by_combined_score_descending(self):
        sigs = {
            "LOW": _make_signal("LOW", sma_golden_cross=True, rsi_not_overbought=True),  # 2/4
            "MID": _make_signal("MID", sma_golden_cross=True, twenty_day_breakout=True, macd_bullish=True, rsi_not_overbought=True),  # 4/4
            "HI": _make_signal("HI", sma_golden_cross=True, twenty_day_breakout=True, rsi_not_overbought=True),  # 3/4
        }
        cache = _build_price_cache_with_signals(sigs)
        news_scores = {
            "LOW": 0.3,  # 2/4 = 0.5; combined = 0.5*0.6 + 0.3*0.4 = 0.42
            "MID": 0.3,  # 4/4 = 1.0; combined = 1.0*0.6 + 0.3*0.4 = 0.72
            "HI": 0.3,   # 3/4 = 0.75; combined = 0.75*0.6 + 0.3*0.4 = 0.57
        }

        with (
            patch(
                "portfoliomind.signals.combined.compute_technical_signal",
                side_effect=lambda t, **kw: sigs[t.upper()],
            ),
            patch(
                "portfoliomind.news.sentiment.score_universe_sentiment",
                return_value=news_scores,
            ),
        ):
            candidates = score_universe(
                tickers=("LOW", "MID", "HI"),
                top_n=10,
                price_cache=cache,
                as_of_date="2026-06-10",
            )

        # All 3 pass; sorted MID > HI > LOW.
        assert [c.ticker for c in candidates] == ["MID", "HI", "LOW"]
        assert candidates[0].combined_score > candidates[1].combined_score
        assert candidates[1].combined_score > candidates[2].combined_score

    def test_top_n_truncates(self):
        # 5 candidates; ask for top 3.
        sigs = {
            f"T{i}": _make_signal(
                f"T{i}",
                sma_golden_cross=True,
                twenty_day_breakout=True,
                macd_bullish=True,
                rsi_not_overbought=True,
            )
            for i in range(5)
        }
        cache = _build_price_cache_with_signals(sigs)
        news_scores = {f"T{i}": 0.3 + i * 0.1 for i in range(5)}

        with (
            patch(
                "portfoliomind.signals.combined.compute_technical_signal",
                side_effect=lambda t, **kw: sigs[t.upper()],
            ),
            patch(
                "portfoliomind.news.sentiment.score_universe_sentiment",
                return_value=news_scores,
            ),
        ):
            candidates = score_universe(
                tickers=tuple(sigs.keys()),
                top_n=3,
                price_cache=cache,
                as_of_date="2026-06-10",
            )

        assert len(candidates) == 3
        # Highest news = T4 first.
        assert candidates[0].ticker == "T4"

    def test_empty_universe(self):
        with patch(
            "portfoliomind.news.sentiment.score_universe_sentiment",
            return_value={},
        ):
            candidates = score_universe(tickers=(), top_n=10)
        assert candidates == []

    def test_full_universe_default(self):
        # Don't actually pull anything — patch both compute_technical_signal
        # and the sentiment call to return empty/empty.
        with (
            patch(
                "portfoliomind.signals.combined.compute_technical_signal",
                side_effect=RuntimeError("not exercised"),
            ),
            patch(
                "portfoliomind.news.sentiment.score_universe_sentiment",
                return_value={},
            ),
        ):
            candidates = score_universe(tickers=None, top_n=10)
        # Every ticker hits the runtime error → empty result.
        assert candidates == []

    def test_yfinance_failure_for_one_ticker_does_not_break_others(self):
        sigs = {
            "AAPL": _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True),
        }

        def _maybe_fail(ticker, **kw):
            if ticker.upper() == "MSFT":
                raise RuntimeError("rate limited")
            return sigs[ticker.upper()]

        cache = _build_price_cache_with_signals(sigs)
        news_scores = {"AAPL": 0.5, "MSFT": 0.5}

        with (
            patch(
                "portfoliomind.signals.combined.compute_technical_signal",
                side_effect=_maybe_fail,
            ),
            patch(
                "portfoliomind.news.sentiment.score_universe_sentiment",
                return_value=news_scores,
            ),
        ):
            candidates = score_universe(
                tickers=("AAPL", "MSFT"),
                top_n=10,
                price_cache=cache,
                as_of_date="2026-06-10",
            )

        tickers = [c.ticker for c in candidates]
        assert "AAPL" in tickers
        assert "MSFT" not in tickers

    def test_empty_signal_treated_as_skip(self):
        # An "empty signal" (yfinance miss) has bullish_count=0
        # AND close=0.0. The score_universe loop must skip it.
        sigs = {
            "AAPL": _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True),
            "EMPTY": TechnicalSignal(
                ticker="EMPTY", as_of_date="2026-06-10",
                sma_golden_cross=False, twenty_day_breakout=False,
                macd_bullish=False, rsi_not_overbought=False,
                bullish_count=0, sma_50=0, sma_200=0, rsi_14=0,
                macd=0, macd_signal=0, close=0.0,
            ),
        }
        cache = _build_price_cache_with_signals(sigs)
        news_scores = {"AAPL": 0.5, "EMPTY": 0.5}

        with (
            patch(
                "portfoliomind.signals.combined.compute_technical_signal",
                side_effect=lambda t, **kw: sigs[t.upper()],
            ),
            patch(
                "portfoliomind.news.sentiment.score_universe_sentiment",
                return_value=news_scores,
            ),
        ):
            candidates = score_universe(
                tickers=("AAPL", "EMPTY"),
                top_n=10,
                price_cache=cache,
                as_of_date="2026-06-10",
            )

        assert [c.ticker for c in candidates] == ["AAPL"]

    def test_sentiment_failure_falls_back_to_zero(self):
        sigs = {
            "AAPL": _make_signal("AAPL", sma_golden_cross=True, rsi_not_overbought=True),
        }
        cache = _build_price_cache_with_signals(sigs)

        with (
            patch(
                "portfoliomind.signals.combined.compute_technical_signal",
                side_effect=lambda t, **kw: sigs[t.upper()],
            ),
            patch(
                "portfoliomind.news.sentiment.score_universe_sentiment",
                side_effect=RuntimeError("openai down"),
            ),
        ):
            candidates = score_universe(
                tickers=("AAPL",),
                top_n=10,
                price_cache=cache,
                as_of_date="2026-06-10",
            )

        # Sentiment defaulted to 0.0, news gate fails → no candidates.
        assert candidates == []


# --- Candidate dataclass ----------------------------------------------------


class TestCandidateDataclass:
    def test_is_frozen(self):
        cand = Candidate(
            ticker="AAPL",
            strategy="x",
            timeframe="swing",
            entry_price=100.0,
            technical_score=0.5,
            news_score=0.5,
            combined_score=0.5,
            top_signal_reason="rsi",
            technical_signal=_make_signal("AAPL"),
        )
        with pytest.raises(Exception):  # FrozenInstanceError is a subclass of AttributeError
            cand.ticker = "MSFT"  # type: ignore[misc]

    def test_to_dict_includes_technical_signal(self):
        cand = Candidate(
            ticker="AAPL",
            strategy="x",
            timeframe="swing",
            entry_price=100.0,
            technical_score=0.5,
            news_score=0.5,
            combined_score=0.5,
            top_signal_reason="rsi",
            technical_signal=_make_signal("AAPL"),
        )
        d = cand.to_dict()
        assert d["ticker"] == "AAPL"
        assert isinstance(d["technical_signal"], dict)
        assert d["technical_signal"]["ticker"] == "AAPL"
        assert "sma_golden_cross" in d["technical_signal"]


# --- Public constants ------------------------------------------------------


def test_weight_sums_to_one():
    # The two weights should sum to 1.0; a regression that changes
    # one without the other would shift the rank ordering.
    assert math.isclose(WEIGHT_TECHNICAL + WEIGHT_NEWS, 1.0)


def test_min_technical_bullish_is_two():
    # The "at least 2 of 4" gate is the spec's deliberate choice.
    assert MIN_TECHNICAL_BULLISH == 2


def test_min_news_sentiment_is_0_2():
    # The "positive" threshold is +0.2 per the spec.
    assert math.isclose(MIN_NEWS_SENTIMENT, 0.2)


def test_strategy_and_timeframe_defaults():
    assert STRATEGY == "swing-bullish-news"
    assert TIMEFRAME == "swing"

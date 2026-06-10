"""Unit tests for :mod:`portfoliomind.signals.combiner`.

Hermetic — mocks yfinance (via :func:`portfoliomind.signals.technicals.fetch_ohlcv`)
and the LLM (via :func:`portfoliomind.news.sentiment.score_ticker_sentiment`).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from portfoliomind.signals.cache import TechnicalCache
from portfoliomind.signals.combiner import (
    MIN_HISTORY_BARS,
    WEIGHT_SENTIMENT,
    WEIGHT_TECHNICAL,
    Signal,
    _confidence,
    score_ticker,
    score_universe,
)


# --- Helpers ---------------------------------------------------------------


def _synthetic_closes(trend: str = "up", n: int = 100) -> list[float]:
    """Build a synthetic close series for a given trend direction."""
    if trend == "up":
        return [100.0 + 0.5 * i for i in range(n)]
    if trend == "down":
        return [200.0 - 0.5 * i for i in range(n)]
    if trend == "flat":
        return [100.0 + 0.05 * i for i in range(n)]
    raise ValueError(f"unknown trend: {trend}")


# --- Combine math (pure) ---------------------------------------------------


class TestConfidence:
    def test_perfect_agreement_high_magnitude(self):
        # Both at +0.6 → agreement 1.0, combined 0.6, confidence 0.6.
        c = _confidence(0.6, 0.6, 0.6)
        assert c == pytest.approx(0.6)

    def test_disagreement_zero_confidence(self):
        # tech +0.6, sentiment -0.6 → abs(diff)=1.2 (capped at 1.0),
        # agreement 0, confidence 0.
        c = _confidence(0.0, 0.6, -0.6)
        assert c == 0.0

    def test_partial_agreement(self):
        # tech +0.6, sentiment +0.2 → diff 0.4, agreement 0.6,
        # combined 0.44, confidence 0.44 * 0.6 = 0.264.
        c = _confidence(0.44, 0.6, 0.2)
        assert c == pytest.approx(0.44 * 0.6)

    def test_low_magnitude_low_confidence(self):
        # Both near 0 → high agreement but low magnitude → low confidence.
        c = _confidence(0.05, 0.05, 0.05)
        assert c == pytest.approx(0.05)

    def test_bounds_zero_to_one(self):
        c = _confidence(2.0, 2.0, 2.0)  # over-saturated inputs
        assert 0.0 <= c <= 1.0


class TestWeights:
    def test_weights_sum_to_one(self):
        assert WEIGHT_TECHNICAL + WEIGHT_SENTIMENT == pytest.approx(1.0)

    def test_weights_in_published_ranges(self):
        # Spec is 0.6 / 0.4; if these change it's a deliberate re-tune.
        assert WEIGHT_TECHNICAL == pytest.approx(0.6)
        assert WEIGHT_SENTIMENT == pytest.approx(0.4)


# --- score_ticker (no network) --------------------------------------------


class TestScoreTickerNoNetwork:
    """The combiner calls fetch_ohlcv and score_ticker_sentiment.
    Both are mocked here so the tests are hermetic."""

    def _patched(
        self,
        *,
        closes: list[float] | None,
        sentiment: float = 0.0,
        sentiment_raises: Exception | None = None,
    ):
        """Return a context manager stack that mocks both network calls."""
        from contextlib import ExitStack

        stack = ExitStack()
        if closes is None:
            stack.enter_context(
                patch(
                    "portfoliomind.signals.combiner.fetch_ohlcv",
                    return_value=[],
                )
            )
        else:
            stack.enter_context(
                patch(
                    "portfoliomind.signals.combiner.fetch_ohlcv",
                    return_value=list(closes),
                )
            )
        if sentiment_raises is not None:
            stack.enter_context(
                patch(
                    "portfoliomind.signals.combiner.score_ticker_sentiment",
                    side_effect=sentiment_raises,
                )
            )
        else:
            stack.enter_context(
                patch(
                    "portfoliomind.signals.combiner.score_ticker_sentiment",
                    return_value=sentiment,
                )
            )
        return stack

    def test_combined_uses_60_40_weights(self):
        # Build a series whose technical score is known: 100-bar uptrend
        # yields trend ≈ 0.8, momentum = 1.0, vol = 0.
        # Combined technical score = 0.5*0.8 + 0.3*1.0 + 0.2*0 = 0.7.
        with self._patched(closes=_synthetic_closes("up"), sentiment=0.0):
            sig = score_ticker("AAPL", openai_api_key="dummy")
        assert sig.ticker == "AAPL"
        assert sig.sentiment == 0.0
        # Technical sub-score is deterministic for this synthetic series.
        assert sig.technical == pytest.approx(0.7, abs=0.05)
        # combined = 0.6 * technical + 0.4 * sentiment
        assert sig.combined == pytest.approx(WEIGHT_TECHNICAL * sig.technical)

    def test_combined_with_positive_sentiment(self):
        with self._patched(closes=_synthetic_closes("up"), sentiment=0.5):
            sig = score_ticker("AAPL", openai_api_key="dummy")
        assert sig.combined == pytest.approx(
            WEIGHT_TECHNICAL * sig.technical + WEIGHT_SENTIMENT * 0.5
        )

    def test_combined_with_negative_sentiment(self):
        with self._patched(closes=_synthetic_closes("down"), sentiment=-0.4):
            sig = score_ticker("AAPL", openai_api_key="dummy")
        assert sig.technical < 0  # downtrend → negative
        assert sig.sentiment == -0.4
        # combined should be clearly negative
        assert sig.combined < 0

    def test_confidence_high_when_components_agree(self):
        with self._patched(closes=_synthetic_closes("up"), sentiment=0.5):
            sig_agree = score_ticker("AAPL", openai_api_key="dummy")
        # Both bullish → high confidence.
        assert sig_agree.confidence > 0.3

    def test_confidence_low_when_components_disagree(self):
        with self._patched(closes=_synthetic_closes("up"), sentiment=-0.5):
            sig_disagree = score_ticker("AAPL", openai_api_key="dummy")
        # Tech positive, sentiment negative → low confidence.
        assert sig_disagree.confidence < 0.1

    def test_never_raises_on_yfinance_failure(self):
        with self._patched(closes=[], sentiment=0.0):
            sig = score_ticker("AAPL", openai_api_key="dummy")
        # Empty history → zero technical, but no crash.
        assert sig.technical == 0.0
        assert sig.combined == 0.0
        assert sig.error == ""

    def test_never_raises_on_sentiment_failure(self):
        with self._patched(
            closes=_synthetic_closes("up"),
            sentiment_raises=RuntimeError("openai down"),
        ):
            sig = score_ticker("AAPL", openai_api_key="dummy")
        # Sentiment failure → 0.0 (the spec's "no news" default).
        assert sig.sentiment == 0.0
        assert sig.error == ""
        # Technical component still produced.
        assert sig.technical != 0.0

    def test_never_raises_on_missing_api_key(self):
        """No OPENAI_API_KEY → sentiment defaults to 0.0, no crash."""
        with self._patched(closes=_synthetic_closes("up"), sentiment=0.0):
            sig = score_ticker("AAPL", openai_api_key=None)
        assert sig.sentiment == 0.0
        # Technical is real; combined is 0.6 * technical.
        assert sig.combined == pytest.approx(WEIGHT_TECHNICAL * sig.technical)
        # Reasons list should mention the missing key.
        assert any("OPENAI_API_KEY" in r for r in sig.reasons)

    def test_insufficient_history_returns_zero_score(self):
        # Only 20 bars → far below MIN_HISTORY_BARS.
        with self._patched(
            closes=[100.0 + 0.1 * i for i in range(20)],
            sentiment=0.0,
        ):
            sig = score_ticker("AAPL", openai_api_key="dummy")
        assert sig.technical == 0.0
        assert sig.combined == 0.0
        # Reasons list should mention insufficient history.
        assert any("history" in r.lower() for r in sig.reasons)

    def test_yfinance_returns_empty_returns_zero_score(self):
        with self._patched(closes=[], sentiment=0.0):
            sig = score_ticker("AAPL", openai_api_key="dummy")
        assert sig.combined == 0.0
        assert any("history" in r.lower() for r in sig.reasons)

    def test_reasons_contain_human_readable_explanation(self):
        with self._patched(closes=_synthetic_closes("up"), sentiment=0.4):
            sig = score_ticker("AAPL", openai_api_key="dummy")
        # Per spec, reasons include the technical sub-scores + the
        # combined + confidence lines.
        text = " | ".join(sig.reasons)
        assert "trend" in text.lower()
        assert "momentum" in text.lower()
        assert "news sentiment" in text.lower()
        assert "contribution" in text.lower()
        assert "combined" in text.lower()
        assert "confidence" in text.lower()

    def test_ticker_is_uppercased(self):
        with self._patched(closes=_synthetic_closes("up"), sentiment=0.0):
            sig = score_ticker("aapl", openai_api_key="dummy")
        assert sig.ticker == "AAPL"

    def test_asof_date_is_set(self):
        with self._patched(closes=_synthetic_closes("up"), sentiment=0.0):
            sig = score_ticker("AAPL", openai_api_key="dummy")
        assert sig.asof_date  # non-empty YYYY-MM-DD

    def test_error_field_populated_on_unexpected_failure(self):
        """If everything blows up, score_ticker must still return a Signal."""
        with (
            patch(
                "portfoliomind.signals.combiner.fetch_ohlcv",
                side_effect=RuntimeError("network down"),
            ),
            patch(
                "portfoliomind.signals.combiner.score_ticker_sentiment",
                side_effect=RuntimeError("openai down"),
            ),
        ):
            sig = score_ticker("AAPL", openai_api_key="dummy")
        # The outer try/except catches it.
        assert isinstance(sig, Signal)
        assert sig.combined == 0.0
        assert sig.confidence == 0.0
        assert "RuntimeError" in sig.error or "network" in sig.error


# --- Idempotency via cache -------------------------------------------------


class TestIdempotency:
    """The cache is the only thing that makes score_ticker idempotent in
    a single day. Re-runs without cache would re-fetch from yfinance +
    re-call the LLM."""

    def test_second_call_with_cache_hits_yfinance_once(
        self, tmp_path
    ):
        from unittest.mock import patch

        cache = TechnicalCache.open(tmp_path / "idem.sqlite")
        closes = _synthetic_closes("up")
        with patch(
            "portfoliomind.signals.combiner.fetch_ohlcv",
            return_value=list(closes),
        ) as mock_fetch, patch(
            "portfoliomind.signals.combiner.score_ticker_sentiment",
            return_value=0.0,
        ):
            sig1 = score_ticker("AAPL", cache=cache, openai_api_key="dummy")
            sig2 = score_ticker("AAPL", cache=cache, openai_api_key="dummy")
        # First call fetches + caches; second call hits the cache.
        assert mock_fetch.call_count == 1
        # Results identical.
        assert sig1.combined == sig2.combined
        assert sig1.technical == sig2.technical
        assert sig1.asof_date == sig2.asof_date


# --- score_universe --------------------------------------------------------


class TestScoreUniverse:
    def test_returns_one_signal_per_ticker(self):
        # Mock everything so the test is hermetic.
        with (
            patch(
                "portfoliomind.signals.combiner.fetch_ohlcv",
                return_value=_synthetic_closes("up"),
            ),
            patch(
                "portfoliomind.signals.combiner.score_ticker_sentiment",
                return_value=0.0,
            ),
        ):
            signals = score_universe(
                tickers=("AAPL", "MSFT", "TSLA"),
                openai_api_key="dummy",
            )
        assert len(signals) == 3
        assert {s.ticker for s in signals} == {"AAPL", "MSFT", "TSLA"}

    def test_sorted_by_combined_desc(self):
        # Three different synthetic trends → different technical scores.
        series = {
            "UP": _synthetic_closes("up"),
            "DOWN": _synthetic_closes("down"),
            "FLAT": _synthetic_closes("flat"),
        }

        def _fetch(ticker, **_):
            return series[ticker]

        with (
            patch(
                "portfoliomind.signals.combiner.fetch_ohlcv",
                side_effect=_fetch,
            ),
            patch(
                "portfoliomind.signals.combiner.score_ticker_sentiment",
                return_value=0.0,
            ),
        ):
            signals = score_universe(
                tickers=("UP", "DOWN", "FLAT"),
                openai_api_key="dummy",
            )
        scores = [s.combined for s in signals]
        assert scores == sorted(scores, reverse=True)
        # The first should be UP, the last should be DOWN.
        assert signals[0].ticker == "UP"
        assert signals[-1].ticker == "DOWN"

    def test_never_raises_on_full_failure(self):
        with (
            patch(
                "portfoliomind.signals.combiner.fetch_ohlcv",
                side_effect=RuntimeError("yfinance down"),
            ),
            patch(
                "portfoliomind.signals.combiner.score_ticker_sentiment",
                side_effect=RuntimeError("openai down"),
            ),
        ):
            signals = score_universe(
                tickers=("AAPL", "MSFT"),
                openai_api_key="dummy",
            )
        # Every ticker is represented, even on failure.
        assert len(signals) == 2
        assert all(s.combined == 0.0 for s in signals)
        # At least one should have an error field populated.
        assert any(s.error for s in signals)

    def test_empty_tickers_returns_empty(self):
        assert score_universe(tickers=(), openai_api_key="dummy") == []


# --- Signal dataclass sanity ----------------------------------------------


class TestSignalDataclass:
    def test_to_dict_round_trip(self):
        s = Signal(
            ticker="AAPL",
            combined=0.4,
            technical=0.5,
            sentiment=0.2,
            confidence=0.35,
            reasons=["r1", "r2"],
            error="",
            asof_date="2026-06-10",
        )
        d = s.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["combined"] == 0.4
        assert d["technical"] == 0.5
        assert d["sentiment"] == 0.2
        assert d["confidence"] == 0.35
        assert d["reasons"] == ["r1", "r2"]
        assert d["asof_date"] == "2026-06-10"

    def test_frozen(self):
        s = Signal(ticker="AAPL", combined=0.0, technical=0.0, sentiment=0.0, confidence=0.0)
        with pytest.raises(Exception):
            s.combined = 0.5  # type: ignore[misc]


# --- min-history constant --------------------------------------------------


class TestMinHistory:
    def test_min_history_bars_at_least_slow_sma(self):
        # The MIN_HISTORY_BARS threshold must be ≥ SMA_SLOW so the
        # 50-day slow SMA can actually be computed.
        from portfoliomind.signals.technicals import SMA_SLOW
        assert MIN_HISTORY_BARS >= SMA_SLOW

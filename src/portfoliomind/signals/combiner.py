"""Combine technical + sentiment into a single :class:`Signal` (card 6).

The combiner is the public entry point cards 7/8 will import. The
contract is strict: **never raise**. Every failure mode — bad ticker,
network down, malformed cache, LLM timeout — returns a
:class:`Signal` with ``combined=0.0`` and the error string in
``reasons``.

Combine math (per the card 6 spec):

* ``combined = 0.6 * technical.score + 0.4 * sentiment``
* ``confidence = abs(combined) * (1 - abs(technical.score - sentiment))``

  Confidence is the *magnitude* of the combined signal multiplied by
  the *agreement* of the two components. A high-confidence signal is
  one where technicals and sentiment both agree on direction AND the
  combined magnitude is large. A signal where the two components
  contradict (e.g. technicals +0.6, sentiment -0.6) has near-zero
  confidence and the operator should ignore it.

Idempotency: the same-day cache means a re-run in the same Bogota day
returns identical :class:`Signal` objects for every ticker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..logging_setup import get_logger
from ..news.sentiment import score_ticker_sentiment
from ..universe import UNIVERSE
from .cache import TechnicalCache
from .technicals import (
    TechnicalScore,
    compute_technical_score,
    fetch_ohlcv,
)

log = get_logger(__name__)


# --- Combine weights -------------------------------------------------------
#: How much technicals contribute to ``combined``.
WEIGHT_TECHNICAL: float = 0.6
#: How much sentiment contributes to ``combined``.
WEIGHT_SENTIMENT: float = 0.4

#: Minimum history (closes) before we even try to compute a technical
#: score. Below this, we return a zero score with an "insufficient
#: history" reason — no crash, no NaN.
MIN_HISTORY_BARS: int = 60


# --- Public dataclass ------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """The single-ticker output consumed by card 7 / card 8.

    Fields:

    * ``ticker`` — uppercase ticker symbol.
    * ``combined`` — the weighted aggregate in [-1, +1].
    * ``technical`` — the technical sub-score in [-1, +1].
    * ``sentiment`` — the sentiment sub-score in [-1, +1].
    * ``confidence`` — in [0, 1]. Higher = components agree AND the
      combined signal is strong. Card 7 should filter low-confidence
      signals before pinging the operator.
    * ``reasons`` — human-readable list, ready to paste into a Discord
      message or a Google Sheet cell.
    * ``error`` — empty on success; non-empty when something went
      wrong (network, missing data, etc.). The combiner never raises;
      a populated ``error`` is how a downstream caller knows the
      signal is a placeholder.
    * ``asof_date`` — YYYY-MM-DD Bogota. Same value as the technical
      cache key, so a re-run in the same day returns the same
      ``asof_date``.
    """

    ticker: str
    combined: float
    technical: float
    sentiment: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    error: str = ""
    asof_date: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "combined": self.combined,
            "technical": self.technical,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "error": self.error,
            "asof_date": self.asof_date,
        }


class SignalError(RuntimeError):
    """Raised only by internal helpers; the public API never raises."""


# --- Internal helpers ------------------------------------------------------


def _zero_signal(ticker: str, *, error: str = "", asof_date: str = "") -> Signal:
    """Build a placeholder Signal with combined=0 and confidence=0."""
    return Signal(
        ticker=ticker.upper(),
        combined=0.0,
        technical=0.0,
        sentiment=0.0,
        confidence=0.0,
        reasons=[f"error: {error}"] if error else [],
        error=error,
        asof_date=asof_date,
    )


def _get_or_compute_technical(
    ticker: str,
    *,
    cache: Optional[TechnicalCache],
    asof_date: str,
) -> TechnicalScore:
    """Read the cache, fall back to yfinance, write back on miss.

    Never raises: a yfinance failure returns a zero-score
    :class:`TechnicalScore` with an explanatory reason.
    """
    if cache is not None:
        cached = cache.get(ticker)
        if cached is not None:
            return cached

    closes = fetch_ohlcv(ticker)
    if not closes or len(closes) < MIN_HISTORY_BARS:
        score = TechnicalScore(
            ticker=ticker.upper(),
            trend=0.0,
            momentum=0.0,
            volatility=0.0,
            score=0.0,
            reasons=["insufficient price history (need ≥ 60 daily closes)"],
            asof_date=asof_date,
        )
    else:
        score = compute_technical_score(ticker, closes=closes, asof_date=asof_date)

    if cache is not None:
        # Stamp the asof_date so the cached row matches.
        score = TechnicalScore(
            ticker=score.ticker,
            trend=score.trend,
            momentum=score.momentum,
            volatility=score.volatility,
            score=score.score,
            reasons=list(score.reasons),
            asof_date=asof_date,
        )
        cache.put(score)

    return score


def _confidence(combined: float, technical: float, sentiment: float) -> float:
    """Confidence in [0, 1]: magnitude × agreement.

    Agreement shrinks linearly with the absolute disagreement between
    the two components. When technical = +0.6 and sentiment = -0.4,
    abs(diff) = 1.0, so agreement term = 0 and confidence = 0.
    """
    agreement = max(0.0, 1.0 - abs(technical - sentiment))
    return max(0.0, min(1.0, abs(combined) * agreement))


def _contribution_string(label: str, value: float, weight: float) -> str:
    """One-line reason string for a sub-score's contribution to the combined signal."""
    contrib = weight * value
    return f"{label} {value:+.3f} (weight {weight:.1f}) → {contrib:+.3f} contribution"


# --- Public API ------------------------------------------------------------


def score_ticker(
    ticker: str,
    *,
    cache: Optional[TechnicalCache] = None,
    openai_api_key: Optional[str] = None,
) -> Signal:
    """Compute a single :class:`Signal` for ``ticker``.

    The contract: **never raise**. Any failure mode is converted into a
    :class:`Signal` with ``combined=0.0``, ``confidence=0.0``, and the
    error string in ``reasons`` and ``error``. Cards 7/8 depend on this.

    ``openai_api_key`` is required for the sentiment sub-score; if
    missing, the technical sub-score is still produced and the
    sentiment defaults to 0.0 (so a missing API key never blocks a
    technical signal).
    """
    ticker = ticker.upper()
    try:
        # 1) Technical
        # Resolve asof_date up front so a cache miss + yfinance failure
        # still produce a Signal with the right day stamp.
        from ..time_utils import BOGOTA_TZ, now_bogota  # local import: avoid cycles
        asof_date = now_bogota().astimezone(BOGOTA_TZ).strftime("%Y-%m-%d")
        tech = _get_or_compute_technical(ticker, cache=cache, asof_date=asof_date)

        # 2) Sentiment — graceful when the key is missing.
        if openai_api_key:
            try:
                sentiment = float(score_ticker_sentiment(ticker, api_key=openai_api_key))
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "signals: sentiment failed for %s: %s", ticker, type(e).__name__,
                )
                sentiment = 0.0
        else:
            sentiment = 0.0

        # 3) Combine
        combined = WEIGHT_TECHNICAL * tech.score + WEIGHT_SENTIMENT * sentiment
        combined = max(-1.0, min(1.0, combined))
        confidence = _confidence(combined, tech.score, sentiment)

        # 4) Reasons — operator-facing prose.
        reasons: list[str] = []
        # Push the technical reasons first (already human-readable).
        reasons.extend(tech.reasons)
        # Then a one-liner for sentiment + each component's contribution.
        if openai_api_key:
            reasons.append(f"news sentiment: {sentiment:+.3f}")
        else:
            reasons.append("news sentiment: 0.000 (OPENAI_API_KEY not set)")
        reasons.append(_contribution_string("technical", tech.score, WEIGHT_TECHNICAL))
        reasons.append(_contribution_string("sentiment", sentiment, WEIGHT_SENTIMENT))
        reasons.append(
            f"combined: {combined:+.3f} | confidence: {confidence:+.3f} "
            f"(agreement={1.0 - abs(tech.score - sentiment):.2f})"
        )

        return Signal(
            ticker=ticker,
            combined=combined,
            technical=tech.score,
            sentiment=sentiment,
            confidence=confidence,
            reasons=reasons,
            error="",
            asof_date=asof_date,
        )
    except Exception as e:  # noqa: BLE001 — last-ditch: never raise
        log.warning(
            "signals: score_ticker(%s) failed: %s", ticker, type(e).__name__,
        )
        return _zero_signal(ticker, error=f"{type(e).__name__}: {e}")


def score_universe(
    tickers=UNIVERSE,
    *,
    cache: Optional[TechnicalCache] = None,
    openai_api_key: Optional[str] = None,
) -> list[Signal]:
    """Score every ticker in ``tickers`` (default: the full UNIVERSE).

    Returns a list sorted by ``combined`` descending. **Never raises**:
    a per-ticker failure is captured as a :class:`Signal` with a
    populated ``error`` field and ``combined=0.0``. The list always
    has one entry per input ticker.
    """
    out: list[Signal] = []
    for t in tickers:
        sig = score_ticker(t, cache=cache, openai_api_key=openai_api_key)
        out.append(sig)
    out.sort(key=lambda s: s.combined, reverse=True)
    log.info(
        "signals: scored %d tickers (errors=%d, high_conf=%d)",
        len(out),
        sum(1 for s in out if s.error),
        sum(1 for s in out if s.confidence >= 0.5),
    )
    return out


__all__ = [
    "WEIGHT_TECHNICAL",
    "WEIGHT_SENTIMENT",
    "MIN_HISTORY_BARS",
    "Signal",
    "SignalError",
    "score_ticker",
    "score_universe",
]

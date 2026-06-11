"""Bullish candidate selection — the card-6/7 seam the strategy runner imports.

:func:`score_universe` is the function
:mod:`portfoliomind.strategy_runner` lazy-imports as
``portfoliomind.signals.combined.score_universe``. It scans the
universe and returns the top-N **long-only bullish** candidates that
pass every gate:

* **Bullish-tech gate** — the technical score (trend + momentum +
  vol regime) must be positive.
* **Pattern gate** — the posterior probability of upside from the
  bullish-pattern catalogue (:mod:`portfoliomind.signals.patterns`)
  must be at least :data:`MIN_P_BULLISH`.
* **Positive-news gate** — the LLM news-sentiment score must not be
  negative (:data:`SENTIMENT_FLOOR`). Bad news disqualifies a setup
  no matter how pretty the chart is.
* **Strength gate** — the blended score must clear
  :data:`MIN_COMBINED`.

Blend (weights are module constants so the operator can re-tune
without forking)::

    combined = 0.40 * technical + 0.35 * pattern_score + 0.25 * sentiment

where ``pattern_score = 2 * p_bullish - 1`` puts the posterior on the
same [-1, +1] scale as the other two components.

The module follows the card-6 contract: **never raises**. A
per-ticker failure (network, short history, LLM error) drops that
ticker from the candidate list with a DEBUG/WARNING log — it never
aborts the scan. Tests inject ``fetch`` and ``sentiment_fn`` so the
module is hermetic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from ..logging_setup import get_logger
from ..universe import UNIVERSE
from .patterns import BullishPatterns, detect_bullish_patterns
from .technicals import (
    VOL_SHORT,
    TechnicalScore,
    compute_technical_score,
    fetch_ohlcv,
    realized_vol,
)

log = get_logger(__name__)


# --- Blend weights + gates ----------------------------------------------------
WEIGHT_TECHNICAL: float = 0.40
WEIGHT_PATTERNS: float = 0.35
WEIGHT_SENTIMENT: float = 0.25

#: Pattern-gate: minimum posterior P(upside) to qualify.
MIN_P_BULLISH: float = 0.55
#: Strength gate: minimum blended score to qualify.
MIN_COMBINED: float = 0.15
#: Positive-news gate: sentiment below this disqualifies. 0.0 means
#: "neutral or better" — no-news tickers (sentiment 0.0) still pass.
SENTIMENT_FLOOR: float = 0.0
#: Minimum daily closes before a ticker is scored at all.
MIN_HISTORY_BARS: int = 60


# --- Candidate dataclass --------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One qualified bullish candidate, ready for the position sizer.

    Carries everything downstream needs: the last close (sizing
    anchor), the 20-day realized vol (stop-distance anchor), the
    posterior ``p_bullish`` (Kelly edge), and operator-facing reasons.
    """

    ticker: str
    last_close: float
    technical: float
    pattern_score: float
    p_bullish: float
    sentiment: float
    combined: float
    confidence: float
    vol_20d: float
    patterns: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    asof_date: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "last_close": self.last_close,
            "technical": self.technical,
            "pattern_score": self.pattern_score,
            "p_bullish": self.p_bullish,
            "sentiment": self.sentiment,
            "combined": self.combined,
            "confidence": self.confidence,
            "vol_20d": self.vol_20d,
            "patterns": list(self.patterns),
            "reasons": list(self.reasons),
            "asof_date": self.asof_date,
        }


# --- Internal helpers --------------------------------------------------------------


def _default_sentiment_fn() -> Callable[[str], float]:
    """Build the production sentiment callable.

    Lazily imports the news layer so importing this module stays cheap
    (the strategy runner imports it on every morning tick). Without an
    ``OPENAI_API_KEY`` the sentiment is a constant 0.0 — the technical
    + pattern layers still work, matching the card-6 behavior.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        log.info("combined: OPENAI_API_KEY not set — news sentiment defaults to 0.0")
        return lambda ticker: 0.0

    def _score(ticker: str) -> float:
        from ..news.sentiment import score_ticker_sentiment  # lazy: keeps import light

        return float(score_ticker_sentiment(ticker, api_key=api_key))

    return _score


def _confidence(combined: float, components: tuple[float, float, float]) -> float:
    """Magnitude × agreement, like the card-6 combiner.

    ``spread`` is the max disagreement between any two components on
    the [-1, +1] scale (range 0-2), normalized to a [0, 1] agreement
    factor.
    """
    spread = max(components) - min(components)
    agreement = max(0.0, 1.0 - spread / 2.0)
    return max(0.0, min(1.0, abs(combined) * agreement))


def _score_one(
    ticker: str,
    *,
    fetch: Callable[[str], list[float]],
    sentiment_fn: Callable[[str], float],
    asof_date: str,
) -> Optional[Candidate]:
    """Score one ticker. Returns None when it fails any gate (or errors)."""
    closes = fetch(ticker)
    if not closes or len(closes) < MIN_HISTORY_BARS:
        log.debug("combined: %s skipped — insufficient history", ticker)
        return None

    tech: TechnicalScore = compute_technical_score(ticker, closes=closes, asof_date=asof_date)
    pats: BullishPatterns = detect_bullish_patterns(ticker, closes=closes, asof_date=asof_date)

    try:
        sentiment = max(-1.0, min(1.0, float(sentiment_fn(ticker))))
    except Exception as e:  # noqa: BLE001 — sentiment failure never blocks the tech signal
        log.warning("combined: sentiment failed for %s: %s", ticker, type(e).__name__)
        sentiment = 0.0

    combined = (
        WEIGHT_TECHNICAL * tech.score
        + WEIGHT_PATTERNS * pats.score
        + WEIGHT_SENTIMENT * sentiment
    )
    combined = max(-1.0, min(1.0, combined))

    # --- Gates (long-only bullish) ---------------------------------------
    if tech.score <= 0:
        log.debug("combined: %s failed bullish-tech gate (%.3f)", ticker, tech.score)
        return None
    if pats.p_bullish < MIN_P_BULLISH:
        log.debug("combined: %s failed pattern gate (p=%.3f)", ticker, pats.p_bullish)
        return None
    if sentiment < SENTIMENT_FLOOR:
        log.debug("combined: %s failed positive-news gate (%.3f)", ticker, sentiment)
        return None
    if combined < MIN_COMBINED:
        log.debug("combined: %s failed strength gate (%.3f)", ticker, combined)
        return None

    confidence = _confidence(combined, (tech.score, pats.score, sentiment))
    vol = realized_vol(closes, VOL_SHORT) or 0.0

    reasons: list[str] = []
    reasons.extend(tech.reasons)
    reasons.extend(pats.reasons)
    reasons.append(f"news sentiment: {sentiment:+.3f}")
    reasons.append(
        f"combined: {combined:+.3f} "
        f"(tech {tech.score:+.3f}×{WEIGHT_TECHNICAL:.2f} + "
        f"patterns {pats.score:+.3f}×{WEIGHT_PATTERNS:.2f} + "
        f"sentiment {sentiment:+.3f}×{WEIGHT_SENTIMENT:.2f}) | "
        f"confidence {confidence:.3f}"
    )

    return Candidate(
        ticker=ticker.upper(),
        last_close=float(closes[-1]),
        technical=tech.score,
        pattern_score=pats.score,
        p_bullish=pats.p_bullish,
        sentiment=sentiment,
        combined=combined,
        confidence=confidence,
        vol_20d=vol,
        patterns=[h.name for h in pats.hits],
        reasons=reasons,
        asof_date=asof_date,
    )


# --- Public API ----------------------------------------------------------------------


def score_universe(
    *,
    top_n: int = 5,
    tickers: Iterable[str] = UNIVERSE,
    fetch: Optional[Callable[[str], list[float]]] = None,
    sentiment_fn: Optional[Callable[[str], float]] = None,
) -> list[Candidate]:
    """Return the top-``top_n`` bullish candidates across ``tickers``.

    This is the entry point the strategy runner calls every morning.
    **Never raises** — a per-ticker failure drops the ticker, a
    catastrophic failure returns an empty list. Sorted by ``combined``
    descending.
    """
    tickers = tuple(tickers)
    if fetch is None:
        fetch = fetch_ohlcv
    if sentiment_fn is None:
        sentiment_fn = _default_sentiment_fn()

    try:
        from ..time_utils import BOGOTA_TZ, now_bogota  # local import: avoid cycles

        asof_date = now_bogota().astimezone(BOGOTA_TZ).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        asof_date = ""

    out: list[Candidate] = []
    for t in tickers:
        try:
            cand = _score_one(t, fetch=fetch, sentiment_fn=sentiment_fn, asof_date=asof_date)
        except Exception as e:  # noqa: BLE001 — one bad ticker never kills the scan
            log.warning("combined: scoring %s failed: %s", t, type(e).__name__)
            continue
        if cand is not None:
            out.append(cand)

    out.sort(key=lambda c: c.combined, reverse=True)
    selected = out[: max(0, int(top_n))]
    log.info(
        "combined: %d/%d tickers qualified; returning top %d",
        len(out),
        len(tickers),
        len(selected),
    )
    return selected


__all__ = [
    "WEIGHT_TECHNICAL",
    "WEIGHT_PATTERNS",
    "WEIGHT_SENTIMENT",
    "MIN_P_BULLISH",
    "MIN_COMBINED",
    "SENTIMENT_FLOOR",
    "MIN_HISTORY_BARS",
    "Candidate",
    "score_universe",
]

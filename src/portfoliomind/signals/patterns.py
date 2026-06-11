"""Bullish pattern detection + probabilistic aggregation (card 9).

This module is the bullish-pattern layer of the strategy: it scans a
ticker's daily close history for a fixed catalogue of bullish setups
(golden cross, 55-day breakout, oversold recovery, MACD bull cross,
higher lows, pullback bounce, uptrend stack) and folds the detected
patterns into a single **posterior probability of upside**
``p_bullish`` in ``[0, 1]`` via naive-Bayes log-odds aggregation.

Probabilistic model
-------------------

Each pattern carries a ``hit_rate`` — the assumed historical
probability that the pattern resolves upward over the strategy's
holding window. These are literature-informed priors (most classic
bullish continuation/reversal setups test out in the 55-65% band on
liquid US large caps), exposed as module constants so the operator
can re-tune them from backtests without forking the module.

Aggregation starts from the long-run drift prior
(:data:`PRIOR_P_UP` — US large caps close higher on ~53% of days)
and adds each detected pattern's evidence in log-odds space::

    log_odds = logit(PRIOR_P_UP)
             + Σ  EVIDENCE_SHRINK * (logit(hit_rate) - logit(0.5))

    p_bullish = sigmoid(log_odds)   # clamped to [P_FLOOR, P_CEIL]

The :data:`EVIDENCE_SHRINK` factor (< 1) discounts each pattern's
evidence because the patterns are *not* independent (a golden cross
and an uptrend stack co-occur often); naive-Bayes without shrinkage
would be overconfident. The posterior is clamped away from 0/1 —
no pattern stack is ever a certainty.

Design constraints (mirroring the card-6 conventions):

* Pure Python over plain lists — no pandas, no network. The caller
  supplies the closes (typically from
  :func:`portfoliomind.signals.technicals.fetch_ohlcv`).
* **Never raises** from :func:`detect_bullish_patterns` — short or
  malformed history yields zero hits and ``p_bullish == PRIOR_P_UP``
  with an explanatory reason.
* No raw OHLCV at INFO log level; DEBUG only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..logging_setup import get_logger
from .technicals import SMA_FAST, SMA_SLOW, rsi, sma

log = get_logger(__name__)


# --- Probabilistic priors ----------------------------------------------------
#: Long-run base rate of an up-resolution with no pattern evidence.
#: US large caps drift up on roughly 53% of days/weeks.
PRIOR_P_UP: float = 0.53

#: Discount applied to each pattern's log-odds evidence. Patterns are
#: correlated (trend patterns co-occur), so full naive-Bayes stacking
#: would overstate conviction. 0.7 keeps a 3-pattern stack in the
#: ~0.70-0.75 posterior range rather than >0.85.
EVIDENCE_SHRINK: float = 0.7

#: Posterior clamps — pattern evidence alone is never a certainty.
P_FLOOR: float = 0.05
P_CEIL: float = 0.95

#: Minimum history before any pattern detection is attempted. Matches
#: the combiner's MIN_HISTORY_BARS so the two layers gate identically.
MIN_PATTERN_BARS: int = 60

# --- Per-pattern hit-rate priors (tunable from backtests) --------------------
HIT_RATE_GOLDEN_CROSS: float = 0.62
HIT_RATE_UPTREND_STACK: float = 0.58
HIT_RATE_BREAKOUT: float = 0.63
HIT_RATE_RSI_RECOVERY: float = 0.60
HIT_RATE_MACD_CROSS: float = 0.61
HIT_RATE_HIGHER_LOWS: float = 0.59
HIT_RATE_PULLBACK_BOUNCE: float = 0.60

# --- Detection windows --------------------------------------------------------
#: A golden cross counts when SMA20 crossed above SMA50 within this
#: many bars — recent enough that the new trend is still young.
GOLDEN_CROSS_LOOKBACK: int = 10

#: Breakout window: close at a new high over the prior N bars
#: (Donchian-style 55-day channel, the classic turtle entry).
BREAKOUT_WINDOW: int = 55

#: RSI recovery: RSI dipped below this within RSI_RECOVERY_LOOKBACK
#: bars and has since recovered above RSI_RECOVERY_CONFIRM.
RSI_RECOVERY_OVERSOLD: float = 35.0
RSI_RECOVERY_CONFIRM: float = 45.0
RSI_RECOVERY_LOOKBACK: int = 10

#: MACD bull cross must have happened within this many bars.
MACD_CROSS_LOOKBACK: int = 5

#: Higher-lows: the last HIGHER_LOWS_WINDOW bars are split into this
#: many equal segments whose minima must be strictly increasing.
HIGHER_LOWS_WINDOW: int = 60
HIGHER_LOWS_SEGMENTS: int = 3

#: Pullback bounce: in an uptrend, price tagged SMA20 (within this
#: tolerance) inside the last PULLBACK_LOOKBACK bars and closed back
#: above it.
PULLBACK_TOLERANCE: float = 0.005
PULLBACK_LOOKBACK: int = 5


# --- Dataclasses --------------------------------------------------------------


@dataclass(frozen=True)
class PatternHit:
    """One detected bullish pattern with its evidence weight."""

    name: str
    hit_rate: float
    description: str


@dataclass(frozen=True)
class BullishPatterns:
    """The pattern-layer output for one ticker.

    * ``hits`` — the detected patterns, in catalogue order.
    * ``p_bullish`` — posterior P(upside) in [0, 1] from the log-odds
      aggregation. Equals :data:`PRIOR_P_UP` when nothing fired.
    * ``score`` — the same posterior mapped to [-1, +1]
      (``2 * p_bullish - 1``) so the combiner can blend it with the
      technical and sentiment scores on a common scale.
    * ``reasons`` — operator-facing prose, one line per hit plus the
      posterior summary.
    """

    ticker: str
    hits: list[PatternHit] = field(default_factory=list)
    p_bullish: float = PRIOR_P_UP
    score: float = 2.0 * PRIOR_P_UP - 1.0
    reasons: list[str] = field(default_factory=list)
    asof_date: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "patterns": [h.name for h in self.hits],
            "p_bullish": self.p_bullish,
            "score": self.score,
            "reasons": list(self.reasons),
            "asof_date": self.asof_date,
        }


# --- Pure math helpers --------------------------------------------------------


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def ema_series(values: list[float], span: int) -> list[float]:
    """Standard EMA over the full series (seeded with the first value)."""
    if span <= 0:
        raise ValueError("ema span must be positive")
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


def macd_histogram(closes: list[float], *, fast: int = 12, slow: int = 26, signal: int = 9) -> list[float]:
    """MACD-line minus signal-line series (positive = bullish side)."""
    if len(closes) < slow + signal:
        return []
    ema_fast = ema_series(closes, fast)
    ema_slow = ema_series(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema_series(macd_line, signal)
    return [m - s for m, s in zip(macd_line, signal_line)]


# --- Individual detectors ------------------------------------------------------
# Each detector takes the chronological closes and returns a PatternHit
# when the setup is present, else None. All are pure and never raise on
# well-formed float lists of length >= MIN_PATTERN_BARS.


def detect_golden_cross(closes: list[float]) -> Optional[PatternHit]:
    """SMA20 crossed above SMA50 within the last GOLDEN_CROSS_LOOKBACK bars."""
    if len(closes) < SMA_SLOW + GOLDEN_CROSS_LOOKBACK:
        return None
    now_fast = sma(closes, SMA_FAST)
    now_slow = sma(closes, SMA_SLOW)
    if now_fast is None or now_slow is None or not now_fast > now_slow:
        return None
    for k in range(1, GOLDEN_CROSS_LOOKBACK + 1):
        past = closes[:-k]
        past_fast = sma(past, SMA_FAST)
        past_slow = sma(past, SMA_SLOW)
        if past_fast is None or past_slow is None:
            return None
        if past_fast <= past_slow:
            return PatternHit(
                name="golden_cross",
                hit_rate=HIT_RATE_GOLDEN_CROSS,
                description=(
                    f"SMA{SMA_FAST} crossed above SMA{SMA_SLOW} {k} bar(s) ago"
                ),
            )
    return None


def detect_uptrend_stack(closes: list[float]) -> Optional[PatternHit]:
    """Close > SMA20 > SMA50 — the basic bullish alignment."""
    fast = sma(closes, SMA_FAST)
    slow = sma(closes, SMA_SLOW)
    if fast is None or slow is None:
        return None
    last = closes[-1]
    if last > fast > slow:
        return PatternHit(
            name="uptrend_stack",
            hit_rate=HIT_RATE_UPTREND_STACK,
            description=f"close {last:.2f} > SMA{SMA_FAST} {fast:.2f} > SMA{SMA_SLOW} {slow:.2f}",
        )
    return None


def detect_breakout(closes: list[float]) -> Optional[PatternHit]:
    """Close at a new high over the prior BREAKOUT_WINDOW bars (Donchian breakout)."""
    if len(closes) < BREAKOUT_WINDOW + 1:
        return None
    prior_high = max(closes[-(BREAKOUT_WINDOW + 1):-1])
    if closes[-1] > prior_high:
        return PatternHit(
            name="breakout",
            hit_rate=HIT_RATE_BREAKOUT,
            description=(
                f"close {closes[-1]:.2f} broke the {BREAKOUT_WINDOW}-day high {prior_high:.2f}"
            ),
        )
    return None


def detect_rsi_recovery(closes: list[float]) -> Optional[PatternHit]:
    """RSI dipped oversold recently and has recovered — bullish mean-reversion."""
    current = rsi(closes)
    if current is None or current < RSI_RECOVERY_CONFIRM:
        return None
    lowest = None
    for k in range(1, RSI_RECOVERY_LOOKBACK + 1):
        past = rsi(closes[:-k])
        if past is None:
            break
        lowest = past if lowest is None else min(lowest, past)
    if lowest is not None and lowest < RSI_RECOVERY_OVERSOLD:
        return PatternHit(
            name="rsi_recovery",
            hit_rate=HIT_RATE_RSI_RECOVERY,
            description=(
                f"RSI recovered from {lowest:.1f} to {current:.1f} "
                f"within {RSI_RECOVERY_LOOKBACK} bars"
            ),
        )
    return None


def detect_macd_cross(closes: list[float]) -> Optional[PatternHit]:
    """MACD line crossed above its signal line within MACD_CROSS_LOOKBACK bars."""
    hist = macd_histogram(closes)
    if len(hist) < MACD_CROSS_LOOKBACK + 1 or hist[-1] <= 0:
        return None
    for k in range(1, MACD_CROSS_LOOKBACK + 1):
        if hist[-1 - k] <= 0:
            return PatternHit(
                name="macd_cross",
                hit_rate=HIT_RATE_MACD_CROSS,
                description=f"MACD crossed above signal {k} bar(s) ago",
            )
    return None


def detect_higher_lows(closes: list[float]) -> Optional[PatternHit]:
    """Segment minima over the last HIGHER_LOWS_WINDOW bars strictly increase."""
    if len(closes) < HIGHER_LOWS_WINDOW:
        return None
    window = closes[-HIGHER_LOWS_WINDOW:]
    seg = HIGHER_LOWS_WINDOW // HIGHER_LOWS_SEGMENTS
    minima = [
        min(window[i * seg : (i + 1) * seg]) for i in range(HIGHER_LOWS_SEGMENTS)
    ]
    if all(minima[i] < minima[i + 1] for i in range(len(minima) - 1)):
        lows = " < ".join(f"{m:.2f}" for m in minima)
        return PatternHit(
            name="higher_lows",
            hit_rate=HIT_RATE_HIGHER_LOWS,
            description=f"higher lows over {HIGHER_LOWS_WINDOW} bars: {lows}",
        )
    return None


def detect_pullback_bounce(closes: list[float]) -> Optional[PatternHit]:
    """In an uptrend, price tagged SMA20 recently and closed back above it."""
    fast = sma(closes, SMA_FAST)
    slow = sma(closes, SMA_SLOW)
    if fast is None or slow is None or not (fast > slow):
        return None
    if closes[-1] <= fast:
        return None
    touched = any(
        closes[-1 - k] <= fast * (1.0 + PULLBACK_TOLERANCE)
        for k in range(1, PULLBACK_LOOKBACK + 1)
        if len(closes) > k
    )
    if touched:
        return PatternHit(
            name="pullback_bounce",
            hit_rate=HIT_RATE_PULLBACK_BOUNCE,
            description=(
                f"pullback to SMA{SMA_FAST} within {PULLBACK_LOOKBACK} bars, "
                f"closed back above at {closes[-1]:.2f}"
            ),
        )
    return None


#: The pattern catalogue, in evaluation order. Adding a detector here
#: is the only step needed to extend the strategy.
DETECTORS: tuple[Callable[[list[float]], Optional[PatternHit]], ...] = (
    detect_golden_cross,
    detect_uptrend_stack,
    detect_breakout,
    detect_rsi_recovery,
    detect_macd_cross,
    detect_higher_lows,
    detect_pullback_bounce,
)


# --- Aggregation ----------------------------------------------------------------


def aggregate_probability(
    hits: list[PatternHit],
    *,
    prior: float = PRIOR_P_UP,
    shrink: float = EVIDENCE_SHRINK,
) -> float:
    """Fold pattern hits into a posterior P(upside) via shrunk log-odds.

    Each hit contributes its evidence relative to an uninformative 0.5
    base, discounted by ``shrink`` to compensate for inter-pattern
    correlation. The result is clamped to [P_FLOOR, P_CEIL].
    """
    log_odds = _logit(prior)
    for h in hits:
        log_odds += shrink * (_logit(h.hit_rate) - _logit(0.5))
    return min(max(_sigmoid(log_odds), P_FLOOR), P_CEIL)


# --- Public entry point ----------------------------------------------------------


def detect_bullish_patterns(
    ticker: str,
    *,
    closes: list[float],
    asof_date: str = "",
) -> BullishPatterns:
    """Scan ``closes`` for the bullish catalogue and aggregate the posterior.

    Never raises: insufficient or malformed history yields zero hits
    and ``p_bullish == PRIOR_P_UP`` with an explanatory reason.
    """
    ticker = ticker.upper()
    if not closes or len(closes) < MIN_PATTERN_BARS:
        return BullishPatterns(
            ticker=ticker,
            reasons=[
                f"patterns: insufficient history (need ≥ {MIN_PATTERN_BARS} closes, "
                f"got {len(closes) if closes else 0})"
            ],
            asof_date=asof_date,
        )

    hits: list[PatternHit] = []
    reasons: list[str] = []
    for detector in DETECTORS:
        try:
            hit = detector(closes)
        except Exception as e:  # noqa: BLE001 — a single detector bug never kills the scan
            log.warning(
                "patterns: %s failed for %s: %s", detector.__name__, ticker, type(e).__name__
            )
            continue
        if hit is not None:
            hits.append(hit)
            reasons.append(
                f"pattern {hit.name} (hit-rate {hit.hit_rate:.0%}): {hit.description}"
            )

    p = aggregate_probability(hits)
    score = 2.0 * p - 1.0
    if not hits:
        reasons.append(f"patterns: none detected → p_bullish stays at prior {PRIOR_P_UP:.2f}")
    reasons.append(f"patterns: {len(hits)} hit(s) → p_bullish={p:.3f} (score {score:+.3f})")
    log.debug("patterns: %s hits=%d p_bullish=%.3f", ticker, len(hits), p)

    return BullishPatterns(
        ticker=ticker,
        hits=hits,
        p_bullish=p,
        score=score,
        reasons=reasons,
        asof_date=asof_date,
    )


__all__ = [
    "PRIOR_P_UP",
    "EVIDENCE_SHRINK",
    "P_FLOOR",
    "P_CEIL",
    "MIN_PATTERN_BARS",
    "PatternHit",
    "BullishPatterns",
    "DETECTORS",
    "ema_series",
    "macd_histogram",
    "aggregate_probability",
    "detect_golden_cross",
    "detect_uptrend_stack",
    "detect_breakout",
    "detect_rsi_recovery",
    "detect_macd_cross",
    "detect_higher_lows",
    "detect_pullback_bounce",
    "detect_bullish_patterns",
]

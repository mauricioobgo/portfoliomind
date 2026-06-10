"""Technical indicators → single [-1, +1] technical score (card 6).

Three sub-indicators, weighted and combined into one number in [-1, +1]:

* **Trend (50% weight)** — SMA(20)/SMA(50) ratio, smoothed into
  [-1, +1] via ``tanh``. SMA20 above SMA50 → positive; below → negative.
* **Momentum (30% weight)** — 14-day RSI(14). Map the
  ``[30, 70]`` band linearly to ``[-1, +1]``; below 30 → -1 (deeply
  oversold), above 70 → +1 (deeply overbought).
* **Volatility regime (20% weight)** — 20-day realized vol vs
  60-day baseline. Expanding vol + falling price → negative; expanding
  vol + rising price → mildly positive; contracting → 0.

The "weights live in one function" rule from the card 6 spec means the
three constants are colocated at the top of :func:`compute_technical_score`
— easy to tune without hunting through the file.

Design notes:

* All math is pure Python over plain lists. The only network call is
  :func:`fetch_ohlcv`, which wraps yfinance and is hermetic in tests
  (always monkeypatched).
* The fetch is short (6 months daily = ~130 bars) so SMA-50 is well
  defined. If the response is shorter (a recently-listed ticker),
  :func:`compute_technical_score` returns a zero score with an
  ``"insufficient history"`` reason — never raises. Card 7/8 rely on
  this.
* No raw OHLCV at INFO log level; DEBUG only. Same constraint as
  card 5 (no raw headlines at INFO).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..logging_setup import get_logger

log = get_logger(__name__)


# --- Weights ---------------------------------------------------------------
#: How much each sub-indicator contributes to the final technical score.
#: Sum to 1.0; tune in one place when card 7 feedback comes in.
WEIGHT_TREND: float = 0.5
WEIGHT_MOMENTUM: float = 0.3
WEIGHT_VOLATILITY: float = 0.2

#: Indicator windows. 20 / 50 / 14 are the textbook defaults; the
#: 60-day vol baseline is long enough to be "calm" but short enough
#: to react to regime changes.
SMA_FAST: int = 20
SMA_SLOW: int = 50
RSI_PERIOD: int = 14
VOL_SHORT: int = 20
VOL_LONG: int = 60

#: Threshold above which 20-day vol is "expanding" relative to 60-day
#: baseline. 1.10 = 10% expansion. Tuned to fire on real regime shifts
#: without noise from minor daily variance.
VOL_EXPANSION_RATIO: float = 1.10

#: RSI mapping band. RSI < RSI_OVERSOLD maps to -1, RSI > RSI_OVERBOUGHT
#: maps to +1, linear in between.
RSI_OVERSOLD: float = 30.0
RSI_OVERBOUGHT: float = 70.0

#: yfinance fetch parameters — keep in one place so tests can match.
YF_PERIOD: str = "6mo"
YF_INTERVAL: str = "1d"


# --- Data classes ----------------------------------------------------------


@dataclass(frozen=True)
class TechnicalScore:
    """One ticker's full technical score, with provenance + reasons.

    The ``score`` field is the weighted aggregate in [-1, +1]. The
    per-component fields (trend, momentum, volatility) are exposed
    individually so the combiner (and operator-facing demos) can
    explain *why* the aggregate is where it is.
    """

    ticker: str
    trend: float
    momentum: float
    volatility: float
    score: float
    reasons: list[str] = field(default_factory=list)
    asof_date: str = ""  # YYYY-MM-DD Bogota, set by the cache or fetcher

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "trend": self.trend,
            "momentum": self.momentum,
            "volatility": self.volatility,
            "score": self.score,
            "reasons": list(self.reasons),
            "asof_date": self.asof_date,
        }


class TechnicalsError(RuntimeError):
    """Raised when OHLCV is so malformed the score cannot be computed."""


# --- Pure helpers (no I/O) ------------------------------------------------


def sma(values: list[float], window: int) -> Optional[float]:
    """Simple moving average of the last ``window`` values, or None.

    Returns None when fewer than ``window`` values are supplied so the
    caller can decide what to do (typically: bail with an "insufficient
    history" reason rather than produce a meaningless score).
    """
    if window <= 0:
        raise ValueError("sma window must be positive")
    if len(values) < window:
        return None
    return sum(values[-window:]) / float(window)


def rsi(closes: list[float], period: int = RSI_PERIOD) -> Optional[float]:
    """Wilder-style RSI on a list of closes.

    Returns a value in [0, 100] (caller maps to [-1, +1]) or None when
    insufficient history. NaN-safe: any non-finite close is treated as
    a missing data point.
    """
    if period <= 0:
        raise ValueError("rsi period must be positive")
    if len(closes) <= period:
        return None

    # Use the standard "average gain / average loss" with the Wilder
    # smoothing approximation: first average = mean of first ``period``
    # deltas; subsequent averages = (prev_avg * (period-1) + current) / period.
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if not (math.isfinite(prev) and math.isfinite(cur)):
            continue
        delta = cur - prev
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-delta)
    if len(gains) < period:
        return None
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    # Smooth over the remaining deltas.
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        # No losses in the window → RSI is effectively 100. (Avoid div by 0.)
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def realized_vol(closes: list[float], window: int) -> Optional[float]:
    """Realized vol (stddev of log-returns) over the last ``window`` returns.

    Returns None when the window is too short. Log-returns (not simple
    returns) match the standard quant convention.
    """
    if window <= 0:
        raise ValueError("vol window must be positive")
    if len(closes) < window + 1:
        return None
    rets: list[float] = []
    for i in range(len(closes) - window, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0 or cur <= 0 or not (math.isfinite(prev) and math.isfinite(cur)):
            continue
        rets.append(math.log(cur / prev))
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


def _sma_ratio_score(closes: list[float]) -> tuple[float, str]:
    """SMA(20)/SMA(50) ratio → tanh-smoothed score in (-1, +1).

    The ratio is centered at 1.0 (no trend). A 5% spread above the slow
    SMA maps to ~+0.5 via tanh scaling.
    """
    fast = sma(closes, SMA_FAST)
    slow = sma(closes, SMA_SLOW)
    if fast is None or slow is None or slow <= 0:
        return 0.0, "trend: insufficient SMA history"
    ratio = fast / slow
    # Centre at 1.0, scale so ±5% → ±~0.46 (tanh(1) ≈ 0.76).
    centered = (ratio - 1.0) * 20.0
    score = math.tanh(centered)
    return max(-1.0, min(1.0, score)), (
        f"trend: SMA{SMA_FAST}/SMA{SMA_SLOW}={ratio:.4f} → score {score:+.3f}"
    )


def _rsi_score(closes: list[float]) -> tuple[float, str]:
    """RSI(14) → linear map of [30, 70] → [-1, +1]."""
    r = rsi(closes, RSI_PERIOD)
    if r is None:
        return 0.0, f"momentum: insufficient RSI history (need >{RSI_PERIOD} bars)"
    if r <= RSI_OVERSOLD:
        score = -1.0
        tag = "oversold"
    elif r >= RSI_OVERBOUGHT:
        score = 1.0
        tag = "overbought"
    else:
        # Linear in [RSI_OVERSOLD, RSI_OVERBOUGHT] → [-1, +1]
        score = (r - 50.0) / 20.0
        tag = "neutral"
    score = max(-1.0, min(1.0, score))
    return score, f"momentum: RSI({RSI_PERIOD})={r:.1f} ({tag}) → score {score:+.3f}"


def _volatility_score(closes: list[float]) -> tuple[float, str]:
    """20-day vs 60-day realized vol → regime signal in [-0.5, +0.5].

    * Expanding vol + price falling → -0.5 (danger)
    * Expanding vol + price rising → +0.2 (eager)
    * Contracting vol → 0 (calm)
    * No clear regime → 0
    """
    short_v = realized_vol(closes, VOL_SHORT)
    long_v = realized_vol(closes, VOL_LONG)
    if short_v is None or long_v is None or long_v <= 0:
        return 0.0, "volatility: insufficient history for regime"
    ratio = short_v / long_v
    # 20-day return sign over the same window as the short vol.
    if len(closes) < VOL_SHORT + 1:
        return 0.0, "volatility: insufficient history for price-change"
    ret = closes[-1] / closes[-1 - VOL_SHORT] - 1.0
    if ratio < (1.0 / VOL_EXPANSION_RATIO):
        return 0.0, f"volatility: contracting (ratio={ratio:.2f}) → score 0.000"
    if ratio < VOL_EXPANSION_RATIO:
        return 0.0, f"volatility: steady (ratio={ratio:.2f}) → score 0.000"
    # Expanding.
    if ret < 0:
        return -0.5, (
            f"volatility: expanding (ratio={ratio:.2f}) + price falling "
            f"({ret:+.1%}) → score -0.500"
        )
    if ret > 0:
        return 0.2, (
            f"volatility: expanding (ratio={ratio:.2f}) + price rising "
            f"({ret:+.1%}) → score +0.200"
        )
    return 0.0, f"volatility: expanding (ratio={ratio:.2f}) + flat → score 0.000"


# --- Public compute --------------------------------------------------------


def compute_technical_score(
    ticker: str,
    *,
    closes: list[float],
    asof_date: str = "",
) -> TechnicalScore:
    """Compute the combined technical score for ``ticker`` from price history.

    ``closes`` is the list of close prices in chronological order (oldest
    first). Returns a :class:`TechnicalScore` with the per-component
    sub-scores, the weighted aggregate, and human-readable reasons.

    Never raises: if there isn't enough history, the per-component
    sub-scores default to 0.0 and the reasons explain why.
    """
    if not closes:
        return TechnicalScore(
            ticker=ticker.upper(),
            trend=0.0,
            momentum=0.0,
            volatility=0.0,
            score=0.0,
            reasons=["no price history supplied"],
            asof_date=asof_date,
        )

    trend, trend_reason = _sma_ratio_score(closes)
    momentum, momentum_reason = _rsi_score(closes)
    volatility, vol_reason = _volatility_score(closes)

    score = (
        WEIGHT_TREND * trend
        + WEIGHT_MOMENTUM * momentum
        + WEIGHT_VOLATILITY * volatility
    )
    score = max(-1.0, min(1.0, score))

    return TechnicalScore(
        ticker=ticker.upper(),
        trend=trend,
        momentum=momentum,
        volatility=volatility,
        score=score,
        reasons=[trend_reason, momentum_reason, vol_reason],
        asof_date=asof_date,
    )


# --- yfinance fetch (the only network call) --------------------------------


def fetch_ohlcv(ticker: str, *, period: str = YF_PERIOD, interval: str = YF_INTERVAL) -> list[float]:
    """Fetch 6 months of daily close prices for ``ticker`` via yfinance.

    Returns the closes in chronological order (oldest first). Returns
    an empty list on any failure — the caller treats that as "no
    history" and produces a zero score with the right reason.

    Tests monkeypatch this function. Production code calls it once per
    ticker per day, then writes the result to the technical cache.
    """
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - import-time failure
        log.warning("technicals: yfinance import failed for %s: %s", ticker, type(e).__name__)
        return []

    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception as e:
        log.warning("technicals: yfinance fetch failed for %s: %s", ticker, type(e).__name__)
        return []

    if df is None or df.empty:
        log.debug("technicals: yfinance returned empty frame for %s", ticker)
        return []

    # Prefer "Adj Close" so dividends/splits don't fake a trend; fall
    # back to "Close" for tickers where adjusted isn't available.
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if col not in df.columns:
        log.debug("technicals: no close column in yfinance frame for %s", ticker)
        return []
    closes = [float(v) for v in df[col].tolist() if v is not None and math.isfinite(float(v))]
    # DEBUG only — card 6 spec says no raw OHLCV at INFO.
    log.debug("technicals: %s fetched %d closes (last=%.2f)", ticker, len(closes), closes[-1])
    return closes


__all__ = [
    "WEIGHT_TREND",
    "WEIGHT_MOMENTUM",
    "WEIGHT_VOLATILITY",
    "SMA_FAST",
    "SMA_SLOW",
    "RSI_PERIOD",
    "VOL_SHORT",
    "VOL_LONG",
    "VOL_EXPANSION_RATIO",
    "RSI_OVERSOLD",
    "RSI_OVERBOUGHT",
    "YF_PERIOD",
    "YF_INTERVAL",
    "TechnicalScore",
    "TechnicalsError",
    "sma",
    "rsi",
    "realized_vol",
    "compute_technical_score",
    "fetch_ohlcv",
]

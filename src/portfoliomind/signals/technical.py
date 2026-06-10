"""Technical indicators for the strategy engine (card 6).

Four indicators computed from daily OHLCV:

1. **50/200 SMA golden cross** — 50-day SMA > 200-day SMA. The
   "golden cross" is the classic long-term trend confirmation.
2. **20-day high breakout** — today's close > the highest close in
   the prior 20 days (we use the last 20 closes excluding today
   to avoid comparing the close to itself).
3. **MACD bullish crossover (12/26/9)** — MACD line above signal
   line at the most recent bar. The 12/26/9 defaults are the spec.
4. **RSI(14) not overbought** — RSI(14) < 70. A bullish indicator
   only when the stock is NOT overbought. A stock with RSI=85 is
   extended and risky to chase.

Each indicator is exposed both as a pure function (for unit tests
that feed synthetic OHLCV) and as part of the high-level
:func:`compute_technical_signal` (which handles the yfinance pull +
caching + combine-into-TechnicalSignal).

The yfinance path is lazy-imported inside the function so a unit
test that only exercises the pure indicator math never imports
yfinance (the tests assert that).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

from ..logging_setup import get_logger
from ..time_utils import now_bogota
from .price_cache import (
    PRICE_HISTORY_DAYS,
    PRICE_TTL_SECONDS,
    PriceCache,
    PriceCacheError,
)

log = get_logger(__name__)


#: Number of days of history to pull. We need 200+ for the SMA(200),
#: so 365d gives us buffer for holidays / non-trading days.
DEFAULT_LOOKBACK_DAYS: int = PRICE_HISTORY_DAYS

#: yfinance is occasionally slow — the timeout below caps a single
#: ticker's pull. A 30s ceiling is generous; if a ticker is stuck
#: longer, the run moves on (fail-soft).
YFINANCE_TIMEOUT_S: float = 30.0


# --- Public dataclass ------------------------------------------------------


@dataclass(frozen=True)
class TechnicalSignal:
    """The 4 indicator booleans + the underlying numbers.

    All booleans default to ``False`` so a half-initialized
    :class:`TechnicalSignal` (e.g. from a yfinance timeout) is
    safely "no bullish evidence".

    The dataclass is the public contract — card 7 reads it
    directly, and the demo script prints it.

    Attributes
    ----------
    ticker:
        Upper-cased ticker symbol.
    as_of_date:
        ``YYYY-MM-DD`` (Bogota) — the trading day the indicators
        are computed "as of".
    sma_golden_cross:
        ``True`` iff SMA(50) > SMA(200) on ``as_of_date``.
    twenty_day_breakout:
        ``True`` iff today's close > the highest close in the
        prior 20 days (excluding today).
    macd_bullish:
        ``True`` iff MACD(12,26,9) line > signal line at the
        most recent bar.
    rsi_not_overbought:
        ``True`` iff RSI(14) < 70. A bullish precondition, not
        a bullish signal in itself.
    bullish_count:
        Sum of the 4 booleans (0..4). Card 7 may use this directly
        for the "high conviction vs moderate" classification.
    sma_50:
        The SMA(50) value on ``as_of_date`` (NaN when there is
        insufficient history).
    sma_200:
        The SMA(200) value on ``as_of_date`` (NaN when there is
        insufficient history).
    rsi_14:
        The RSI(14) value on ``as_of_date``.
    macd:
        The MACD line value on ``as_of_date``.
    macd_signal:
        The MACD signal line value on ``as_of_date``.
    close:
        The closing price on ``as_of_date``. This is the "entry
        price" reference for card 7 (yesterday's close, because
        the morning run fires at 09:30 ET and the open is just
        starting).
    """

    ticker: str
    as_of_date: str
    sma_golden_cross: bool
    twenty_day_breakout: bool
    macd_bullish: bool
    rsi_not_overbought: bool
    bullish_count: int
    sma_50: float
    sma_200: float
    rsi_14: float
    macd: float
    macd_signal: float
    close: float

    def to_dict(self) -> dict:
        return asdict(self)


# --- Pure indicator functions ----------------------------------------------


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average. Returns a ``Series`` of the same length.

    The first ``window-1`` entries are NaN by construction — the
    caller is expected to drop or ``.iloc[-1]`` to read the most
    recent value.
    """
    return series.rolling(window=window, min_periods=window).mean()


def indicator_sma_golden_cross(close: pd.Series) -> bool:
    """True iff SMA(50) > SMA(200) at the most recent bar.

    Parameters
    ----------
    close:
        A ``pandas.Series`` of daily close prices, oldest first,
        with at least 200 non-NaN observations.

    Returns
    -------
    bool
        ``True`` when SMA(50) > SMA(200). ``False`` otherwise,
        including the case where either SMA is NaN (insufficient
        history).
    """
    if close is None or len(close) < 200:
        return False
    s50 = _sma(close, 50).iloc[-1]
    s200 = _sma(close, 200).iloc[-1]
    if pd.isna(s50) or pd.isna(s200):
        return False
    return bool(s50 > s200)


def indicator_twenty_day_breakout(close: pd.Series) -> bool:
    """True iff the most recent close is strictly greater than the
    maximum of the prior 20 closes.

    We slice ``close.iloc[-21:-1]`` (the 20 trading days BEFORE
    today) to compute the reference high. Comparing today's close
    to the 20-day window INCLUDING today would let the close
    trivially equal itself and make the indicator always bullish
    on a daily-bars series.

    Requires at least 21 bars.
    """
    if close is None or len(close) < 21:
        return False
    today = close.iloc[-1]
    prior_20 = close.iloc[-21:-1]
    if pd.isna(today) or prior_20.isna().any():
        return False
    return bool(today > prior_20.max())


def indicator_macd_bullish(
    close: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> bool:
    """True iff MACD line > signal line at the most recent bar.

    Uses the standard 12/26/9 parameters. Requires at least
    ``slow + signal`` (i.e. 35) bars.
    """
    if close is None or len(close) < slow + signal:
        return False
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    m = macd_line.iloc[-1]
    s = signal_line.iloc[-1]
    if pd.isna(m) or pd.isna(s):
        return False
    return bool(m > s)


def indicator_rsi_not_overbought(close: pd.Series, *, period: int = 14) -> bool:
    """True iff RSI(14) is below the overbought threshold (70).

    Computed with the standard Wilder smoothing (exponential).
    Requires at least ``period + 1`` bars so the first delta is
    defined.

    Note: this is a *precondition* for a bullish interpretation,
    not a bullish signal in itself. A stock with RSI=30 is also
    "not overbought" but may be in free-fall. The combined signal
    in :mod:`portfoliomind.signals.combined` is the right place
    to read the gate.
    """
    if close is None or len(close) < period + 1:
        return False
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing: equivalent to ewm with alpha=1/period, adjust=False.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    last = rsi.iloc[-1]
    if pd.isna(last):
        return False
    return bool(last < 70.0)


def indicator_buy(close: pd.Series) -> int:
    """Aggregated bullish count from the 4 individual indicators.

    A convenience helper for tests and one-off scripts. Production
    callers should use :func:`compute_technical_signal` which also
    returns the underlying numbers + handles the yfinance pull.

    Returns the count in ``0..4``.
    """
    return int(
        indicator_sma_golden_cross(close)
        + indicator_twenty_day_breakout(close)
        + indicator_macd_bullish(close)
        + indicator_rsi_not_overbought(close)
    )


# --- yfinance + cache integration ------------------------------------------


def _bars_to_df(bars: list[dict]) -> pd.DataFrame:
    """Convert a list of bar dicts (from :class:`PriceCache`) to a DataFrame.

    The DataFrame has a DatetimeIndex (oldest first) and columns
    ``open, high, low, close, volume`` (+ ``adj_close`` if present).
    Empty input returns an empty DataFrame with the same columns.
    """
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(bars)
    if "date" not in df.columns:
        raise ValueError("bar dicts must include a 'date' key")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Coerce numeric types defensively (caller may have stored strings).
    for col in ("open", "high", "low", "close", "volume", "adj_close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_yfinance_bars(
    ticker: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    timeout_s: float = YFINANCE_TIMEOUT_S,
) -> list[dict]:
    """Pull daily OHLCV from yfinance and return as a list of dicts.

    Returns ``[]`` (and logs WARNING) on any failure — the caller
    must check the return value and skip the ticker. We deliberately
    do NOT raise: the strategy engine is fail-soft per the card
    spec.

    Lazy-imports ``yfinance`` so a unit test that only exercises
    the pure indicator math never imports it.
    """
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except ImportError as e:
        log.warning("technical: yfinance not installed (%s) — skipping %s", e, ticker)
        return []

    try:
        # ``period`` is the yfinance-native knob; 1y is the spec.
        # The yfinance API supports ``period="1y"`` or a date range.
        # We use the period form for simplicity and accept the
        # calendar-day approximation — 365d gives > 252 trading
        # days, well past the 200 needed for SMA(200).
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=False,  # keep raw OHLCV; we use 'close'
            timeout=timeout_s,
            raise_errors=False,
        )
    except Exception as e:  # noqa: BLE001 — yfinance raises a zoo
        log.warning(
            "technical: yfinance pull failed for %s (%s) — skipping",
            ticker,
            type(e).__name__,
        )
        return []

    if df is None or df.empty:
        log.warning("technical: yfinance returned no data for %s — skipping", ticker)
        return []

    out: list[dict] = []
    for ts, row in df.iterrows():
        try:
            date_str = ts.strftime("%Y-%m-%d")
        except AttributeError:
            # pandas Timestamp
            date_str = pd.Timestamp(ts).strftime("%Y-%m-%d")
        try:
            close = float(row["Close"])
            open_ = float(row["Open"])
            high = float(row["High"])
            low = float(row["Low"])
            volume = int(row["Volume"]) if row["Volume"] == row["Volume"] else 0
        except (KeyError, ValueError, TypeError):
            continue
        entry: dict = {
            "date": date_str,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        if "Adj Close" in row and row["Adj Close"] == row["Adj Close"]:
            try:
                entry["adj_close"] = float(row["Adj Close"])
            except (ValueError, TypeError):
                pass
        out.append(entry)
    return out


def _empty_signal(ticker: str, as_of_date: str) -> TechnicalSignal:
    """The all-False "no bullish evidence" signal for a ticker we
    could not compute.

    Used when yfinance fails or the history is too short. Keeping
    the close=0.0 sentinel is intentional — the demo script can
    spot these rows and warn the operator.
    """
    return TechnicalSignal(
        ticker=ticker,
        as_of_date=as_of_date,
        sma_golden_cross=False,
        twenty_day_breakout=False,
        macd_bullish=False,
        rsi_not_overbought=False,
        bullish_count=0,
        sma_50=float("nan"),
        sma_200=float("nan"),
        rsi_14=float("nan"),
        macd=float("nan"),
        macd_signal=float("nan"),
        close=0.0,
    )


def compute_technical_signal(
    ticker: str,
    *,
    as_of_date: str | None = None,
    cache: Optional[PriceCache] = None,
    max_age_s: int = PRICE_TTL_SECONDS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> TechnicalSignal:
    """Compute the :class:`TechnicalSignal` for ``ticker`` on
    ``as_of_date`` (default: today in Bogota).

    The yfinance pull is cached in :class:`PriceCache` for
    ``max_age_s`` seconds; a fresh-enough cache hit skips the
    network entirely.

    On any failure (yfinance error, no data, short history), the
    returned signal has all 4 booleans ``False`` and a 0.0 close
    — the combined gate in :mod:`portfoliomind.signals.combined`
    will drop the ticker for being below the bullish threshold.

    Parameters
    ----------
    ticker:
        Upper-cased ticker symbol. We upper-case defensively.
    as_of_date:
        ``YYYY-MM-DD`` string. Default: today in Bogota.
    cache:
        Optional :class:`PriceCache` instance. When ``None`` a
        fresh default one is created (the morning run passes
        its own to share state).
    max_age_s:
        Cache freshness window. The default (1h) is the spec.
    lookback_days:
        How many calendar days of history to pull. Default 365
        gives well over 252 trading days, comfortable for
        SMA(200).
    """
    ticker = ticker.upper()
    if as_of_date is None:
        as_of_date = now_bogota().strftime("%Y-%m-%d")
    if cache is None:
        cache = PriceCache()

    # 1. Cache lookup. A miss or a stale row triggers the network pull.
    bars = cache.fetch_bars(ticker=ticker, as_of_date=as_of_date, max_age_s=max_age_s)
    if bars is None:
        bars = _fetch_yfinance_bars(ticker, lookback_days=lookback_days)
        if not bars:
            log.warning(
                "technical: no price data for %s on %s — returning empty signal",
                ticker,
                as_of_date,
            )
            return _empty_signal(ticker, as_of_date)
        try:
            cache.store_bars(ticker=ticker, as_of_date=as_of_date, bars=bars)
        except PriceCacheError as e:  # pragma: no cover
            log.debug("technical: cache store failed for %s: %s", ticker, e)

    # 2. Trim to data up to as_of_date. The pull may have given us
    # today's partial bar — we want yesterday's close + indicators
    # computed at the close of as_of_date. Spec: "Use yesterday's
    # close." If as_of_date is today, the latest bar IS today's
    # open-session, not a close; we want the most recent CLOSED bar
    # (i.e. everything <= as_of_date).
    df = _bars_to_df(bars)
    if df.empty:
        return _empty_signal(ticker, as_of_date)
    # Filter to bars up to and including as_of_date.
    as_of_ts = pd.Timestamp(as_of_date)
    df = df[df.index <= as_of_ts]
    if df.empty:
        return _empty_signal(ticker, as_of_date)
    # yfinance returns the CURRENT (partial) bar with NaN close. We
    # only want CLOSED bars; drop any row whose close is NaN so the
    # indicator math runs on real settlement prices. The
    # "yesterday's close" semantics from the spec are honored
    # implicitly: today's partial bar (if any) is dropped here.
    df = df.dropna(subset=["close"])
    if df.empty:
        return _empty_signal(ticker, as_of_date)
    close = df["close"].astype(float)

    # 3. Compute the 4 indicator booleans.
    sma_gc = indicator_sma_golden_cross(close)
    breakout = indicator_twenty_day_breakout(close)
    macd_bull = indicator_macd_bullish(close)
    rsi_ok = indicator_rsi_not_overbought(close)

    # 4. Underlying numbers for the demo / card 7.
    def _last(s: pd.Series) -> float:
        v = s.iloc[-1]
        return float(v) if not pd.isna(v) else float("nan")

    s50 = _sma(close, 50)
    s200 = _sma(close, 200)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
    # Recompute RSI for the demo's number display. MUST match
    # ``indicator_rsi_not_overbought``'s formula exactly — in
    # particular, the loss series is negated. Do NOT refactor one
    # without the other; see the unit tests in
    # ``tests/test_signals_technical.py`` which pin both.
    _rsi_delta = close.diff()
    _rsi_gain = _rsi_delta.clip(lower=0.0)
    _rsi_loss = -_rsi_delta.clip(upper=0.0)
    _rsi_avg_gain = _rsi_gain.ewm(alpha=1.0 / 14, adjust=False).mean()
    _rsi_avg_loss = _rsi_loss.ewm(alpha=1.0 / 14, adjust=False).mean()
    _rsi_rs = _rsi_avg_gain / _rsi_avg_loss.replace(0, pd.NA)
    rsi_series = 100.0 - (100.0 / (1.0 + _rsi_rs))

    bullish_count = int(sma_gc) + int(breakout) + int(macd_bull) + int(rsi_ok)

    return TechnicalSignal(
        ticker=ticker,
        as_of_date=as_of_date,
        sma_golden_cross=sma_gc,
        twenty_day_breakout=breakout,
        macd_bullish=macd_bull,
        rsi_not_overbought=rsi_ok,
        bullish_count=bullish_count,
        sma_50=_last(s50),
        sma_200=_last(s200),
        rsi_14=_last(rsi_series),
        macd=_last(macd_line),
        macd_signal=_last(macd_signal_line),
        close=_last(close),
    )


__all__ = [
    "TechnicalSignal",
    "DEFAULT_LOOKBACK_DAYS",
    "YFINANCE_TIMEOUT_S",
    "compute_technical_signal",
    "indicator_sma_golden_cross",
    "indicator_twenty_day_breakout",
    "indicator_macd_bullish",
    "indicator_rsi_not_overbought",
    "indicator_buy",
]

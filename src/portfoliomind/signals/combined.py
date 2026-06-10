"""Combined bullish+news signal ranking (card 6).

Public entry point: :func:`score_universe`.

This module is the gate. A ticker is a candidate when BOTH:

1. At least :data:`MIN_TECHNICAL_BULLISH` of the 4 technical
   indicators are bullish ("some bullish evidence" — the loose
   gate).
2. News sentiment is strictly greater than :data:`MIN_NEWS_SENTIMENT`
   ("positive" — the strict gate).

The two gates are intentionally **AND**, not weighted. A ticker with
3+ technical bullish + negative news is dropped (don't fight the
headline tape). A ticker with strong positive news + 1/4 technical
is dropped (no price confirmation).

Weighting (operator-preference, documented in the module docstring):

* Technicals carry 0.6 of the combined score.
* News sentiment carries 0.4 of the combined score.

The combined score is a ranking tool, not a gate — once a ticker
passes the AND-of-two gate above, the combined score orders the
candidates for the operator's review (card 7 / Discord approval).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence

from ..logging_setup import get_logger
from ..universe import UNIVERSE
from .price_cache import PriceCache
from .technical import TechnicalSignal, compute_technical_signal

log = get_logger(__name__)


# --- Constants: the gate ---------------------------------------------------

#: A ticker must have at least this many of 4 technical indicators
#: bullish to pass the technical gate. 2/4 is the spec's "some
#: bullish evidence" floor.
MIN_TECHNICAL_BULLISH: int = 2

#: News sentiment must be strictly greater than this threshold to
#: pass the news gate. +0.2 is the spec's "positive" floor.
MIN_NEWS_SENTIMENT: float = 0.2

#: Weight of the technical component in the combined score. The
#: operator's stated preference: technicals (price action) are the
#: primary signal.
WEIGHT_TECHNICAL: float = 0.6

#: Weight of the news component in the combined score. The operator's
#: stated preference: news is a confirmation filter, not a primary
#: driver.
WEIGHT_NEWS: float = 0.4

#: The default strategy label. Card 7 (sizer + Discord approval)
#: uses this to pick order-spec parameters.
STRATEGY: str = "swing-bullish-news"

#: The default timeframe. The morning run uses daily bars + a multi-
#: day holding window. "swing" reflects that.
TIMEFRAME: str = "swing"


# --- Public dataclass ------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A single ranked candidate for card 7 (sizer + Discord approval).

    This is part of the public contract — card 7 reads every field
    of every Candidate. The dataclass is frozen so an operator
    reviewing the list cannot accidentally mutate it.

    Attributes
    ----------
    ticker:
        Upper-cased ticker symbol.
    strategy:
        The strategy label, e.g. ``"swing-bullish-news"``. Card 7
        uses this to look up order-spec parameters.
    timeframe:
        The holding timeframe, e.g. ``"swing"``. Card 7 may use
        this to choose stop-loss / take-profit multipliers.
    entry_price:
        Reference entry price (yesterday's close). The morning run
        uses this as the basis for stop-loss / take-profit math in
        card 7; the actual fill price may differ because the market
        opens after the run.
    technical_score:
        Number of bullish technical indicators, normalized to
        ``[0, 1]`` (i.e. ``bullish_count / 4``).
    news_score:
        The news sentiment in ``[-1, +1]`` from card 5.
    combined_score:
        ``technical_score * 0.6 + news_score * 0.4``. This is the
        rank key. Always in ``[-0.4, 0.6 + 0.4]`` = ``[-0.4, 1.0]``
        in practice (a -1 news score with 0 technical gives -0.4,
        a +1 news score with 4/4 technical gives 1.0). In practice
        the AND-of-two gate already keeps the range tighter.
    top_signal_reason:
        A short string naming the highest-scoring technical
        indicator. Card 7 may include this in the Discord approval
        message so the operator can quickly see WHY the ticker is
        on the list. Examples: ``"MACD bullish crossover"``,
        ``"20-day high breakout"``, ``"50>200 SMA + breakout"``,
        ``"multi-indicator confluence"``.
    technical_signal:
        The full :class:`TechnicalSignal` (4 booleans + numbers).
        Included for operator inspection / debugging; card 7 may
        also read it.
    """

    ticker: str
    strategy: str
    timeframe: str
    entry_price: float
    technical_score: float
    news_score: float
    combined_score: float
    top_signal_reason: str
    technical_signal: TechnicalSignal

    def to_dict(self) -> dict:
        d = asdict(self)
        # The TechnicalSignal is nested as a dict; flatten for sheet writes.
        d["technical_signal"] = self.technical_signal.to_dict()
        return d


# --- Reason helpers --------------------------------------------------------


_REASON_BY_FLAG: tuple[tuple[str, str], ...] = (
    ("twenty_day_breakout", "20-day high breakout"),
    ("sma_golden_cross", "50>200 SMA golden cross"),
    ("macd_bullish", "MACD bullish crossover"),
    ("rsi_not_overbought", "RSI not overbought (<70)"),
)


def _top_signal_reason(sig: TechnicalSignal) -> str:
    """Pick the highest-conviction technical reason for the operator.

    The priority order is the operator's stated preference:
    breakout > golden cross > MACD > RSI. A multi-flag signal gets
    a "confluence" message; a single-flag signal gets the
    indicator's name.
    """
    if sig.bullish_count == 0:
        return "no technical signal"
    if sig.bullish_count >= 3:
        return f"{sig.bullish_count}/4 technical confluence"
    # 1 or 2 flags. Return the highest-priority flag, and mention
    # any co-occurring ones.
    for key, label in _REASON_BY_FLAG:
        if getattr(sig, key):
            if sig.bullish_count == 1:
                return label
            others = sum(
                1 for k, _ in _REASON_BY_FLAG if k != key and getattr(sig, k)
            )
            return f"{label} + {others} other{'s' if others > 1 else ''}"
    return "no technical signal"


# --- Core API --------------------------------------------------------------


def _normalize_tickers(tickers: Iterable[str] | None) -> tuple[str, ...]:
    """Upper-case + dedupe + preserve order. ``None`` = full UNIVERSE."""
    if tickers is None:
        return UNIVERSE
    seen: set[str] = set()
    out: list[str] = []
    for t in tickers:
        u = t.upper().strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return tuple(out)


def _build_candidate(sig: TechnicalSignal, news_score: float) -> Optional[Candidate]:
    """Build a :class:`Candidate` for ``sig`` if it passes the gate.

    Returns ``None`` when the AND-of-two gate is not met.
    """
    if sig.bullish_count < MIN_TECHNICAL_BULLISH:
        return None
    if news_score <= MIN_NEWS_SENTIMENT:
        return None
    technical_score = sig.bullish_count / 4.0
    combined = (technical_score * WEIGHT_TECHNICAL) + (news_score * WEIGHT_NEWS)
    return Candidate(
        ticker=sig.ticker,
        strategy=STRATEGY,
        timeframe=TIMEFRAME,
        entry_price=sig.close,
        technical_score=technical_score,
        news_score=news_score,
        combined_score=combined,
        top_signal_reason=_top_signal_reason(sig),
        technical_signal=sig,
    )


def score_universe(
    tickers: Sequence[str] | None = None,
    *,
    top_n: int = 10,
    price_cache: Optional[PriceCache] = None,
    news_api_key: Optional[str] = None,
    since_hours: int = 24,
    as_of_date: Optional[str] = None,
) -> list[Candidate]:
    """Compute the top-N combined-signal candidates for the universe.

    Iterates ``tickers`` (default: full :data:`portfoliomind.universe.UNIVERSE`),
    pulls daily OHLCV for each (cached on disk), computes the 4 technical
    indicators, fetches news sentiment (cached on disk per day), and
    returns the top-N :class:`Candidate` records that pass BOTH gates
    (technical >= 2/4 AND news > +0.2), sorted by combined score
    descending.

    Fail-soft: a yfinance failure for one ticker logs WARNING and
    skips that ticker — the run continues for the rest of the
    universe.

    Parameters
    ----------
    tickers:
        The tickers to score. Default: full universe (15 ETFs + 30 stocks).
    top_n:
        How many top candidates to return. Default 10 (the demo's
        default; card 7 may pass a smaller number).
    price_cache:
        Optional :class:`PriceCache` to share with the morning run.
        When ``None`` a default one is created.
    news_api_key:
        OpenAI API key. When ``None`` the env var ``OPENAI_API_KEY``
        is consulted. Required because :func:`score_ticker_sentiment`
        calls the LLM (or its cache).
    since_hours:
        News lookback window in hours. Default 24 (the morning run's
        overnight window).
    as_of_date:
        ``YYYY-MM-DD`` string. Default: today in Bogota. The
        :class:`PriceCache` keys on this; passing a different
        ``as_of_date`` forces a re-pull.

    Returns
    -------
    list[Candidate]
        Sorted by combined score descending. May be shorter than
        ``top_n`` when fewer tickers pass the gate.
    """
    from ..time_utils import now_bogota
    from ..news.store import NewsCache

    tickers = _normalize_tickers(tickers)
    if not tickers:
        return []
    if as_of_date is None:
        as_of_date = now_bogota().strftime("%Y-%m-%d")

    if price_cache is None:
        price_cache = PriceCache()
    news_cache = NewsCache()

    log.info(
        "signals: scoring %d tickers (as_of=%s, since_hours=%d, top_n=%d)",
        len(tickers),
        as_of_date,
        since_hours,
        top_n,
    )

    # 1. Pull sentiment first — one batched LLM call covers the whole
    # universe. If this fails, we still proceed with 0.0 sentiment
    # everywhere (the news gate is then "no news", which drops every
    # ticker — fail-soft, the morning run does not collapse).
    from ..news.sentiment import score_universe_sentiment

    news_scores: dict[str, float] = {}
    try:
        news_scores = score_universe_sentiment(
            tickers=tickers,
            since_hours=since_hours,
            api_key=news_api_key,
            cache=news_cache,
        )
    except Exception as e:  # noqa: BLE001 — openai/network/parse
        log.warning(
            "signals: sentiment scoring failed (%s) — every ticker falls back to 0.0",
            type(e).__name__,
        )
        news_scores = {t: 0.0 for t in tickers}

    # 2. Walk the universe ticker-by-ticker, computing the technical
    # signal. A single yfinance failure must not stop the run.
    candidates: list[Candidate] = []
    for ticker in tickers:
        try:
            sig = compute_technical_signal(
                ticker,
                as_of_date=as_of_date,
                cache=price_cache,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "signals: technical signal failed for %s (%s) — skipping",
                ticker,
                type(e).__name__,
            )
            continue
        if sig.bullish_count == 0 and sig.close == 0.0:
            # The empty-signal sentinel from a yfinance miss.
            log.debug("signals: %s has no price data — skipping", ticker)
            continue

        ns = float(news_scores.get(ticker, 0.0))
        cand = _build_candidate(sig, ns)
        if cand is not None:
            candidates.append(cand)

    # 3. Sort by combined score descending, take top_n.
    candidates.sort(key=lambda c: (-c.combined_score, c.ticker))
    out = candidates[: max(0, int(top_n))]

    log.info(
        "signals: %d/%d tickers passed the AND-of-two gate; returning top %d",
        len(candidates),
        len(tickers),
        len(out),
    )
    return out


__all__ = [
    "Candidate",
    "MIN_TECHNICAL_BULLISH",
    "MIN_NEWS_SENTIMENT",
    "WEIGHT_TECHNICAL",
    "WEIGHT_NEWS",
    "STRATEGY",
    "TIMEFRAME",
    "score_universe",
    "_top_signal_reason",
    "_build_candidate",
    "_normalize_tickers",
]

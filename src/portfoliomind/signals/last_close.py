"""Last close price lookup for the card-7 position sizer.

The card 6 :class:`~portfoliomind.signals.combiner.Signal` does not expose
an ``entry_price`` field — it carries the *signal*, not the price. The
sizer needs the latest close to compute position size, SL, and TP.

Per the card 7 spec, we add a small helper here rather than touching
card 6's dataclass:

* :func:`last_close` wraps the same yfinance call as
  :func:`portfoliomind.signals.technicals.fetch_ohlcv` — the sizer
  reuses the same 6-month daily history, so the price returned here
  is identical to the price card 6's technical score was based on
  (modulo intraday movement).
* On any failure mode (network down, empty frame, missing column) we
  return ``None`` — **never raise**. The sizer converts a ``None`` into
  a :class:`RejectReason` with a clear message; the morning run
  continues with the rest of the candidates.
* The function is **not cached**. Card 6's technical cache pins the
  *signal* (technical score, sentiment, confidence) for the day. The
  sizer's view of the latest close is a separate, intraday-valid
  concern. Caching here would risk stale prices after a big move.

The function signature is intentionally narrow: ticker in, price out
(or ``None``). Anything more complex (mid price, bid/ask spread) is a
future card.
"""

from __future__ import annotations

from typing import Optional

from ..logging_setup import get_logger
from .technicals import fetch_ohlcv

log = get_logger(__name__)


def last_close(ticker: str) -> Optional[float]:
    """Return the latest daily close for ``ticker`` in USD, or ``None`` on failure.

    Mirrors :func:`portfoliomind.signals.technicals.fetch_ohlcv` — same
    yfinance call, same 6-month window, same Adj-Close-preferred
    fallback. The only difference: we return the *last* element of the
    closes list (or ``None`` if the list is empty).

    The helper is intentionally separate from ``fetch_ohlcv`` so the
    sizer doesn't need to import the full technical-score machinery.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        log.warning("last_close: empty ticker")
        return None
    try:
        closes = fetch_ohlcv(ticker)
    except Exception as e:  # noqa: BLE001 — never raise to the caller
        log.warning(
            "last_close: fetch_ohlcv raised for %s: %s", ticker, type(e).__name__
        )
        return None
    if not closes:
        log.debug("last_close: no closes returned for %s", ticker)
        return None
    price = closes[-1]
    # Defensive: yfinance occasionally returns NaN in the tail when
    # the last bar is a half-session. Treat that as "no data".
    try:
        f = float(price)
    except (TypeError, ValueError):
        log.warning("last_close: non-numeric last close for %s: %r", ticker, price)
        return None
    if f != f or f <= 0:  # NaN / non-positive check
        log.warning("last_close: non-positive or NaN last close for %s: %r", ticker, f)
        return None
    return f


__all__ = ["last_close"]

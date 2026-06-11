"""Probabilistic position sizing — the card-7 ``PositionSizer`` seam.

The strategy runner lazy-imports
``portfoliomind.signals.sizer.PositionSizer`` and calls
``sizer.size(candidate)`` per qualified bullish candidate. The sizer
turns a :class:`~portfoliomind.signals.combined.Candidate` into a
fully specified :class:`TradeOrder` (qty, entry, SL, TP) that already
satisfies the XTB iron rules (SL and TP mandatory, on the correct
side of entry — validated via :func:`portfoliomind.xtb.order.validate_order`).

Sizing model (fractional Kelly, vol-anchored stops):

* **Stop distance** — ``stop_pct = clamp(3 × vol_20d, 2%, 8%)``.
  Three daily sigmas keeps normal noise from tagging the stop while
  a regime break exits fast. ``vol_20d`` is the 20-day realized vol
  of daily log-returns from the candidate.
* **Targets** — SL = entry × (1 − stop_pct);
  TP = entry × (1 + reward_risk × stop_pct). Default reward:risk 2:1.
* **Edge** — full Kelly fraction for a binary bet with win
  probability ``p = p_bullish`` and payoff ratio ``b = reward_risk``::

      kelly = p − (1 − p) / b

  A non-positive Kelly means the posterior gives no expected edge at
  this R:R — the candidate is rejected with :class:`SizingError`
  (the runner logs and skips it; the batch continues).
* **Allocation** — ``equity × min(KELLY_FRACTION × kelly,
  MAX_POSITION_FRACTION)``. Quarter-Kelly is the standard guard
  against estimation error in ``p``; the hard 10% cap bounds the
  damage of any single thesis.

Equity comes from the ``PORTFOLIOMIND_EQUITY`` env var (default
$10,000) or the constructor. Whole-share quantities only: if one
share exceeds the allocation, the candidate is rejected rather than
silently over-sized.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

from ..logging_setup import get_logger
from ..time_utils import iso_now
from ..universe import UNIVERSE_ETFS
from ..xtb.order import OrderSpec

log = get_logger(__name__)


# --- Defaults (all overridable via the constructor) ---------------------------
#: Env var the operator sets to their account equity in USD.
EQUITY_ENV_VAR: str = "PORTFOLIOMIND_EQUITY"
DEFAULT_EQUITY: float = 10_000.0

#: Fraction of full Kelly actually deployed (quarter-Kelly).
KELLY_FRACTION: float = 0.25
#: Hard cap on any single position as a fraction of equity.
MAX_POSITION_FRACTION: float = 0.10
#: Take-profit distance as a multiple of the stop distance.
REWARD_RISK: float = 2.0
#: Stop-distance clamps (fraction of entry price).
MIN_STOP_PCT: float = 0.02
MAX_STOP_PCT: float = 0.08
#: Stop distance in daily sigmas of realized vol.
STOP_SIGMAS: float = 3.0


class SizingError(RuntimeError):
    """Raised when a candidate cannot be sized (no edge, too expensive, ...).

    The strategy runner treats this as a per-candidate skip, not a
    batch failure.
    """


@dataclass(frozen=True)
class TradeOrder:
    """A sized, validated, long-only order ready for operator approval.

    ``to_approved_row()`` matches the ``APPROVED_TRADES`` tab headers
    exactly so the approval layer can append it without remapping.
    """

    ticker: str
    side: str
    qty: float
    entry_price: float
    sl: float
    tp: float
    allocation: float
    p_bullish: float
    strategy: str = "bullish-patterns"
    timeframe: str = "MEDIUM"
    note: str = ""
    timestamp: str = field(default_factory=iso_now)

    @property
    def instrument_type(self) -> str:
        return "ETF" if self.ticker.upper() in UNIVERSE_ETFS else "Stock"

    def to_approved_row(self) -> list:
        """Row in APPROVED_TRADES column order:
        Timestamp, Ticker, Type, Strategy, Timeframe, Allocation ($),
        Qty, Entry Price, SL, TP, Approval Note."""
        return [
            self.timestamp,
            self.ticker,
            self.instrument_type,
            self.strategy,
            self.timeframe,
            round(self.allocation, 2),
            self.qty,
            self.entry_price,
            self.sl,
            self.tp,
            self.note,
        ]


class PositionSizer:
    """Fractional-Kelly sizer with vol-anchored stops. Long-only."""

    def __init__(
        self,
        *,
        equity: float | None = None,
        kelly_fraction: float = KELLY_FRACTION,
        max_position_fraction: float = MAX_POSITION_FRACTION,
        reward_risk: float = REWARD_RISK,
        min_stop_pct: float = MIN_STOP_PCT,
        max_stop_pct: float = MAX_STOP_PCT,
    ) -> None:
        if equity is None:
            raw = os.environ.get(EQUITY_ENV_VAR, "").strip()
            try:
                equity = float(raw) if raw else DEFAULT_EQUITY
            except ValueError:
                log.warning("sizer: bad %s=%r — using default", EQUITY_ENV_VAR, raw)
                equity = DEFAULT_EQUITY
        if equity <= 0 or not math.isfinite(equity):
            raise SizingError(f"equity must be a finite number > 0, got {equity!r}")
        if reward_risk <= 0:
            raise SizingError(f"reward_risk must be > 0, got {reward_risk!r}")
        self.equity = float(equity)
        self.kelly_fraction = kelly_fraction
        self.max_position_fraction = max_position_fraction
        self.reward_risk = reward_risk
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct

    # -- internals --------------------------------------------------------

    def _stop_pct(self, candidate: Any) -> float:
        vol = float(getattr(candidate, "vol_20d", 0.0) or 0.0)
        if vol <= 0 or not math.isfinite(vol):
            # No vol estimate → use the midpoint of the clamp band.
            return (self.min_stop_pct + self.max_stop_pct) / 2.0
        return min(max(STOP_SIGMAS * vol, self.min_stop_pct), self.max_stop_pct)

    def kelly(self, p_bullish: float) -> float:
        """Full-Kelly fraction for win-prob ``p_bullish`` at this R:R."""
        p = min(max(float(p_bullish), 0.0), 1.0)
        return p - (1.0 - p) / self.reward_risk

    # -- public -----------------------------------------------------------

    def size(self, candidate: Any) -> TradeOrder:
        """Size one candidate into a validated long :class:`TradeOrder`.

        Raises :class:`SizingError` when the candidate has no positive
        edge, no usable price, or is too expensive for its allocation.
        """
        ticker = str(getattr(candidate, "ticker", "")).upper()
        entry = float(getattr(candidate, "last_close", 0.0) or 0.0)
        if not ticker:
            raise SizingError("candidate has no ticker")
        if entry <= 0 or not math.isfinite(entry):
            raise SizingError(f"{ticker}: no usable last_close ({entry!r})")

        p = float(getattr(candidate, "p_bullish", 0.0) or 0.0)
        kelly = self.kelly(p)
        if kelly <= 0:
            raise SizingError(
                f"{ticker}: no positive edge (p_bullish={p:.3f}, "
                f"R:R={self.reward_risk:.1f} → kelly={kelly:.3f})"
            )

        stop_pct = self._stop_pct(candidate)
        sl = round(entry * (1.0 - stop_pct), 4)
        tp = round(entry * (1.0 + self.reward_risk * stop_pct), 4)

        fraction = min(self.kelly_fraction * kelly, self.max_position_fraction)
        allocation = self.equity * fraction
        qty = math.floor(allocation / entry)
        if qty < 1:
            raise SizingError(
                f"{ticker}: one share at {entry:.2f} exceeds the "
                f"{fraction:.1%} allocation (${allocation:.2f})"
            )

        note = (
            f"p_bullish={p:.3f} kelly={kelly:.3f} (×{self.kelly_fraction:.2f}, "
            f"cap {self.max_position_fraction:.0%}) stop={stop_pct:.1%} "
            f"R:R={self.reward_risk:.1f} patterns={','.join(getattr(candidate, 'patterns', []) or []) or 'none'}"
        )

        # Defense in depth: the XTB iron rules (SL/TP mandatory, correct
        # side of entry) are enforced here, before the order ever
        # reaches the approval sheet.
        OrderSpec.checked(ticker, "BUY", float(qty), entry, sl, tp, note=note)

        order = TradeOrder(
            ticker=ticker,
            side="BUY",
            qty=float(qty),
            entry_price=entry,
            sl=sl,
            tp=tp,
            allocation=qty * entry,
            p_bullish=p,
            note=note,
        )
        log.info(
            "sizer: %s qty=%s entry=%.2f sl=%.2f tp=%.2f alloc=%.2f",
            ticker, qty, entry, sl, tp, order.allocation,
        )
        return order


__all__ = [
    "EQUITY_ENV_VAR",
    "DEFAULT_EQUITY",
    "KELLY_FRACTION",
    "MAX_POSITION_FRACTION",
    "REWARD_RISK",
    "MIN_STOP_PCT",
    "MAX_STOP_PCT",
    "STOP_SIGMAS",
    "SizingError",
    "TradeOrder",
    "PositionSizer",
]

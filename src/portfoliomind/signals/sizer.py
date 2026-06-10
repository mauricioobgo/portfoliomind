"""Position sizer for card 7 — commission-aware, SL/TP-driven.

The sizer converts a card-6 :class:`~portfoliomind.signals.combiner.Signal`
into a :class:`TradeOrder` (qty, entry, SL, TP) or a :class:`RejectReason`
(explanation). The sizer is the highest-stakes code in the project —
real money is on the line — so the logic is conservative, explicit, and
backed by exhaustive unit tests.

Rules (per the operator-approved 2026-06-08 spec):

* **Per-trade dollar cap** (``xtb_per_trade_cap``): notional = qty *
  entry_price must be ``<= xtb_per_trade_cap``.
* **Max open positions** (``xtb_max_open_positions``): the sizer is
  called once per approval with a ``open_position_count`` argument.
  When ``open_position_count >= xtb_max_open_positions``, the
  candidate is rejected.
* **Commission rejection** (``xtb_max_commission_pct``): when the
  round-trip commission exceeds ``xtb_max_commission_pct * notional``,
  the candidate is rejected. This is the $8 minimum kicking in for
  small-cap stock orders.
* **Stop-loss** (``xtb_sl_pct``): ``sl = entry * (1 - xtb_sl_pct)``,
  default 7% below entry.
* **Take-profit** (``xtb_tp_pct``): ``tp = entry * (1 + xtb_tp_pct)``,
  default 14% above entry (2:1 R/R).
* **Long-only.** Negative ``combined`` is filtered upstream by the
  card 6 intent; the sizer does not check direction.
* **Never raise.** A failure inside :meth:`PositionSizer.size` is
  converted into a :class:`RejectReason` with a clear ``reason``
  string. The runner (card 8) treats a reject as a soft skip — the
  rest of the batch continues.

The sizer does NOT call yfinance directly. It receives an ``entry_price``
from the caller (which uses :func:`last_close` from this package). This
keeps the sizer a pure-math module that's easy to test exhaustively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional, Protocol

from ..logging_setup import get_logger
from ..universe import UNIVERSE_ETFS
from .commissions import InstrumentType, XTBCommissionModel, default_model
from .last_close import last_close

log = get_logger(__name__)


# --- Public result dataclasses --------------------------------------------


@dataclass(frozen=True)
class TradeOrder:
    """A sized, ready-to-place trade. Output of :meth:`PositionSizer.size`.

    The fields are the contract the approval + persistence layer reads.
    They are all the inputs the XTB runner needs to fill an order:

    * ``ticker`` — uppercase ticker symbol.
    * ``qty`` — integer (or fractional for ETFs — see
      :attr:`allow_fractional`). Whole-share stocks, fractional ETFs.
    * ``entry`` — limit price in USD (the latest close).
    * ``sl`` — stop-loss in USD (``entry * (1 - xtb_sl_pct)``).
    * ``tp`` — take-profit in USD (``entry * (1 + xtb_tp_pct)``).
    * ``notional`` — ``qty * entry`` in USD. Bounded above by
      ``xtb_per_trade_cap``.
    * ``commission_rt`` — round-trip commission in USD. Used by the
      operator to see the true cost.
    * ``r_r_ratio`` — reward / risk ratio (``(tp - entry) /
      (entry - sl)``). For the defaults this is exactly 2.0; the
      field is exposed so a future sizer can compute it differently.
    * ``instrument`` — :class:`InstrumentType` for downstream
      commission tier handling.
    * ``signal_date`` — YYYY-MM-DD Bogota, the day the signal was
      generated. Used as the dedup key in
      :mod:`portfoliomind.approval.persist`.
    * ``asof_date`` — same as ``signal_date`` (alias for clarity in
      the approval message).
    * ``combined`` / ``confidence`` — the card 6 signal magnitudes.
      Persisted in the approval message but not used for sizing.
    * ``reasons`` — human-readable strings from card 6. The
      approval message embeds these verbatim.
    """

    ticker: str
    qty: float
    entry: float
    sl: float
    tp: float
    notional: float
    commission_rt: float
    r_r_ratio: float
    instrument: InstrumentType
    signal_date: str
    asof_date: str
    combined: float
    confidence: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "qty": self.qty,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "notional": self.notional,
            "commission_rt": self.commission_rt,
            "r_r_ratio": self.r_r_ratio,
            "instrument": self.instrument.value,
            "signal_date": self.signal_date,
            "asof_date": self.asof_date,
            "combined": self.combined,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RejectReason:
    """Why a candidate was not sized. Soft skip — the runner continues.

    The fields are deliberately tiny: just the ticker and a
    human-readable reason. The runner (card 8) logs the reject to
    ``AGENT_LOG`` and proceeds with the next candidate.
    """

    ticker: str
    reason: str


# --- Protocol for the card-6 Signal ----------------------------------------


class _SignalLike(Protocol):
    """Structural type for what we read off the input candidate.

    We don't import :class:`portfoliomind.signals.combiner.Signal` here
    to avoid a layering cycle (sizer is consumed by card 8 which is
    also consumed by card 6's tests). The Protocol is what
    :class:`portfoliomind.signals.combiner.Signal` happens to satisfy
    in production.
    """

    ticker: str
    combined: float
    technical: float
    sentiment: float
    confidence: float
    reasons: list[str]
    error: str
    asof_date: str


# --- The sizer -------------------------------------------------------------


@dataclass(frozen=True)
class PositionSizer:
    """Card 7 position sizer. Pure-math, no network.

    Construction parameters map 1:1 to ``PortfoliomindConfig`` fields.
    Tests construct a sizer with non-default values to exercise the
    edge cases (tiny cap, single-slot remaining, etc.).

    The ``commission_model`` and ``entry_price_fetcher`` arguments are
    the *only* I/O surface; both are injectable for tests. Production
    uses the defaults (module-level singletons).
    """

    per_trade_cap: float = 200.0
    max_open_positions: int = 5
    sl_pct: float = 0.07
    tp_pct: float = 0.14
    max_commission_pct: float = 0.05
    # Allow fractional shares for ETFs. Stocks always use whole shares.
    allow_fractional: bool = True
    # Plug-in seams for tests.
    commission_model: XTBCommissionModel = field(default_factory=default_model)
    entry_price_fetcher: Any = None  # callable(ticker: str) -> Optional[float]
    # Whether to log per-candidate decisions. Off in tests, on in prod.
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.per_trade_cap <= 0:
            raise ValueError(f"per_trade_cap must be > 0, got {self.per_trade_cap}")
        if self.max_open_positions <= 0:
            raise ValueError(
                f"max_open_positions must be > 0, got {self.max_open_positions}"
            )
        if self.sl_pct <= 0 or self.sl_pct >= 1:
            raise ValueError(f"sl_pct must be in (0, 1), got {self.sl_pct}")
        if self.tp_pct <= 0 or self.tp_pct >= 1:
            raise ValueError(f"tp_pct must be in (0, 1), got {self.tp_pct}")
        if self.max_commission_pct <= 0 or self.max_commission_pct >= 1:
            raise ValueError(
                f"max_commission_pct must be in (0, 1), got {self.max_commission_pct}"
            )
        if self.entry_price_fetcher is None:
            object.__setattr__(self, "entry_price_fetcher", last_close)

    # --- Public entry point ----------------------------------------------

    def size(self, candidate: Any, open_position_count: int = 0) -> TradeOrder | RejectReason:
        """Size ``candidate`` into a :class:`TradeOrder` or reject it.

        Parameters
        ----------
        candidate:
            A card-6 :class:`Signal` (or any object exposing the
            ``_SignalLike`` fields).
        open_position_count:
            How many positions are *already* open. Defaults to 0; the
            runner passes the live count from ``EXECUTED_ORDERS``.

        Returns
        -------
        :class:`TradeOrder` on success.
        :class:`RejectReason` on any rejection path (sizing, cap,
        commission, missing price, etc.).

        The function **never raises** — a failure inside is converted
        into a :class:`RejectReason` with a reason string.
        """
        try:
            return self._size_unchecked(candidate, open_position_count)
        except Exception as e:  # noqa: BLE001 — last-ditch: never raise
            ticker = getattr(candidate, "ticker", "?")
            log.warning("sizer: size(%s) raised: %s", ticker, type(e).__name__)
            return RejectReason(ticker=str(ticker), reason=f"sizer error: {e!r}")

    # --- Pure internals ---------------------------------------------------

    def _size_unchecked(
        self, candidate: _SignalLike, open_position_count: int
    ) -> TradeOrder | RejectReason:
        ticker = str(getattr(candidate, "ticker", "")).upper()
        if not ticker:
            return RejectReason(ticker="?", reason="missing ticker")

        if open_position_count >= self.max_open_positions:
            return RejectReason(
                ticker=ticker,
                reason=(
                    f"max_open_positions reached: {open_position_count} >= "
                    f"{self.max_open_positions}"
                ),
            )

        if getattr(candidate, "error", ""):
            return RejectReason(
                ticker=ticker, reason=f"signal has error: {candidate.error}"
            )

        instrument = self._classify_instrument(ticker)
        entry_price = self._fetch_entry_price(ticker)
        if entry_price is None:
            return RejectReason(
                ticker=ticker, reason="entry price unavailable (yfinance returned no data)"
            )
        if entry_price <= 0:
            return RejectReason(ticker=ticker, reason=f"non-positive entry price: {entry_price}")

        sl_price, tp_price = self._compute_sl_tp(entry_price)
        r_r = self._compute_r_r(entry_price, sl_price, tp_price)
        if r_r < 2.0 - 1e-9:
            # Spec: 2:1 R/R floor. The default config satisfies this; we
            # keep the check for the case where an operator tightens SL
            # or TP below the 2:1 ratio.
            return RejectReason(
                ticker=ticker,
                reason=(
                    f"R/R ratio {r_r:.3f} < 2.0 (entry={entry_price:.2f} "
                    f"sl={sl_price:.2f} tp={tp_price:.2f})"
                ),
            )

        # Qty sizing: per_trade_cap / entry_price, rounded to whole
        # shares for stocks and (optionally) fractional for ETFs.
        qty = self._compute_qty(self.per_trade_cap, entry_price, instrument=instrument)
        if qty <= 0:
            return RejectReason(
                ticker=ticker,
                reason=(
                    f"per-trade cap ${self.per_trade_cap:.2f} is below the price of "
                    f"one share (${entry_price:.2f})"
                ),
            )

        notional = qty * entry_price
        commission_rt = float(
            self.commission_model.round_trip(Decimal(str(notional)), instrument)
        )

        # Commission rejection: round-trip > max_commission_pct * notional
        commission_threshold = self.max_commission_pct * notional
        if commission_rt > commission_threshold + 1e-9:
            return RejectReason(
                ticker=ticker,
                reason=(
                    f"round-trip commission ${commission_rt:.2f} exceeds "
                    f"{self.max_commission_pct:.0%} of notional "
                    f"(${notional:.2f} → threshold ${commission_threshold:.2f})"
                ),
            )

        if self.verbose:
            log.info(
                "sizer: sized %s qty=%.4f entry=%.2f sl=%.2f tp=%.2f "
                "notional=%.2f commission_rt=%.2f r_r=%.2f",
                ticker,
                qty,
                entry_price,
                sl_price,
                tp_price,
                notional,
                commission_rt,
                r_r,
            )

        return TradeOrder(
            ticker=ticker,
            qty=qty,
            entry=entry_price,
            sl=sl_price,
            tp=tp_price,
            notional=notional,
            commission_rt=commission_rt,
            r_r_ratio=r_r,
            instrument=instrument,
            signal_date=getattr(candidate, "asof_date", "") or "",
            asof_date=getattr(candidate, "asof_date", "") or "",
            combined=float(getattr(candidate, "combined", 0.0) or 0.0),
            confidence=float(getattr(candidate, "confidence", 0.0) or 0.0),
            reasons=list(getattr(candidate, "reasons", []) or []),
        )

    # --- Math helpers ----------------------------------------------------

    def _classify_instrument(self, ticker: str) -> InstrumentType:
        return InstrumentType.US_ETF if ticker.upper() in UNIVERSE_ETFS else InstrumentType.US_STOCK

    def _fetch_entry_price(self, ticker: str) -> Optional[float]:
        fetcher = self.entry_price_fetcher
        if fetcher is None:
            return None
        try:
            return float(fetcher(ticker))
        except (TypeError, ValueError):
            return None

    def _compute_sl_tp(self, entry: float) -> tuple[float, float]:
        return entry * (1.0 - self.sl_pct), entry * (1.0 + self.tp_pct)

    @staticmethod
    def _compute_r_r(entry: float, sl: float, tp: float) -> float:
        risk = entry - sl
        reward = tp - entry
        if risk <= 0:
            return 0.0
        return reward / risk

    def _compute_qty(
        self, cap_usd: float, entry: float, *, instrument: InstrumentType
    ) -> float:
        raw = cap_usd / entry
        if instrument is InstrumentType.US_STOCK or not self.allow_fractional:
            # Whole shares for stocks. ``int(x)`` floors toward zero; we
            # already know ``raw > 0`` because the caller checks qty>0
            # below.
            return float(int(raw))
        # Fractional shares allowed for ETFs.
        # 4 decimal places is the precision most US brokers accept.
        return float(Decimal(str(raw)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

    # --- Factory helpers -------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "PositionSizer":
        """Build a :class:`PositionSizer` from a :class:`PortfoliomindConfig`.

        Convenience factory so the runner doesn't have to enumerate
        every field. Tests can pass any object that exposes the same
        attributes.
        """
        return cls(
            per_trade_cap=float(getattr(config, "xtb_per_trade_cap", 200.0)),
            max_open_positions=int(getattr(config, "xtb_max_open_positions", 5)),
            sl_pct=float(getattr(config, "xtb_sl_pct", 0.07)),
            tp_pct=float(getattr(config, "xtb_tp_pct", 0.14)),
            max_commission_pct=float(getattr(config, "xtb_max_commission_pct", 0.05)),
        )


__all__ = [
    "TradeOrder",
    "RejectReason",
    "PositionSizer",
]

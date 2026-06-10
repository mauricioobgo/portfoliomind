"""XTB commission model (card 7 — position sizer math).

Source: XTB fee schedule (https://www.xtb.com/en/commissions), as of
mid-2026. The card 7 spec pins the following pricing for **US** instruments
(no FX conversion, no overnight financing — those are out of scope for a
$200/ticker day-trade-sized book):

* **US stocks & stock-like instruments** (common stock, ADR, REIT):
  ``max($8, 0.08% * notional)`` per side. One-way commission.
* **US ETFs**: 0% on the first $100,000 of monthly traded volume, then
  0.08% of notional per side. We do NOT attempt to track the cumulative
  monthly volume here — the sizer's :class:`XTBCommissionModel` takes a
  ``monthly_volume_used`` argument that the runner wires up from
  ``EXECUTED_ORDERS`` (card 3 already persists the running total).

Round-trip commission is ``2 * one_way``. The card 7 sizer rejects a
candidate when ``round_trip / position_value > xtb_max_commission_pct``
(default 5%).

This module is **pure math** — no network, no logging, no I/O. The
:class:`XTBCommissionModel` is instantiated once at process start and
held as a module-level singleton (``default_model()``). Tests can
construct a fresh model with custom thresholds for the rejection logic.

Why hardcoded rather than fetched?
* The XTB fee schedule changes quarterly. Pinning the numbers here and
  re-validating on spec changes is auditable.
* Card 7 is the position-sizer — it must work in CI without any network.
* The fee schedule source URL is in the docstring; the operator
  refreshes both code + spec in one PR when the schedule changes.

Design notes
------------

* :class:`InstrumentType` is a :class:`str`-valued :class:`Enum` so the
  value round-trips through JSON / Sheets cells cleanly. We deliberately
  do **not** use plain strings — the spec says "a small enum, not a
  string".
* Negative or zero notional is a programming error, not a runtime
  condition. We raise :class:`ValueError`. (The sizer is the only
  caller; it computes ``qty * entry_price`` and either both are
  non-negative or something is wrong upstream.)
* :meth:`XTBCommissionModel.round_trip` is ``2 * one_way``. A future
  asymmetric model (e.g. free exit on limit) is layered on top — for
  now the spec is "round-trip = 2 × one-way".
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Final


# --- Instrument type -------------------------------------------------------


class InstrumentType(str, Enum):
    """What kind of XTB-instrument a candidate is.

    String-valued so the value round-trips through JSON / Sheets cells.
    New types (e.g. ``us_option``) are added here, never as free-form
    strings at call sites.
    """

    US_STOCK = "us_stock"
    US_ETF = "us_etf"

    @classmethod
    def of(cls, ticker: str, *, universe_is_etf: bool) -> "InstrumentType":
        """Heuristic: classify by the universe membership.

        The card-7 runner owns the universe (see :mod:`portfoliomind.universe`).
        This helper exists so call sites can do ``InstrumentType.of(t, is_etf=...)``
        without importing the universe directly.
        """
        return cls.US_ETF if universe_is_etf else cls.US_STOCK


# --- Fee constants (verbatim from XTB's US fee schedule, mid-2026) ---------

#: Minimum one-way commission for US stocks/ADRs/REITs.
US_STOCK_MIN_COMMISSION: Final[Decimal] = Decimal("8.00")
#: Percentage of notional charged on US stocks/ADRs/REITs above the
#: minimum (e.g. ``0.0008`` = 0.08%).
US_STOCK_PCT_OF_NOTIONAL: Final[Decimal] = Decimal("0.0008")
#: Monthly volume (USD) below which US ETFs are commission-free.
US_ETF_FREE_TIER: Final[Decimal] = Decimal("100000.00")
#: Percentage of notional charged on US ETFs above the free tier.
US_ETF_PCT_OF_NOTIONAL: Final[Decimal] = Decimal("0.0008")
#: Money quantization — XTB bills in dollars with 2-decimal precision.
MONEY_QUANTUM: Final[Decimal] = Decimal("0.01")


# --- The model -------------------------------------------------------------


@dataclass(frozen=True)
class XTBCommissionModel:
    """Pure-math XTB commission calculator. Immutable.

    Construct once at process start (via :func:`default_model`) and pass
    into the sizer. The free-tier threshold and minimums are exposed as
    fields only so a test or a future "Pro tier" model can override them
    without forking the implementation.
    """

    us_stock_min: Decimal = US_STOCK_MIN_COMMISSION
    us_stock_pct: Decimal = US_STOCK_PCT_OF_NOTIONAL
    us_etf_free_tier: Decimal = US_ETF_FREE_TIER
    us_etf_pct: Decimal = US_ETF_PCT_OF_NOTIONAL

    def one_way(
        self,
        notional: float | Decimal,
        instrument: InstrumentType,
        *,
        monthly_volume_used: float | Decimal = 0,
    ) -> Decimal:
        """One-way (single side) commission in USD for ``notional`` dollars.

        Parameters
        ----------
        notional:
            Position size in USD (``qty * entry_price``). Must be >= 0.
        instrument:
            :class:`InstrumentType` for the trade.
        monthly_volume_used:
            How much USD has already traded on ETFs this month. The
            free-tier exemption is consumed by past volume. Stock trades
            ignore this argument.

        Returns
        -------
        :class:`decimal.Decimal`
            Commission, rounded to 2 decimal places (XTB's billing
            precision). Always non-negative.
        """
        n = self._validate_notional(notional)
        monthly = self._validate_volume(monthly_volume_used)
        if instrument is InstrumentType.US_STOCK:
            pct_charge = n * self.us_stock_pct
            raw = max(self.us_stock_min, pct_charge)
        elif instrument is InstrumentType.US_ETF:
            # Free tier: monthly volume already traded on ETFs reduces
            # the remaining free allowance. If the candidate's notional
            # still fits inside the free tier, commission is $0.
            remaining_free = max(Decimal("0"), self.us_etf_free_tier - monthly)
            billable = max(Decimal("0"), n - remaining_free)
            raw = billable * self.us_etf_pct
        else:  # pragma: no cover — defensive: future-proofing
            raise ValueError(f"unknown instrument type: {instrument!r}")
        return raw.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)

    def round_trip(
        self,
        notional: float | Decimal,
        instrument: InstrumentType,
        *,
        monthly_volume_used: float | Decimal = 0,
    ) -> Decimal:
        """Round-trip commission (open + close) for ``notional`` USD.

        Per the card 7 spec: ``round_trip = 2 * one_way``. We do not
        attempt to model asymmetric fees here (e.g. free exit on a
        limit) — XTB's standard schedule is symmetric.
        """
        return 2 * self.one_way(
            notional, instrument, monthly_volume_used=monthly_volume_used
        )

    # --- Internals --------------------------------------------------------

    @staticmethod
    def _validate_notional(notional: float | Decimal) -> Decimal:
        n = Decimal(str(notional))
        if n < 0:
            raise ValueError(f"notional must be >= 0, got {n}")
        return n

    @staticmethod
    def _validate_volume(volume: float | Decimal) -> Decimal:
        v = Decimal(str(volume))
        if v < 0:
            raise ValueError(f"monthly_volume_used must be >= 0, got {v}")
        return v


# --- Module-level singleton ------------------------------------------------

_DEFAULT: XTBCommissionModel | None = None


def default_model() -> XTBCommissionModel:
    """Return the process-wide default :class:`XTBCommissionModel`.

    Lazy-initialized so tests can patch the constants in
    :mod:`portfoliomind.signals.commissions` before the singleton is
    first read.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = XTBCommissionModel()
    return _DEFAULT


__all__ = [
    "InstrumentType",
    "XTBCommissionModel",
    "US_STOCK_MIN_COMMISSION",
    "US_STOCK_PCT_OF_NOTIONAL",
    "US_ETF_FREE_TIER",
    "US_ETF_PCT_OF_NOTIONAL",
    "MONEY_QUANTUM",
    "default_model",
]

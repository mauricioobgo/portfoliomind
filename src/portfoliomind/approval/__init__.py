"""Suggestions-driven approval — the card-7 seam the strategy runner imports.

The original card-7 design waited for Discord reactions. This
implementation replaces that with a **Google-Sheets standing
mandate**: the operator maintains the ``💡 Suggestions`` tab and the
agent invests on their behalf strictly within it. A sized order is
auto-approved only when ALL of:

* the order's ticker has a row in ``💡 Suggestions``;
* that row's ``Action`` is ``BUY`` (the strategy is long-only);
* that row's ``Status`` is active (``ACTIVE`` / ``APPROVED`` /
  ``YES`` / ``OPEN``, case-insensitive);
* the order's allocation fits the row's ``Max Allocation ($)`` cap —
  the quantity is scaled down to fit; if even one share doesn't fit,
  the order is rejected.

Everything else is rejected with a recorded reason. Every decision is
appended to the ``🗒️ Agent Log`` tab so the audit trail is complete.

Public contract (consumed by :mod:`portfoliomind.strategy_runner`)::

    post_candidates_and_collect_reactions(orders, *, timeout_seconds=1800)
        -> ApprovalOutcome   # .approved / .rejected lists
    persist_approved_trades(orders) -> int   # rows appended

``timeout_seconds`` is accepted for contract compatibility but unused:
the suggestions sheet IS the operator's pre-collected reaction, so
there is nothing to wait for.

Clients: production resolves a :class:`SheetsClient` + sheet ID from
:class:`PortfoliomindConfig`. Tests inject fakes via
:func:`set_clients` / :func:`reset_clients` (the same seam pattern as
the platform runners).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from ..logging_setup import get_logger
from ..sheets.schema import AGENT_LOG, APPROVED_TRADES, SUGGESTIONS
from ..time_utils import iso_now
from ..universe import _normalize_ticker

log = get_logger(__name__)


#: Suggestion ``Status`` values that mean "the mandate is live".
ACTIVE_STATUSES: frozenset[str] = frozenset({"ACTIVE", "APPROVED", "YES", "OPEN"})
#: Suggestion ``Action`` values that authorize a long entry.
BUY_ACTIONS: frozenset[str] = frozenset({"BUY", "LONG", "ACCUMULATE"})


class ApprovalError(RuntimeError):
    """Raised when the approval layer cannot reach the sheet at all."""


@dataclass(frozen=True)
class Suggestion:
    """One row of the operator's standing mandate."""

    ticker: str
    action: str = "BUY"
    max_allocation: Optional[float] = None
    conviction: str = ""
    source: str = ""
    notes: str = ""
    status: str = "ACTIVE"

    def is_active_buy(self) -> bool:
        return (
            self.action.strip().upper() in BUY_ACTIONS
            and self.status.strip().upper() in ACTIVE_STATUSES
        )


@dataclass
class ApprovalOutcome:
    """What the strategy runner reads back: approved + rejected orders."""

    approved: list[Any] = field(default_factory=list)
    rejected: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --- Client resolution (test seam) --------------------------------------------

_sheets_override: Optional[Any] = None
_sheet_id_override: Optional[str] = None


def set_clients(*, sheets: Any = None, sheet_id: Optional[str] = None) -> None:
    """Inject a SheetsClient-shaped fake + sheet ID. Test-only."""
    global _sheets_override, _sheet_id_override
    if sheets is not None:
        _sheets_override = sheets
    if sheet_id is not None:
        _sheet_id_override = sheet_id


def reset_clients() -> None:
    """Restore the production config-driven client path. Test-only."""
    global _sheets_override, _sheet_id_override
    _sheets_override = None
    _sheet_id_override = None


def _resolve_clients(sheets: Any = None, sheet_id: Optional[str] = None) -> tuple[Any, str]:
    """Resolve (sheets_client, sheet_id) from args, test seam, or config."""
    if sheets is None:
        sheets = _sheets_override
    if sheet_id is None:
        sheet_id = _sheet_id_override
    if sheets is not None and sheet_id:
        return sheets, sheet_id

    # Production path — lazy imports keep this module cheap to import.
    from ..config import PortfoliomindConfig
    from ..sheets.client import SheetsClient

    config = PortfoliomindConfig.from_env()
    if sheets is None:
        sheets = SheetsClient.from_config(config)
    if not sheet_id:
        sheet_id = config.google_sheet_id
    if not sheet_id:
        raise ApprovalError(
            "GOOGLE_SHEET_ID is blank — run scripts/bootstrap_sheet.py first"
        )
    return sheets, sheet_id


# --- Internal helpers -----------------------------------------------------------


def _parse_float(raw: str) -> Optional[float]:
    s = str(raw).strip().replace("$", "").replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _log_to_sheet(sheets: Any, sheet_id: str, level: str, message: str) -> None:
    """Best-effort AGENT_LOG append — an audit failure never blocks a decision."""
    try:
        sheets.append_rows(
            sheet_id, AGENT_LOG, [[iso_now(), level, "approval", message]]
        )
    except Exception as e:  # noqa: BLE001
        log.warning("approval: AGENT_LOG append failed: %s", type(e).__name__)


# --- Public API -------------------------------------------------------------------


def read_suggestions(*, sheets: Any = None, sheet_id: Optional[str] = None) -> list[Suggestion]:
    """Read the operator's standing mandate from the ``💡 Suggestions`` tab.

    Malformed rows are skipped with a warning; an unreachable tab
    returns an empty list (no mandate = nothing is approved).
    """
    sheets, sheet_id = _resolve_clients(sheets, sheet_id)
    try:
        rows = sheets.read_range(sheet_id, SUGGESTIONS, "A2:H")
    except Exception as e:  # noqa: BLE001
        log.warning("approval: could not read %s: %s", SUGGESTIONS, type(e).__name__)
        return []

    out: list[Suggestion] = []
    for row in rows or []:
        # Columns: Timestamp, Ticker, Action, Max Allocation ($),
        #          Conviction, Source, Notes, Status
        padded = list(row) + [""] * (8 - len(row))
        ticker = str(padded[1]).strip().upper()
        if not ticker:
            continue
        out.append(
            Suggestion(
                ticker=ticker,
                action=str(padded[2]).strip() or "BUY",
                max_allocation=_parse_float(padded[3]),
                conviction=str(padded[4]).strip(),
                source=str(padded[5]).strip(),
                notes=str(padded[6]).strip(),
                status=str(padded[7]).strip() or "ACTIVE",
            )
        )
    return out


def _decide(order: Any, suggestion: Optional[Suggestion]) -> tuple[Optional[Any], str]:
    """Decide one order against its suggestion.

    Returns ``(approved_order_or_None, reason)``. The approved order
    may be a qty-scaled copy when the suggestion carries an
    allocation cap.
    """
    ticker = str(getattr(order, "ticker", "?")).upper()
    if suggestion is None:
        return None, f"{ticker}: no row in {SUGGESTIONS} — agent has no mandate to buy"
    if not suggestion.is_active_buy():
        return None, (
            f"{ticker}: suggestion is not an active BUY "
            f"(action={suggestion.action!r}, status={suggestion.status!r})"
        )

    cap = suggestion.max_allocation
    allocation = float(getattr(order, "allocation", 0.0) or 0.0)
    entry = float(getattr(order, "entry_price", 0.0) or 0.0)
    qty = float(getattr(order, "qty", 0.0) or 0.0)
    if cap is not None and cap > 0 and allocation > cap and entry > 0:
        new_qty = float(int(cap // entry))
        if new_qty < 1:
            return None, (
                f"{ticker}: one share at {entry:.2f} exceeds the "
                f"suggestion cap ${cap:.2f}"
            )
        if dataclasses.is_dataclass(order):
            order = dataclasses.replace(order, qty=new_qty, allocation=new_qty * entry)
        else:  # pragma: no cover — non-dataclass orders from tests
            order.qty = new_qty
            order.allocation = new_qty * entry
        return order, (
            f"{ticker}: approved, qty scaled {qty:g} → {new_qty:g} to fit "
            f"suggestion cap ${cap:.2f}"
        )
    return order, f"{ticker}: approved against active suggestion ({suggestion.source or 'operator'})"


def post_candidates_and_collect_reactions(
    candidates: list[Any],
    *,
    timeout_seconds: int = 1800,  # noqa: ARG001 — contract compat; the sheet is the standing approval
    sheets: Any = None,
    sheet_id: Optional[str] = None,
) -> ApprovalOutcome:
    """Match sized orders against the suggestions mandate.

    Every decision (approve/reject + reason) is appended to
    ``🗒️ Agent Log``. Raises :class:`ApprovalError` only when the
    sheet itself is unreachable — the strategy runner records that as
    a step error without aborting the morning run.
    """
    sheets, sheet_id = _resolve_clients(sheets, sheet_id)
    suggestions = {s.ticker: s for s in read_suggestions(sheets=sheets, sheet_id=sheet_id)}

    outcome = ApprovalOutcome()
    for order in candidates:
        ticker = _normalize_ticker(str(getattr(order, "ticker", "")))
        approved, reason = _decide(order, suggestions.get(ticker))
        outcome.notes.append(reason)
        if approved is not None:
            outcome.approved.append(approved)
            _log_to_sheet(sheets, sheet_id, "INFO", f"APPROVED — {reason}")
        else:
            outcome.rejected.append(order)
            _log_to_sheet(sheets, sheet_id, "INFO", f"REJECTED — {reason}")

    log.info(
        "approval: %d approved, %d rejected of %d proposed",
        len(outcome.approved),
        len(outcome.rejected),
        len(candidates),
    )
    return outcome


def persist_approved_trades(
    orders: list[Any],
    *,
    sheets: Any = None,
    sheet_id: Optional[str] = None,
) -> int:
    """Append approved orders to ``✅ Approved Trades``. Returns rows appended.

    Dedup-keyed on ``(Ticker, Timestamp)`` — re-running the same batch
    in the same day appends nothing, per the project idempotency rule.
    """
    if not orders:
        return 0
    sheets, sheet_id = _resolve_clients(sheets, sheet_id)

    try:
        existing = sheets.read_range(sheet_id, APPROVED_TRADES, "A2:K")
    except Exception as e:  # noqa: BLE001
        log.warning("approval: could not read %s: %s", APPROVED_TRADES, type(e).__name__)
        existing = []
    seen = {(str(r[1]).upper(), str(r[0])) for r in existing or [] if len(r) >= 2}

    rows: list[list] = []
    for order in orders:
        if hasattr(order, "to_approved_row"):
            row = order.to_approved_row()
        else:
            # Order-shaped objects without the helper (test fakes):
            # build the row from the common attribute names.
            row = [
                getattr(order, "timestamp", iso_now()),
                getattr(order, "ticker", ""),
                getattr(order, "instrument_type", "Stock"),
                getattr(order, "strategy", "bullish-patterns"),
                getattr(order, "timeframe", "MEDIUM"),
                getattr(order, "allocation", ""),
                getattr(order, "qty", ""),
                getattr(order, "entry_price", getattr(order, "entry", "")),
                getattr(order, "sl", getattr(order, "stop_loss", "")),
                getattr(order, "tp", getattr(order, "take_profit", "")),
                getattr(order, "note", ""),
            ]
        key = (str(row[1]).upper(), str(row[0]))
        if key in seen:
            log.info("approval: skipping duplicate approved trade %s", key)
            continue
        seen.add(key)
        rows.append(row)

    if rows:
        sheets.append_rows(sheet_id, APPROVED_TRADES, rows)
    return len(rows)


__all__ = [
    "ACTIVE_STATUSES",
    "BUY_ACTIONS",
    "ApprovalError",
    "ApprovalOutcome",
    "Suggestion",
    "read_suggestions",
    "post_candidates_and_collect_reactions",
    "persist_approved_trades",
    "set_clients",
    "reset_clients",
]

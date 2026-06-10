"""Persist approved trades to the ``APPROVED_TRADES`` Google Sheet (card 7).

The persistence step is the second half of the card 7 approval flow:

1. :func:`post_candidates_and_collect_reactions` posts the sized
   candidates to Discord and collects the operator's reactions.
2. :func:`persist_approved_trades` (this module) appends the
   approved subset to the ``APPROVED_TRADES`` tab.

Dedup is the operator-facing contract: a re-run with the same
candidates must not produce duplicate rows. The dedup key is the
triple ``(Ticker, signal_date, Entry Price)`` — same ticker on a new
day is a new trade, same ticker with a different entry price is a
new trade. The price is rounded to 4 decimal places for stability
(``100.0`` and ``100.00001`` would otherwise look like two different
trades to a float-comparing dedup, but they shouldn't be).

Schema (from :mod:`portfoliomind.sheets.schema.APPROVED_TRADES_HEADERS`):

    Timestamp, Ticker, Type, Strategy, Timeframe, Allocation ($), Qty,
    Entry Price, SL, TP, Approval Note

We fill in:

* ``Timestamp`` — the current Bogota ISO timestamp via
  :func:`portfoliomind.time_utils.iso_now`.
* ``Ticker`` — uppercase.
* ``Type`` — ``"stock"`` or ``"etf"`` from the
  :class:`~portfoliomind.signals.commissions.InstrumentType`.
* ``Strategy`` / ``Timeframe`` — hardcoded ``"card-7"`` / ``"swing"``
  until card 8 wires up a richer strategy label.
* ``Allocation ($)`` — the ``notional`` from the :class:`TradeOrder`.
* ``Qty`` — same as the order.
* ``Entry Price`` / ``SL`` / ``TP`` — the order's prices, formatted
  with 2 decimal places.
* ``Approval Note`` — a short summary of the signal's confidence +
  combined magnitudes so the operator can see the rationale later.

The function **never raises**. A failure is converted into a
:class:`PersistResult` with ``error`` set and ``rows_appended=0``.
The card 8 runner treats a non-zero ``error`` as a soft failure
that does not abort the morning job.

A :class:`SheetsClientError` from the underlying sheets client is
caught and turned into the result. Other unexpected exceptions are
caught by the same try/except and reported as ``error=...`` strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from ..logging_setup import get_logger
from ..sheets.schema import APPROVED_TRADES, TAB_HEADERS
from ..time_utils import iso_now
from .discord import ApprovedTrade  # noqa: F401 — used for type hint / isinstance

log = get_logger(__name__)


# --- Result dataclass -----------------------------------------------------


@dataclass(frozen=True)
class PersistResult:
    """The bag of state :func:`persist_approved_trades` returns.

    Fields:

    * ``rows_appended`` — how many NEW rows were written. Zero is a
      legitimate outcome (everything was a duplicate).
    * ``duplicates_skipped`` — how many incoming trades were already
      in ``APPROVED_TRADES`` and therefore skipped. Surface to the
      operator in the morning summary.
    * ``error`` — empty on success; a non-empty string is the
      first error encountered. The function never raises.
    * ``tab_name`` — the tab we wrote to (always ``APPROVED_TRADES``).
    * ``written_rows`` — the actual rows that went out, in the same
      order as the input. Useful for the dry-run CLI to print them
      back to the operator.
    """

    rows_appended: int = 0
    duplicates_skipped: int = 0
    error: str = ""
    tab_name: str = APPROVED_TRADES
    written_rows: list[list[str]] = field(default_factory=list)


# --- Public entry point ---------------------------------------------------


def persist_approved_trades(
    outcome: Any,
    sheets: Any = None,
    *,
    sheet_id: str = "",
    dry_run: bool = False,
) -> PersistResult:
    """Append the approved subset of ``outcome`` to ``APPROVED_TRADES``.

    Parameters
    ----------
    outcome:
        An :class:`~portfoliomind.approval.discord.ApprovalOutcome` (or
        any object exposing ``.approved`` and ``.rejected`` lists). The
        function only reads ``outcome.approved``; ``rejected`` and
        ``waited`` are ignored (they're for logging only).
    sheets:
        A :class:`~portfoliomind.sheets.client.SheetsClient` (or a
        fake). Must expose ``.append_rows(sheet_id, tab, rows) -> int``
        and ``.read_range(sheet_id, tab, range_a1) -> list[list[str]]``.
    sheet_id:
        The Google Sheet ID. The card 7 CLI passes the one from
        config; the card 8 runner passes the morning context's.
    dry_run:
        If True, format the rows but do NOT call
        :meth:`SheetsClient.append_rows`. Used by ``scripts/approve_trades.py
        --dry-run`` to print the would-be writes for operator review.

    Returns
    -------
    :class:`PersistResult`
        Always. Never raises.
    """
    try:
        return _persist_unchecked(
            outcome, sheets, sheet_id=sheet_id, dry_run=dry_run
        )
    except Exception as e:  # noqa: BLE001 — last-ditch: never raise
        log.error("persist_approved_trades: unexpected error: %s", type(e).__name__)
        return PersistResult(error=f"{type(e).__name__}: {e!r}")


# --- Internals ------------------------------------------------------------


def _persist_unchecked(
    outcome: Any,
    sheets: Any,
    *,
    sheet_id: str,
    dry_run: bool,
) -> PersistResult:
    if outcome is None:
        return PersistResult(error="outcome is None")
    approved = list(getattr(outcome, "approved", []) or [])
    if not approved:
        return PersistResult()  # zero rows, zero error
    if not sheet_id and not dry_run:
        return PersistResult(error="missing sheet_id; pass sheet_id= or dry_run=True")
    if sheets is None and not dry_run:
        return PersistResult(error="missing sheets client; pass sheets= or dry_run=True")

    # Build the candidate rows + dedup keys. We build the rows first
    # so the dry-run path can return them without ever touching Sheets.
    rows: list[list[str]] = []
    keys: list[tuple[str, str, str]] = []
    for trade in approved:
        row = _trade_to_row(trade)
        rows.append(row)
        keys.append(_trade_dedup_key(trade))

    if dry_run:
        log.info(
            "persist: dry-run, would append %d rows to %s", len(rows), APPROVED_TRADES
        )
        return PersistResult(
            rows_appended=0,
            duplicates_skipped=0,
            written_rows=rows,
        )

    # Read existing APPROVED_TRADES and build the dedup set.
    try:
        existing = _read_existing_rows(sheets, sheet_id)
    except Exception as e:  # noqa: BLE001
        return PersistResult(
            error=f"failed to read existing APPROVED_TRADES: {type(e).__name__}: {e}"
        )
    existing_keys: set[tuple[str, str, str]] = {
        _dedup_key_for_row(row) for row in existing
    }

    # Filter out duplicates, keeping the original order.
    new_rows: list[list[str]] = []
    duplicates = 0
    for row, key in zip(rows, keys):
        if key in existing_keys:
            duplicates += 1
            log.info(
                "persist: skipping duplicate %s signal_date=%s entry=%s",
                key[0], key[1], key[2],
            )
            continue
        new_rows.append(row)
        existing_keys.add(key)

    if not new_rows:
        return PersistResult(
            rows_appended=0,
            duplicates_skipped=duplicates,
            written_rows=[],
        )

    try:
        sheets.append_rows(sheet_id, APPROVED_TRADES, new_rows)
    except Exception as e:  # noqa: BLE001
        return PersistResult(
            error=f"append_rows failed: {type(e).__name__}: {e}",
            duplicates_skipped=duplicates,
        )

    log.info(
        "persist: appended %d rows to %s (skipped %d duplicates)",
        len(new_rows), APPROVED_TRADES, duplicates,
    )
    return PersistResult(
        rows_appended=len(new_rows),
        duplicates_skipped=duplicates,
        written_rows=new_rows,
    )


def _trade_to_row(trade: Any) -> list[str]:
    """Map a :class:`ApprovedTrade` to the 11-column row layout."""
    ticker = str(getattr(trade, "ticker", "")).upper()
    instrument = getattr(trade, "instrument", None)
    # The ApprovedTrade has `instrument` as a string value of InstrumentType.
    if hasattr(instrument, "value"):
        type_label = "etf" if instrument.value == "us_etf" else "stock"
    else:
        type_label = "etf" if str(instrument) == "us_etf" else "stock"
    qty = getattr(trade, "qty", 0.0)
    entry = float(getattr(trade, "entry", 0.0) or 0.0)
    sl = float(getattr(trade, "sl", 0.0) or 0.0)
    tp = float(getattr(trade, "tp", 0.0) or 0.0)
    notional = float(getattr(trade, "notional", qty * entry) or 0.0)
    combined = float(getattr(trade, "combined", 0.0) or 0.0)
    confidence = float(getattr(trade, "confidence", 0.0) or 0.0)
    signal_date = str(getattr(trade, "signal_date", "") or getattr(trade, "asof_date", "") or "")
    note = (
        f"card-7 approval: combined={combined:+.2f} "
        f"confidence={confidence:.2f} signal_date={signal_date}"
    )
    return [
        iso_now(),              # Timestamp
        ticker,                 # Ticker
        type_label,             # Type
        "card-7",               # Strategy
        "swing",                # Timeframe
        _money(notional),       # Allocation ($)
        _qty(qty),              # Qty
        _money(entry),          # Entry Price
        _money(sl),             # SL
        _money(tp),             # TP
        note,                   # Approval Note
    ]


def _trade_dedup_key(trade: Any) -> tuple[str, str, str]:
    """Build the dedup triple for an *incoming* trade (spec: ticker, signal_date, entry_price).

    The signal_date comes from the trade's own field (preserved from
    the card 6 Signal), NOT the row's Timestamp (which is the
    wall-clock write time). This is the spec's "stable entry price"
    guarantee: re-running on the same day with the same data
    produces the same dedup key.
    """
    ticker = str(getattr(trade, "ticker", "")).strip().upper()
    signal_date = str(
        getattr(trade, "signal_date", "") or getattr(trade, "asof_date", "") or ""
    ).strip()
    if len(signal_date) >= 10:
        signal_date = signal_date[:10]
    entry = float(getattr(trade, "entry", 0.0) or 0.0)
    return (ticker, signal_date, _money(entry))


def _read_existing_rows(sheets: Any, sheet_id: str) -> list[list[str]]:
    """Read the full APPROVED_TRADES tab, returning the data rows only.

    Empty result (tab not yet present) is fine — the bootstrap in
    card 1 creates it lazily, and an empty tab means no dedup hits.

    Note: the catch-all here is the *tab-not-found* case. A genuine
    Sheets error (auth, 500) propagates to the caller, which records
    it on the result.
    """
    last_col = _LAST_COL
    range_a1 = f"A2:{last_col}"
    try:
        rows = sheets.read_range(sheet_id, APPROVED_TRADES, range_a1)
    except Exception as e:
        # The Sheets client raises ``SheetsClientError`` for "tab not
        # found". For the dedup path we treat that as empty so a
        # first-ever run doesn't fail. Other exceptions are
        # re-raised so the caller can surface them.
        msg = str(e).lower()
        if "not found" in msg or "tab" in msg and "not" in msg:
            return []
        raise
    return [list(r) for r in rows if r and any(cell.strip() for cell in r if isinstance(cell, str))]


def _excel_col_letter(n: int) -> str:
    """Convert 1-indexed column count to Excel column letter (1='A', 11='K')."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


_LAST_COL: str = _excel_col_letter(len(TAB_HEADERS[APPROVED_TRADES]))


def _dedup_key_for_row(row: list[str]) -> tuple[str, str, str]:
    """Build the dedup triple for an APPROVED_TRADES row (existing sheet data).

    Index map (matches :func:`_trade_to_row`):

    * 0 = Timestamp
    * 1 = Ticker
    * 7 = Entry Price
    * 10 = Approval Note — contains ``signal_date=YYYY-MM-DD`` (the
      card 6 signal's asof_date, NOT the row's wall-clock Timestamp).

    The signal_date is recovered from the Approval Note so the dedup
    matches the key produced by :func:`_trade_dedup_key` for incoming
    trades. A legacy row without ``signal_date=`` in its note falls
    back to extracting the date from the Timestamp column (so old
    data still dedups correctly).
    """
    if len(row) < 8:
        return ("", "", "")
    ticker = str(row[1]).strip().upper()
    entry = str(row[7]).strip()
    # Try to extract signal_date from the note first.
    note = str(row[10]).strip() if len(row) > 10 else ""
    signal_date = ""
    for token in note.split():
        if token.startswith("signal_date="):
            signal_date = token.split("=", 1)[1]
            break
    if not signal_date:
        # Fallback: parse the Timestamp (legacy rows / hand-edited rows).
        timestamp = str(row[0]).strip()
        signal_date = timestamp[:10] if len(timestamp) >= 10 else timestamp
    return (ticker, signal_date, entry)


def _money(value: float) -> str:
    """Format a dollar amount with 2 decimal places. None → '0.00'."""
    if value is None:
        return "0.00"
    try:
        return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except Exception:  # noqa: BLE001
        return "0.00"


def _qty(value: float) -> str:
    """Format qty: trim trailing zeros after a 4-dp rounding.

    The sizer uses 4-decimal precision (0.0001 quantize) for
    fractional shares. The persistence layer trims trailing zeros so
    the cell reads naturally: 2.0 → "2", 1.5 → "1.5",
    1.3333 → "1.3333". 0 → "0".
    """
    if value is None:
        return "0"
    try:
        d = Decimal(str(value))
        quantized = d.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        if quantized == quantized.to_integral_value():
            return str(int(quantized))
        # Strip trailing zeros without going to scientific notation.
        s = format(quantized, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s or "0"
    except Exception:  # noqa: BLE001
        return "0"


__all__ = ["PersistResult", "persist_approved_trades"]

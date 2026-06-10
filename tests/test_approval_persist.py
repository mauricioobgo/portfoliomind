"""Hermetic tests for :mod:`portfoliomind.approval.persist`.

These tests use a hand-rolled :class:`FakeSheetsClient` that records
every read/write and returns canned data. The dedup logic is the
heart of the test: a re-run with the same candidates must write zero
new rows.

The tests do NOT touch the real :class:`SheetsClient` (no Google
API). They construct the persist result directly and assert the
row counts, error states, and dedup behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from portfoliomind.approval.discord import ApprovalOutcome, ApprovedTrade
from portfoliomind.approval.persist import (
    _dedup_key_for_row,
    _trade_to_row,
    persist_approved_trades,
)
from portfoliomind.sheets.schema import APPROVED_TRADES
from portfoliomind.signals.commissions import InstrumentType


# --- Fakes --------------------------------------------------------------


@dataclass
class FakeSheetsClient:
    """In-memory replacement for :class:`SheetsClient`.

    Mirrors the two methods persist_approved_trades calls:
    :meth:`read_range` and :meth:`append_rows`. Tests seed the
    ``approved_rows`` list with the existing sheet contents.
    """

    approved_rows: list[list[str]] = field(default_factory=list)
    append_calls: list[list[list[str]]] = field(default_factory=list)
    read_calls: list[tuple[str, str, str]] = field(default_factory=list)
    # Optional failure flags for error-path tests.
    raise_on_read: bool = False
    raise_on_append: bool = False

    def read_range(self, sheet_id: str, tab_name: str, range_a1: str) -> list[list[str]]:
        self.read_calls.append((sheet_id, tab_name, range_a1))
        if self.raise_on_read:
            raise RuntimeError("fake sheets read error")
        if tab_name != APPROVED_TRADES:
            return []
        return [list(r) for r in self.approved_rows]

    def append_rows(self, sheet_id: str, tab_name: str, values: list[list[Any]]) -> int:
        self.append_calls.append(values)
        if self.raise_on_append:
            raise RuntimeError("fake sheets append error")
        # Echo back the rows so the test can inspect them.
        for row in values:
            self.approved_rows.append([str(c) for c in row])
        return len(values) + len(self.approved_rows)  # fake row id


def _make_trade(
    *,
    ticker: str = "SPY",
    qty: float = 2.0,
    entry: float = 100.0,
    sl: float = 93.0,
    tp: float = 114.0,
    notional: float = 200.0,
    commission_rt: float = 0.0,
    r_r_ratio: float = 2.0,
    instrument: InstrumentType = InstrumentType.US_ETF,
    signal_date: str = "2026-06-10",
    combined: float = 0.65,
    confidence: float = 0.7,
) -> ApprovedTrade:
    return ApprovedTrade(
        ticker=ticker,
        qty=qty,
        entry=entry,
        sl=sl,
        tp=tp,
        notional=notional,
        commission_rt=commission_rt,
        r_r_ratio=r_r_ratio,
        instrument=instrument,
        signal_date=signal_date,
        asof_date=signal_date,
        combined=combined,
        confidence=confidence,
        reasons=[f"card-7 approval: combined={combined:+.2f} confidence={confidence:.2f}"],
        message_id="m1",
    )


# --- Dedup the spec lists as acceptance ------------------------------


class TestDedupAcceptance:
    """The exact dedup acceptance criterion: re-running with the same
    candidates produces zero new APPROVED_TRADES rows."""

    def test_first_run_appends(self):
        sheets = FakeSheetsClient()
        outcome = ApprovalOutcome(approved=[_make_trade()])
        res = persist_approved_trades(outcome, sheets=sheets, sheet_id="S1")
        assert res.error == ""
        assert res.rows_appended == 1
        assert res.duplicates_skipped == 0
        assert len(sheets.append_calls) == 1
        assert len(sheets.append_calls[0]) == 1

    def test_second_run_idempotent(self):
        sheets = FakeSheetsClient()
        outcome = ApprovalOutcome(approved=[_make_trade()])
        first = persist_approved_trades(outcome, sheets=sheets, sheet_id="S1")
        assert first.rows_appended == 1

        # The sheet now has the row we just wrote. Re-running with the
        # same candidate should produce zero new rows.
        second = persist_approved_trades(outcome, sheets=sheets, sheet_id="S1")
        assert second.error == ""
        assert second.rows_appended == 0
        assert second.duplicates_skipped == 1
        # No second append call to the sheet.
        assert len(sheets.append_calls) == 1

    def test_partial_overlap_dedups_only_dupes(self):
        sheets = FakeSheetsClient()
        # First run writes AAPL.
        aapl = _make_trade(ticker="AAPL", instrument=InstrumentType.US_STOCK)
        first = persist_approved_trades(
            ApprovalOutcome(approved=[aapl]), sheets=sheets, sheet_id="S1"
        )
        assert first.rows_appended == 1

        # Second run: AAPL is a dup, SPY is new.
        spy = _make_trade(ticker="SPY")
        second = persist_approved_trades(
            ApprovalOutcome(approved=[aapl, spy]),
            sheets=sheets,
            sheet_id="S1",
        )
        assert second.rows_appended == 1  # SPY only
        assert second.duplicates_skipped == 1  # AAPL
        # The single append call has exactly one row (SPY).
        assert len(sheets.append_calls[-1]) == 1
        assert "SPY" in sheets.append_calls[-1][0]


# --- Dry-run path ------------------------------------------------------


class TestDryRun:
    def test_dry_run_formats_without_writing(self):
        sheets = FakeSheetsClient()  # never used
        outcome = ApprovalOutcome(approved=[_make_trade()])
        res = persist_approved_trades(
            outcome, sheets=sheets, sheet_id="S1", dry_run=True
        )
        assert res.error == ""
        assert res.rows_appended == 0  # dry-run never reports a write
        assert len(res.written_rows) == 1
        assert sheets.append_calls == []  # no append happened

    def test_dry_run_no_sheets_client_required(self):
        outcome = ApprovalOutcome(approved=[_make_trade()])
        res = persist_approved_trades(
            outcome, sheets=None, sheet_id="", dry_run=True
        )
        assert res.error == ""
        assert len(res.written_rows) == 1

    def test_live_without_sheet_id_errors(self):
        outcome = ApprovalOutcome(approved=[_make_trade()])
        res = persist_approved_trades(outcome, sheets=FakeSheetsClient(), sheet_id="")
        assert "sheet_id" in res.error


# --- Empty / no-op ----------------------------------------------------


class TestEmpty:
    def test_no_approved_trades(self):
        sheets = FakeSheetsClient()
        outcome = ApprovalOutcome(approved=[])
        res = persist_approved_trades(outcome, sheets=sheets, sheet_id="S1")
        assert res.error == ""
        assert res.rows_appended == 0
        assert res.duplicates_skipped == 0
        assert sheets.append_calls == []

    def test_none_outcome(self):
        sheets = FakeSheetsClient()
        res = persist_approved_trades(None, sheets=sheets, sheet_id="S1")
        assert "outcome is None" in res.error


# --- Sheet error paths -----------------------------------------------


class TestSheetErrors:
    def test_read_error_caught(self):
        # The fake raises a non-"not found" error, which propagates
        # through the read step and is caught by the outer try/except
        # in :func:`_persist_unchecked`, which records the error on
        # the result without raising.
        sheets = FakeSheetsClient(raise_on_read=True)
        outcome = ApprovalOutcome(approved=[_make_trade()])
        res = persist_approved_trades(outcome, sheets=sheets, sheet_id="S1")
        assert res.error != ""
        assert "read" in res.error.lower() or "fake" in res.error.lower()

    def test_append_error_caught(self):
        sheets = FakeSheetsClient(raise_on_append=True)
        outcome = ApprovalOutcome(approved=[_make_trade()])
        res = persist_approved_trades(outcome, sheets=sheets, sheet_id="S1")
        assert "append" in res.error.lower()
        assert res.rows_appended == 0

    def test_empty_sheet_is_fine(self):
        # First-ever run: APPROVED_TRADES doesn't exist yet, read
        # returns []. Persist should still write the new row.
        sheets = FakeSheetsClient(approved_rows=[])
        outcome = ApprovalOutcome(approved=[_make_trade()])
        res = persist_approved_trades(outcome, sheets=sheets, sheet_id="S1")
        assert res.rows_appended == 1
        assert res.duplicates_skipped == 0


# --- Row format ------------------------------------------------------


class TestRowFormat:
    def test_row_layout_matches_schema(self):
        trade = _make_trade()
        row = _trade_to_row(trade)
        # 11 columns per the schema.
        assert len(row) == 11
        # Timestamp, Ticker, Type, Strategy, Timeframe, Allocation ($),
        # Qty, Entry Price, SL, TP, Approval Note
        assert row[0]  # Timestamp
        assert row[1] == "SPY"
        assert row[2] == "etf"
        assert row[3] == "card-7"
        assert row[4] == "swing"
        assert row[5] == "200.00"
        assert row[6] == "2"  # whole number, no decimals
        assert row[7] == "100.00"
        assert row[8] == "93.00"
        assert row[9] == "114.00"
        assert "combined" in row[10]
        assert "confidence" in row[10]

    def test_stock_type_label(self):
        trade = _make_trade(ticker="AAPL", instrument=InstrumentType.US_STOCK)
        row = _trade_to_row(trade)
        assert row[2] == "stock"

    def test_qty_formatting(self):
        # Fractional qty keeps 4 decimal places.
        trade = _make_trade(qty=1.5)
        row = _trade_to_row(trade)
        assert row[6] == "1.5"

    def test_qty_formatting_4_dp(self):
        # More decimals: rounded.
        trade = _make_trade(qty=1.3333)
        row = _trade_to_row(trade)
        assert row[6] == "1.3333"


# --- Dedup key ---------------------------------------------------------


class TestDedupKey:
    def test_dedup_key_basic(self):
        # The note column carries ``signal_date=YYYY-MM-DD`` so the
        # dedup key matches the trade-side key.
        row = [
            "2026-06-10T08:30:00-05:00",  # Timestamp (wall-clock, ignored for dedup)
            "SPY", "etf", "card-7", "swing",
            "200.00", "2", "100.00", "93.00", "114.00",
            "card-7 approval: combined=+0.65 confidence=0.70 signal_date=2026-06-10",
        ]
        key = _dedup_key_for_row(row)
        assert key == ("SPY", "2026-06-10", "100.00")

    def test_dedup_key_falls_back_to_timestamp(self):
        # Legacy row without ``signal_date=`` in the note: extract
        # the date from the Timestamp.
        row = ["2026-12-31T23:59:59-05:00", "AAPL", "stock", "card-7", "swing",
               "200.00", "1", "200.00", "186.00", "228.00", "legacy note"]
        key = _dedup_key_for_row(row)
        assert key == ("AAPL", "2026-12-31", "200.00")

    def test_dedup_key_ticker_uppercased(self):
        row = ["2026-06-10T08:30:00-05:00", "spy", "etf", "card-7", "swing",
               "200.00", "2", "100.00", "93.00", "114.00", "signal_date=2026-06-10"]
        key = _dedup_key_for_row(row)
        assert key[0] == "SPY"

    def test_dedup_key_short_row_safe(self):
        # Defensive: short row yields empty key.
        assert _dedup_key_for_row([]) == ("", "", "")
        assert _dedup_key_for_row(["only", "two"]) == ("", "", "")


# --- Different signal_date = different trade ------------------------


class TestCrossDayDedup:
    def test_same_ticker_different_day_appends(self):
        sheets = FakeSheetsClient()
        day1 = _make_trade(ticker="SPY", signal_date="2026-06-10")
        day2 = _make_trade(ticker="SPY", signal_date="2026-06-11")
        first = persist_approved_trades(
            ApprovalOutcome(approved=[day1]), sheets=sheets, sheet_id="S1"
        )
        second = persist_approved_trades(
            ApprovalOutcome(approved=[day2]), sheets=sheets, sheet_id="S1"
        )
        assert first.rows_appended == 1
        assert second.rows_appended == 1
        assert second.duplicates_skipped == 0

    def test_same_ticker_different_entry_appends(self):
        # The sizer might compute a slightly different entry on a
        # different day. Different entry = different trade.
        sheets = FakeSheetsClient()
        day1 = _make_trade(ticker="SPY", entry=100.0)
        day2 = _make_trade(ticker="SPY", entry=101.5)
        first = persist_approved_trades(
            ApprovalOutcome(approved=[day1]), sheets=sheets, sheet_id="S1"
        )
        second = persist_approved_trades(
            ApprovalOutcome(approved=[day2]), sheets=sheets, sheet_id="S1"
        )
        assert first.rows_appended == 1
        assert second.rows_appended == 1

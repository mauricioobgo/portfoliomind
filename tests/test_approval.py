"""Hermetic tests for :mod:`portfoliomind.approval`.

A FakeSheets stand-in replaces the SheetsClient; no Google API, no
config, no network.
"""

from __future__ import annotations

import pytest

from portfoliomind import approval
from portfoliomind.sheets.schema import AGENT_LOG, APPROVED_TRADES, SUGGESTIONS, TAB_HEADERS
from portfoliomind.signals.sizer import PositionSizer


class FakeSheets:
    """Just enough of the SheetsClient surface for the approval layer."""

    def __init__(self, suggestions: list[list] | None = None, approved: list[list] | None = None):
        self.data: dict[str, list[list]] = {
            SUGGESTIONS: suggestions or [],
            APPROVED_TRADES: approved or [],
            AGENT_LOG: [],
        }

    def read_range(self, sheet_id: str, tab_name: str, range_a1: str) -> list[list]:
        return [list(r) for r in self.data.get(tab_name, [])]

    def append_rows(self, sheet_id: str, tab_name: str, values: list[list]) -> int:
        self.data.setdefault(tab_name, []).extend(values)
        return len(self.data[tab_name])


def suggestion_row(
    ticker: str,
    action: str = "BUY",
    max_alloc: str = "",
    status: str = "ACTIVE",
) -> list[str]:
    # Timestamp, Ticker, Action, Max Allocation ($), Conviction, Source, Notes, Status
    return ["2026-06-11T08:00:00-05:00", ticker, action, max_alloc, "HIGH", "operator", "", status]


@pytest.fixture
def order():
    """A real sized order: 10 × $100 = $1000."""

    class _C:
        ticker = "AAPL"
        last_close = 100.0
        p_bullish = 0.70
        vol_20d = 0.01
        patterns = ["golden_cross"]

    return PositionSizer(equity=10_000.0).size(_C())


@pytest.fixture(autouse=True)
def _clean_seam():
    approval.reset_clients()
    yield
    approval.reset_clients()


# --- read_suggestions ---------------------------------------------------------


def test_read_suggestions_parses_rows():
    sheets = FakeSheets(suggestions=[suggestion_row("AAPL", max_alloc="$1,500.00")])
    out = approval.read_suggestions(sheets=sheets, sheet_id="sid")
    assert len(out) == 1
    s = out[0]
    assert s.ticker == "AAPL"
    assert s.max_allocation == 1500.0
    assert s.is_active_buy()


def test_read_suggestions_skips_blank_ticker_rows():
    sheets = FakeSheets(suggestions=[["ts", "", "BUY", "", "", "", "", "ACTIVE"]])
    assert approval.read_suggestions(sheets=sheets, sheet_id="sid") == []


def test_read_suggestions_unreachable_tab_returns_empty():
    class Broken(FakeSheets):
        def read_range(self, *a):
            raise RuntimeError("API down")

    assert approval.read_suggestions(sheets=Broken(), sheet_id="sid") == []


# --- post_candidates_and_collect_reactions --------------------------------------


def test_active_buy_suggestion_approves(order):
    sheets = FakeSheets(suggestions=[suggestion_row("AAPL")])
    outcome = approval.post_candidates_and_collect_reactions([order], sheets=sheets, sheet_id="sid")
    assert len(outcome.approved) == 1
    assert outcome.rejected == []
    # Every decision is audited.
    assert any("APPROVED" in row[3] for row in sheets.data[AGENT_LOG])


def test_no_suggestion_rejects(order):
    sheets = FakeSheets()
    outcome = approval.post_candidates_and_collect_reactions([order], sheets=sheets, sheet_id="sid")
    assert outcome.approved == []
    assert len(outcome.rejected) == 1
    assert any("no row" in n for n in outcome.notes)


def test_inactive_status_rejects(order):
    sheets = FakeSheets(suggestions=[suggestion_row("AAPL", status="CLOSED")])
    outcome = approval.post_candidates_and_collect_reactions([order], sheets=sheets, sheet_id="sid")
    assert outcome.approved == []
    assert any("not an active BUY" in n for n in outcome.notes)


def test_sell_action_rejects(order):
    """Long-only: a SELL suggestion never authorizes a trade."""
    sheets = FakeSheets(suggestions=[suggestion_row("AAPL", action="SELL")])
    outcome = approval.post_candidates_and_collect_reactions([order], sheets=sheets, sheet_id="sid")
    assert outcome.approved == []


def test_allocation_cap_scales_qty_down(order):
    sheets = FakeSheets(suggestions=[suggestion_row("AAPL", max_alloc="550")])
    outcome = approval.post_candidates_and_collect_reactions([order], sheets=sheets, sheet_id="sid")
    assert len(outcome.approved) == 1
    scaled = outcome.approved[0]
    assert scaled.qty == 5.0
    assert scaled.allocation == pytest.approx(500.0)


def test_cap_below_one_share_rejects(order):
    sheets = FakeSheets(suggestions=[suggestion_row("AAPL", max_alloc="50")])
    outcome = approval.post_candidates_and_collect_reactions([order], sheets=sheets, sheet_id="sid")
    assert outcome.approved == []
    assert any("exceeds the" in n for n in outcome.notes)


def test_case_insensitive_matching(order):
    sheets = FakeSheets(suggestions=[suggestion_row("aapl", action="buy", status="active")])
    outcome = approval.post_candidates_and_collect_reactions([order], sheets=sheets, sheet_id="sid")
    assert len(outcome.approved) == 1


def test_timeout_param_accepted_for_contract_compat(order):
    sheets = FakeSheets(suggestions=[suggestion_row("AAPL")])
    outcome = approval.post_candidates_and_collect_reactions(
        [order], timeout_seconds=1, sheets=sheets, sheet_id="sid"
    )
    assert len(outcome.approved) == 1


# --- persist_approved_trades ----------------------------------------------------------


def test_persist_appends_rows_matching_headers(order):
    sheets = FakeSheets()
    n = approval.persist_approved_trades([order], sheets=sheets, sheet_id="sid")
    assert n == 1
    rows = sheets.data[APPROVED_TRADES]
    assert len(rows) == 1
    assert len(rows[0]) == len(TAB_HEADERS[APPROVED_TRADES])
    assert rows[0][1] == "AAPL"


def test_persist_dedups_on_ticker_timestamp(order):
    sheets = FakeSheets()
    assert approval.persist_approved_trades([order], sheets=sheets, sheet_id="sid") == 1
    # Same order again — same (Ticker, Timestamp) key → nothing appended.
    assert approval.persist_approved_trades([order], sheets=sheets, sheet_id="sid") == 0
    assert len(sheets.data[APPROVED_TRADES]) == 1


def test_persist_empty_is_zero():
    assert approval.persist_approved_trades([], sheets=FakeSheets(), sheet_id="sid") == 0


# --- Client seam ------------------------------------------------------------------------


def test_set_clients_seam(order):
    sheets = FakeSheets(suggestions=[suggestion_row("AAPL")])
    approval.set_clients(sheets=sheets, sheet_id="sid")
    outcome = approval.post_candidates_and_collect_reactions([order])
    assert len(outcome.approved) == 1
    approval.reset_clients()

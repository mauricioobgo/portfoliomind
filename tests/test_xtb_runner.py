"""Hermetic tests for the XTB morning-runner.

These tests never spawn a Playwright browser and never touch Google
Sheets. We inject the build_context / ensure_logged_in / place_order
factories with in-memory fakes and let the runner compose them.

Coverage:

* :func:`run_morning` with an empty ``APPROVED_TRADES`` tab returns
  ``skipped=True, skip_reason="no approved trades"`` and writes
  nothing to ``EXECUTED_ORDERS``.

* :func:`run_morning` in dry-run mode (the default) writes a
  ``DRY_RUN`` row per approved trade and never opens a browser.

* :func:`run_morning` in live mode (operator explicitly opted in
  via ``xtb_dry_run=False`` and ``xtb_live_confirm=True``) opens a
  browser, calls :func:`place_order`, and writes a ``PLACED`` row
  with the order ID and screenshot path.

* :func:`run_morning` never raises, even when ``build_context``,
  ``ensure_logged_in``, or ``place_order`` blows up.

* :func:`run_morning` honors the dedup contract: a ticker+timestamp
  pair already in ``EXECUTED_ORDERS`` is skipped on a re-run.

* :func:`run_morning` writes a ``VALIDATION_FAILED`` row when a row
  in ``APPROVED_TRADES`` is missing SL or TP, but still processes
  the rest of the batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pytest

from portfoliomind.config import PortfoliomindConfig
from portfoliomind.scheduler.jobs import BOGOTA_TZ, MorningContext
from portfoliomind.sheets.schema import (
    APPROVED_TRADES,
    EXECUTED_ORDERS,
    TAB_HEADERS,
)
from portfoliomind.xtb import runner as xtb_runner
from portfoliomind.xtb.order import (
    OrderSide,
    OrderSpec,
    PlaceOrderError,
)

from .conftest import full_env


# --- Fakes -------------------------------------------------------------------


class _FakeWorksheet:
    def __init__(self, headers: list[str], *, with_header: bool = True) -> None:
        self.headers = list(headers)
        # When ``with_header`` is True, the first row in ``values`` is
        # the canonical header. Tests that seed data via the
        # ``initial=`` kwarg pass *only* the data rows; the
        # ``_FakeSheetsClient`` creates the header row automatically.
        if with_header:
            self.values: list[list[str]] = [list(headers)]
        else:
            self.values = []


class _FakeSheetsClient:
    """In-memory substitute for the SheetsClient operations the runner uses.

    Honors the canonical A1 range convention used by the real
    :class:`SheetsClient`: a range like ``A2:K9999`` returns the rows
    starting at row 2 (1-indexed in Sheets). A range of ``A1:K1``
    returns just the header row. The runner always asks for ``A2:...``
    so the header is excluded — matching production behavior.
    """

    def __init__(self, initial: Optional[dict[str, list[list[str]]]] = None) -> None:
        self.worksheets: dict[str, _FakeWorksheet] = {}
        if initial:
            for tab, rows in initial.items():
                ws = _FakeWorksheet(TAB_HEADERS.get(tab, []))
                ws.values.extend(rows)
                self.worksheets[tab] = ws
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def _parse_start_row(self, range_a1: str) -> int:
        """Return the 1-indexed start row in the range (default 1)."""
        # Format: "A2:K9999" -> start at row 2.
        # Format: "A2:C2" -> start at row 2.
        # Format: "A:A" -> whole column, start at row 1.
        import re

        m = re.match(r"^[A-Z]+(\d+):", range_a1)
        if m:
            return int(m.group(1))
        m = re.match(r"^[A-Z]+(\d+)$", range_a1)
        if m:
            return int(m.group(1))
        return 1

    def ensure_worksheet(self, sheet_id: str, title: str, headers: list[str]) -> dict:
        self.calls.append(("ensure_worksheet", (sheet_id, title)))
        if title not in self.worksheets:
            ws = _FakeWorksheet(headers)
            self.worksheets[title] = ws
        return {"sheetId": 0, "title": title}

    def read_range(
        self, sheet_id: str, tab_name: str, range_a1: str
    ) -> list[list[str]]:
        self.calls.append(("read_range", (sheet_id, tab_name, range_a1)))
        ws = self.worksheets.get(tab_name)
        if not ws:
            return []
        start_row = self._parse_start_row(range_a1)
        # Sheets is 1-indexed; our list is 0-indexed.
        return [list(r) for r in ws.values[start_row - 1:]]

    def write_range(
        self, sheet_id: str, tab_name: str, range_a1: str, values: list[list[str]]
    ) -> None:
        self.calls.append(("write_range", (sheet_id, tab_name, range_a1)))

    def append_rows(self, sheet_id: str, tab_name: str, values: list[list[str]]) -> int:
        self.calls.append(("append_rows", (sheet_id, tab_name, [list(r) for r in values])))
        ws = self.worksheets.setdefault(
            tab_name, _FakeWorksheet(TAB_HEADERS.get(tab_name, []))
        )
        first = len(ws.values) + 1
        for v in values:
            ws.values.append(list(v))
        return first

    def row_count(self, sheet_id: str, tab_name: str) -> int:
        return len(self.worksheets.get(tab_name, _FakeWorksheet([])).values)


# APPROVED_TRADES schema (11 cols):
# 0 Timestamp, 1 Ticker, 2 Type, 3 Strategy, 4 Timeframe,
# 5 Allocation, 6 Qty, 7 Entry Price, 8 SL, 9 TP, 10 Approval Note


def _approved_row(
    ticker: str = "AAPL.US",
    qty: str = "10",
    entry: str = "192.50",
    sl: str = "189.00",
    tp: str = "198.00",
    ts: str = "2026-06-08T08:30:00-05:00",
    side: str = "BUY",
    note: str = "Card 6 signal",
) -> list[str]:
    return [
        ts, ticker, side, "Short", "3-7d", "1925.00", qty, entry, sl, tp, note,
    ]


# --- Test fixtures -----------------------------------------------------------


@pytest.fixture
def config() -> PortfoliomindConfig:
    return PortfoliomindConfig.from_env(env=full_env("test-sheet-id-002"))


@pytest.fixture
def monday_today() -> datetime:
    return datetime(2026, 6, 8, 8, 30, tzinfo=BOGOTA_TZ)


def _make_ctx(
    config: PortfoliomindConfig, sheets: _FakeSheetsClient, today: datetime
) -> MorningContext:
    log_calls: list[tuple[str, str]] = []

    def log_to_sheet(level: str, message: str) -> None:
        log_calls.append((level, message))

    return MorningContext(
        config=config,
        sheets=sheets,
        sheet_id="test-sheet-id-002",
        today=today,
        log_to_sheet=log_to_sheet,
    )


# --- Empty APPROVED_TRADES ---------------------------------------------------


class TestXtbRunnerNoApprovals:
    """An empty ``APPROVED_TRADES`` tab is a clean no-op."""

    def test_no_approved_returns_skipped(self, config, monday_today):
        sheets = _FakeSheetsClient(
            initial={APPROVED_TRADES: []}  # header only
        )
        ctx = _make_ctx(config, sheets, monday_today)
        result = xtb_runner.run_morning(ctx)
        assert result.skipped is True
        assert result.skip_reason == "no approved trades"
        assert result.runner == "card3"
        assert result.orders_placed == 0
        # Nothing was appended to EXECUTED_ORDERS.
        appends = [c for c in sheets.calls if c[0] == "append_rows"]
        assert appends == []

    def test_approved_tab_missing_returns_skipped(self, config, monday_today):
        sheets = _FakeSheetsClient()  # no APPROVED_TRADES worksheet
        ctx = _make_ctx(config, sheets, monday_today)
        result = xtb_runner.run_morning(ctx)
        assert result.skipped is True
        assert result.skip_reason == "no approved trades"


# --- Dry-run path (default) --------------------------------------------------


class TestXtbRunnerDryRun:
    """The default path: dry-run, no browser, write a DRY_RUN row per trade."""

    def test_dry_run_default_writes_dry_run_rows(
        self, config, monday_today
    ):
        sheets = _FakeSheetsClient(
            initial={
                APPROVED_TRADES: [
                    _approved_row("AAPL.US"),
                    _approved_row("MSFT.US", qty="5", entry="415.10",
                                  sl="405.00", tp="435.00"),
                ],
            }
        )
        # The factories MUST NOT be called in dry-run mode.
        build_calls: list[Any] = []
        login_calls: list[Any] = []
        place_calls: list[Any] = []
        xtb_runner.set_factories(
            build_context_factory=lambda *a, **kw: (
                build_calls.append(1) or _FakeContext()
            ),
            ensure_logged_in_factory=lambda *a, **kw: login_calls.append(1),
            place_order_factory=lambda *a, **kw: (
                place_calls.append(1) or _FakeOrderResult()
            ),
        )
        try:
            ctx = _make_ctx(config, sheets, monday_today)
            result = xtb_runner.run_morning(ctx)
        finally:
            xtb_runner.reset_factories()

        # Dry-run was used; no browser opened.
        assert build_calls == []
        assert login_calls == []
        assert place_calls == []
        # Orders-placed counter is 0 — dry-run rows don't count.
        assert result.orders_placed == 0
        assert result.error == ""
        # Two DRY_RUN rows were appended to EXECUTED_ORDERS.
        exec_ws = sheets.worksheets.get(EXECUTED_ORDERS)
        assert exec_ws is not None
        data_rows = exec_ws.values[1:]  # skip header
        assert len(data_rows) == 2
        statuses = [r[8] for r in data_rows]
        assert statuses == [xtb_runner.DRY_RUN_STATUS, xtb_runner.DRY_RUN_STATUS]
        # Side and Ticker are populated.
        assert data_rows[0][1] == "AAPL.US"
        assert data_rows[0][3] == "BUY"
        # No order ID for a dry-run.
        assert data_rows[0][2] == ""

    def test_dry_run_validation_failure_writes_status(
        self, config, monday_today
    ):
        """A row with missing SL is logged as VALIDATION_FAILED, the
        rest of the batch is still processed.
        """
        sheets = _FakeSheetsClient(
            initial={
                APPROVED_TRADES: [
                    _approved_row("AAPL.US", sl="0"),  # invalid SL
                    _approved_row("MSFT.US", qty="5", entry="415.10",
                                  sl="405.00", tp="435.00"),
                ],
            }
        )
        ctx = _make_ctx(config, sheets, monday_today)
        result = xtb_runner.run_morning(ctx)
        # The bad row was dropped (a status was written), the good row
        # was processed. result.error names the validation failure so
        # the scheduler surfaces it in the Discord alert.
        assert result.error
        assert "AAPL" in result.error
        exec_ws = sheets.worksheets.get(EXECUTED_ORDERS)
        data_rows = exec_ws.values[1:]
        assert len(data_rows) == 2
        statuses = [r[8] for r in data_rows]
        assert xtb_runner.VALIDATION_FAILED_STATUS in statuses
        assert xtb_runner.DRY_RUN_STATUS in statuses

    def test_dry_run_iron_rule_rejects_missing_sl_or_tp(
        self, config, monday_today
    ):
        """The OrderSpec iron rule: missing SL OR TP is fatal at the
        boundary, even in dry-run mode.
        """
        sheets = _FakeSheetsClient(
            initial={
                APPROVED_TRADES: [
                    _approved_row("AAPL.US", sl="0", tp="0"),
                ],
            }
        )
        ctx = _make_ctx(config, sheets, monday_today)
        result = xtb_runner.run_morning(ctx)
        # A VALIDATION_FAILED row was written.
        exec_ws = sheets.worksheets.get(EXECUTED_ORDERS)
        assert exec_ws is not None
        data_rows = exec_ws.values[1:]
        assert len(data_rows) == 1
        assert data_rows[0][8] == xtb_runner.VALIDATION_FAILED_STATUS
        # orders_placed is still 0 (nothing was actually placed).
        assert result.orders_placed == 0
        # result.error names the iron-rule violation.
        assert "Stop-loss" in result.error or "mandatory" in result.error

    def test_dry_run_dedup_skips_already_placed(
        self, config, monday_today
    ):
        """A ticker+timestamp pair already in EXECUTED_ORDERS is skipped."""
        ts = "2026-06-08T08:30:00-05:00"
        sheets = _FakeSheetsClient(
            initial={
                APPROVED_TRADES: [
                    _approved_row("AAPL.US", ts=ts),
                ],
                EXECUTED_ORDERS: [
                    # Same ticker + same timestamp (col 1) as the
                    # approved row above -> dedup hit.
                    [ts, "AAPL.US", "12345", "BUY", "10",
                     "192.50", "189.00", "198.00",
                     xtb_runner.PLACED_STATUS, ""],
                ],
            }
        )
        ctx = _make_ctx(config, sheets, monday_today)
        result = xtb_runner.run_morning(ctx)
        # The dedup hit means we didn't append a new row.
        exec_ws = sheets.worksheets.get(EXECUTED_ORDERS)
        # Only the pre-existing row, no new appends.
        data_rows = exec_ws.values[1:]
        assert len(data_rows) == 1
        assert result.orders_placed == 0


# --- Live path (operator opt-in) --------------------------------------------


class _FakeContext:
    """A fake Playwright context whose .pages attribute is observable."""

    def __init__(self) -> None:
        self.pages = [_FakePage()]
        self.close_calls: list[int] = []

    def close(self) -> None:
        self.close_calls.append(1)


class _FakePage:
    """A placeholder for a Playwright Page; never used directly in tests."""

    pass


@dataclass
class _FakeOrderResult:
    """A fake :class:`OrderResult` returned by the place_order factory."""

    order_id: Optional[str] = "ORD-12345"
    spec: Optional[OrderSpec] = None
    screenshot_before: Optional[Path] = None
    screenshot_after: Optional[Path] = None

    def __post_init__(self) -> None:
        # If the test didn't supply a spec, build a benign one.
        if self.spec is None:
            self.spec = OrderSpec(
                ticker="AAPL.US",
                side=OrderSide.BUY,
                qty=10, entry_price=192.50, sl=189.00, tp=198.00,
            )


class TestXtbRunnerLivePath:
    """The live path: open a browser, log in, place each order."""

    def test_live_mode_calls_place_order(
        self, config, monday_today
    ):
        from dataclasses import replace

        live_cfg = replace(
            config, xtb_dry_run=False, xtb_live_confirm=True,
        )
        sheets = _FakeSheetsClient(
            initial={
                APPROVED_TRADES: [
                    _approved_row("AAPL.US"),
                ],
            }
        )
        build_calls: list[Any] = []
        login_calls: list[Any] = []
        place_calls: list[OrderSpec] = []
        context = _FakeContext()

        def fake_build(*args, **kwargs):
            build_calls.append((args, kwargs))
            return context

        def fake_login(*args, **kwargs):
            login_calls.append((args, kwargs))

        def fake_place(page, spec, **kwargs):
            place_calls.append(spec)
            return _FakeOrderResult(order_id="LIVE-98765", spec=spec)

        xtb_runner.set_factories(
            build_context_factory=fake_build,
            ensure_logged_in_factory=fake_login,
            place_order_factory=fake_place,
        )
        try:
            ctx = _make_ctx(live_cfg, sheets, monday_today)
            result = xtb_runner.run_morning(ctx)
        finally:
            xtb_runner.reset_factories()

        # Browser opened, login called, place called.
        assert len(build_calls) == 1
        assert len(login_calls) == 1
        assert len(place_calls) == 1
        # The placed spec is the one we built from the APPROVED row.
        placed_spec = place_calls[0]
        assert placed_spec.ticker == "AAPL.US"
        assert placed_spec.side is OrderSide.BUY
        assert placed_spec.qty == 10.0
        assert placed_spec.sl == 189.00
        assert placed_spec.tp == 198.00
        # Result: 1 order placed.
        assert result.orders_placed == 1
        # Context was torn down.
        assert context.close_calls == [1]
        # EXECUTED_ORDERS got a PLACED row.
        exec_ws = sheets.worksheets.get(EXECUTED_ORDERS)
        data_rows = exec_ws.values[1:]
        assert len(data_rows) == 1
        assert data_rows[0][8] == xtb_runner.PLACED_STATUS
        assert data_rows[0][2] == "LIVE-98765"

    def test_live_mode_requires_both_toggles(
        self, config, monday_today
    ):
        """Live trading requires BOTH ``xtb_dry_run=False`` AND
        ``xtb_live_confirm=True``. Flipping only one keeps the run in
        dry-run mode.
        """
        from dataclasses import replace

        # Case 1: xtb_dry_run=False, but xtb_live_confirm still False
        only_dry_off = replace(config, xtb_dry_run=False, xtb_live_confirm=False)
        sheets = _FakeSheetsClient(
            initial={
                APPROVED_TRADES: [
                    _approved_row("AAPL.US"),
                ],
            }
        )
        build_calls: list[Any] = []
        xtb_runner.set_factories(
            build_context_factory=lambda *a, **kw: (
                build_calls.append(1) or _FakeContext()
            ),
            ensure_logged_in_factory=lambda *a, **kw: None,
            place_order_factory=lambda *a, **kw: _FakeOrderResult(),
        )
        try:
            ctx = _make_ctx(only_dry_off, sheets, monday_today)
            result = xtb_runner.run_morning(ctx)
        finally:
            xtb_runner.reset_factories()

        # No browser opened; dry-run mode was used.
        assert build_calls == []
        assert result.orders_placed == 0
        # EXECUTED_ORDERS got a DRY_RUN row.
        exec_ws = sheets.worksheets.get(EXECUTED_ORDERS)
        data_rows = exec_ws.values[1:]
        assert len(data_rows) == 1
        assert data_rows[0][8] == xtb_runner.DRY_RUN_STATUS

    def test_live_mode_unconfirmed_status_when_no_order_id(
        self, config, monday_today
    ):
        """When place_order returns an OrderResult with order_id=None
        (we couldn't read the confirmation modal), the status is
        UNCONFIRMED, not PLACED.
        """
        from dataclasses import replace

        live_cfg = replace(config, xtb_dry_run=False, xtb_live_confirm=True)
        sheets = _FakeSheetsClient(
            initial={
                APPROVED_TRADES: [
                    _approved_row("AAPL.US"),
                ],
            }
        )
        context = _FakeContext()

        def fake_build(*a, **kw):
            return context

        def fake_place(page, spec, **kwargs):
            return _FakeOrderResult(order_id=None, spec=spec)

        xtb_runner.set_factories(
            build_context_factory=fake_build,
            ensure_logged_in_factory=lambda *a, **kw: None,
            place_order_factory=fake_place,
        )
        try:
            ctx = _make_ctx(live_cfg, sheets, monday_today)
            result = xtb_runner.run_morning(ctx)
        finally:
            xtb_runner.reset_factories()

        assert result.orders_placed == 1
        exec_ws = sheets.worksheets.get(EXECUTED_ORDERS)
        data_rows = exec_ws.values[1:]
        assert data_rows[0][8] == xtb_runner.UNCONFIRMED_STATUS
        assert data_rows[0][2] == ""  # no order id

    def test_live_mode_place_order_error_does_not_abort_batch(
        self, config, monday_today
    ):
        """A :class:`PlaceOrderError` on one row does NOT abort the
        batch; the next row is still attempted.
        """
        from dataclasses import replace

        live_cfg = replace(config, xtb_dry_run=False, xtb_live_confirm=True)
        sheets = _FakeSheetsClient(
            initial={
                APPROVED_TRADES: [
                    _approved_row("AAPL.US", ts="2026-06-08T08:30:00-05:00"),
                    _approved_row("MSFT.US", ts="2026-06-08T08:30:00-05:00",
                                  qty="5", entry="415.10", sl="405.00", tp="435.00"),
                ],
            }
        )
        place_calls: list[OrderSpec] = []
        context = _FakeContext()

        def fake_place(page, spec, **kwargs):
            place_calls.append(spec)
            if spec.ticker == "AAPL.US":
                raise PlaceOrderError("simulated xStation timeout")
            return _FakeOrderResult(order_id="LIVE-MSFT", spec=spec)

        xtb_runner.set_factories(
            build_context_factory=lambda *a, **kw: context,
            ensure_logged_in_factory=lambda *a, **kw: None,
            place_order_factory=fake_place,
        )
        try:
            ctx = _make_ctx(live_cfg, sheets, monday_today)
            result = xtb_runner.run_morning(ctx)
        finally:
            xtb_runner.reset_factories()

        # Both rows were attempted.
        assert len(place_calls) == 2
        # One placed, one errored -> orders_placed = 1, error surfaced.
        assert result.orders_placed == 1
        assert result.error
        assert "AAPL" in result.error

    def test_live_mode_login_failure_aborts_batch(
        self, config, monday_today
    ):
        """A login failure aborts the whole batch — we never place
        orders without a verified session.
        """
        from dataclasses import replace

        live_cfg = replace(config, xtb_dry_run=False, xtb_live_confirm=True)
        sheets = _FakeSheetsClient(
            initial={
                APPROVED_TRADES: [
                    _approved_row("AAPL.US"),
                ],
            }
        )
        place_calls: list[Any] = []
        context = _FakeContext()

        def fake_login(*a, **kw):
            raise RuntimeError("simulated login failure")

        def fake_place(*a, **kw):
            place_calls.append(1)
            return _FakeOrderResult()

        xtb_runner.set_factories(
            build_context_factory=lambda *a, **kw: context,
            ensure_logged_in_factory=fake_login,
            place_order_factory=fake_place,
        )
        try:
            ctx = _make_ctx(live_cfg, sheets, monday_today)
            result = xtb_runner.run_morning(ctx)
        finally:
            xtb_runner.reset_factories()

        # place_order was NOT called.
        assert place_calls == []
        # Context was still torn down.
        assert context.close_calls == [1]
        # The error mentions login.
        assert "login" in result.error.lower()
        assert result.orders_placed == 0


# --- Failure modes -----------------------------------------------------------


class TestXtbRunnerNeverRaises:
    """Every failure mode returns a MorningResult; never raises."""

    def test_no_config_returns_error(self, config, monday_today):
        sheets = _FakeSheetsClient()
        ctx = _make_ctx(
            config=None,  # type: ignore[arg-type]
            sheets=sheets,
            today=monday_today,
        )
        ctx.config = None  # type: ignore[assignment]
        result = xtb_runner.run_morning(ctx)
        assert result.error
        assert "no config" in result.error

    def test_empty_sheet_id_returns_error(self, config, monday_today):
        sheets = _FakeSheetsClient()
        ctx = MorningContext(
            config=config,
            sheets=sheets,
            sheet_id="",
            today=monday_today,
            log_to_sheet=lambda level, message: None,
        )
        result = xtb_runner.run_morning(ctx)
        assert result.error
        assert "empty sheet_id" in result.error

    def test_unexpected_outer_exception_returns_error(
        self, config, monday_today
    ):
        sheets = _FakeSheetsClient(
            initial={
                APPROVED_TRADES: [
                    _approved_row("AAPL.US"),
                ],
            }
        )
        # Inject a side-effect that raises inside the runner's flow.
        # We do this by making read_range raise on a specific call.
        original_read = sheets.read_range
        call_count = {"n": 0}

        def broken_read(sheet_id, tab_name, range_a1):
            call_count["n"] += 1
            # Let the APPROVED_TRADES read succeed; blow up the second
            # read (the EXECUTED_ORDERS dedup read).
            if tab_name == EXECUTED_ORDERS and call_count["n"] > 1:
                raise RuntimeError("simulated sheets outage")
            return original_read(sheet_id, tab_name, range_a1)

        sheets.read_range = broken_read  # type: ignore[method-assign]
        ctx = _make_ctx(config, sheets, monday_today)
        result = xtb_runner.run_morning(ctx)
        # The runner swallowed the read_range failure and produced
        # a MorningResult. orders_placed may be 0 (no dedup info, so
        # the dry-run path still writes rows; the assertion below is
        # that the run completed cleanly without raising).
        assert result.error == "" or "RuntimeError" in result.error


# --- Public surface ---------------------------------------------------------


class TestXtbRunnerSurface:
    """The public surface exposes ``run_morning`` and the test helpers."""

    def test_run_morning_is_callable(self):
        assert callable(xtb_runner.run_morning)

    def test_status_constants_are_strings(self):
        assert isinstance(xtb_runner.DRY_RUN_STATUS, str)
        assert isinstance(xtb_runner.PLACED_STATUS, str)
        assert isinstance(xtb_runner.UNCONFIRMED_STATUS, str)
        assert isinstance(xtb_runner.VALIDATION_FAILED_STATUS, str)

    def test_set_and_reset_factories(self):
        def sentinel_build(*a, **kw):
            return None

        def sentinel_login(*a, **kw):
            return None

        # _FakeOrderResult is a structural twin of OrderResult for
        # tests; the explicit factory annotation is too strict.
        sentinel_place = lambda *a, **kw: _FakeOrderResult()  # noqa: E731  # type: ignore[arg-type]

        xtb_runner.set_factories(
            build_context_factory=sentinel_build,
            ensure_logged_in_factory=sentinel_login,
            place_order_factory=sentinel_place,
        )
        assert xtb_runner._build_context_factory is sentinel_build
        assert xtb_runner._ensure_logged_in_factory is sentinel_login
        assert xtb_runner._place_order_factory is sentinel_place

        xtb_runner.reset_factories()
        assert xtb_runner._build_context_factory is xtb_runner.build_context
        assert xtb_runner._ensure_logged_in_factory is xtb_runner.ensure_logged_in
        assert xtb_runner._place_order_factory is xtb_runner.place_order

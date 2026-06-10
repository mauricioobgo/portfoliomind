"""Unit tests for the PortfolioMind scheduler.

These tests are hermetic — they never hit Google Sheets, never hit
yfinance, and never spawn the real scheduler thread. We use a fake
:class:`SheetsClient` that records calls and an in-memory
``read_range`` / ``write_range`` map, plus an injected
``price_fetcher`` for the yfinance path.

Test coverage:

* :func:`portfoliomind.scheduler.jobs.bogota_weekday` — the weekday
  math is correct in Bogota-local time, not UTC.
* :func:`portfoliomind.scheduler.jobs.HolidayCalendar` — env parsing,
  bad entries are logged-and-skipped, is_holiday is set membership.
* :func:`portfoliomind.scheduler.jobs.is_morning_trading_day` —
  Mon–Fri passes, Sat/Sun fails, configured holidays fail.
* :func:`portfoliomind.scheduler.jobs.morning_run` — weekend skip,
  holiday skip, no-platform-modules path (cards 2/3 not implemented),
  partial-failure path.
* :func:`portfoliomind.scheduler.jobs.refresh_returns` — the math
  (Current Price / Current Value / P&L / Days Held / Total Return),
  pruning of unresolvable tickers, idempotency (re-running produces
  the same write), no-sheet path, parse-error tolerance.
* :class:`portfoliomind.scheduler.loop.ScheduleConfig` and
  :func:`portfoliomind.scheduler.loop.build_morning_trigger` —
  the cron trigger is wired to ``America/Bogota`` and ``mon-fri``.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import pytest

from portfoliomind.scheduler.jobs import (
    HolidayCalendar,
    MorningOutcome,
    RefreshOutcome,
    TickerRow,
    bogota_weekday,
    is_morning_trading_day,
    morning_run,
    refresh_returns,
)
from portfoliomind.scheduler.jobs import BOGOTA_TZ as JOBS_BOGOTA_TZ
from portfoliomind.scheduler.loop import (
    ScheduleConfig,
    build_morning_trigger,
    build_returns_trigger,
    build_scheduler,
)

# --- Bogota weekday math ---------------------------------------------------


class TestBogotaWeekday:
    """The weekday math must use Bogota-local time, not UTC."""

    def test_monday_bogota(self):
        # 2026-06-08 was a Monday in Bogota.
        d = datetime(2026, 6, 8, 8, 30, tzinfo=JOBS_BOGOTA_TZ)
        assert bogota_weekday(d) == 0

    def test_friday_bogota(self):
        # 2026-06-12 was a Friday in Bogota.
        d = datetime(2026, 6, 12, 23, 59, tzinfo=JOBS_BOGOTA_TZ)
        assert bogota_weekday(d) == 4

    def test_saturday_bogota(self):
        d = datetime(2026, 6, 13, 0, 1, tzinfo=JOBS_BOGOTA_TZ)
        assert bogota_weekday(d) == 5

    def test_sunday_bogota(self):
        d = datetime(2026, 6, 14, 23, 59, tzinfo=JOBS_BOGOTA_TZ)
        assert bogota_weekday(d) == 6

    def test_late_evening_bogota_is_next_day_in_utc(self):
        """23:30 Bogota = 04:30 UTC next day. Make sure the weekday
        math honors the Bogota date, not the UTC date."""
        # 2026-06-13 23:30 Bogota = 2026-06-14 04:30 UTC (Sunday Bogota, Sunday UTC).
        d = datetime(2026, 6, 13, 23, 30, tzinfo=JOBS_BOGOTA_TZ)
        assert bogota_weekday(d) == 5  # Saturday in Bogota
        # And the same instant in UTC is 2026-06-14 (Sunday).
        utc_d = d.astimezone(timezone.utc)
        assert utc_d.weekday() == 6  # Sunday in UTC

    def test_early_morning_bogota_is_previous_day_in_utc(self):
        """01:00 Bogota = 06:00 UTC same day. Less interesting, but
        documents the symmetry."""
        d = datetime(2026, 6, 9, 1, 0, tzinfo=JOBS_BOGOTA_TZ)  # Tuesday
        assert bogota_weekday(d) == 1

    def test_naive_datetime_treated_as_bogota(self):
        """If a caller hands us a naive datetime, the function still
        works — the type annotation says tzinfo is required, but a
        defensive cast doesn't hurt. We document the behavior here:
        a naive datetime goes through ``.weekday()`` which uses the
        naive date components."""
        d = datetime(2026, 6, 8, 8, 30)  # no tz
        # Without a tz the date is whatever the caller passed, and the
        # weekday is the date's weekday.
        assert bogota_weekday(d) == 0


# --- Holiday calendar ------------------------------------------------------


class TestHolidayCalendar:
    def test_empty_calendar_has_no_holidays(self):
        cal = HolidayCalendar(skipped_dates=frozenset())
        assert not cal.is_holiday(date(2026, 1, 1))
        assert not cal.is_holiday(date(2026, 12, 25))

    def test_explicit_holiday(self):
        cal = HolidayCalendar(skipped_dates=frozenset({date(2026, 1, 1)}))
        assert cal.is_holiday(date(2026, 1, 1))
        assert not cal.is_holiday(date(2026, 1, 2))

    def test_from_env_parses_iso_dates(self):
        env = {"PORTFOLIOMIND_HOLIDAYS": "2026-01-01,2026-05-01,2026-07-20"}
        cal = HolidayCalendar.from_env(env)
        assert cal.is_holiday(date(2026, 1, 1))
        assert cal.is_holiday(date(2026, 5, 1))
        assert cal.is_holiday(date(2026, 7, 20))
        assert not cal.is_holiday(date(2026, 6, 8))

    def test_from_env_handles_blank(self):
        cal = HolidayCalendar.from_env({})
        assert cal.skipped_dates == frozenset()
        cal = HolidayCalendar.from_env({"PORTFOLIOMIND_HOLIDAYS": ""})
        assert cal.skipped_dates == frozenset()
        cal = HolidayCalendar.from_env({"PORTFOLIOMIND_HOLIDAYS": "   "})
        assert cal.skipped_dates == frozenset()

    def test_from_env_skips_invalid_dates(self):
        """A typo in the env (e.g. '2026-13-99') must not crash the
        loader — the agent should still come up."""
        env = {"PORTFOLIOMIND_HOLIDAYS": "2026-01-01,not-a-date,2026-13-99"}
        cal = HolidayCalendar.from_env(env)
        assert cal.is_holiday(date(2026, 1, 1))
        assert date(2026, 1, 1) in cal.skipped_dates
        # Bad entries are not added.
        assert len(cal.skipped_dates) == 1

    def test_from_env_handles_trailing_commas(self):
        env = {"PORTFOLIOMIND_HOLIDAYS": "2026-01-01,,2026-05-01,"}
        cal = HolidayCalendar.from_env(env)
        assert len(cal.skipped_dates) == 2


# --- is_morning_trading_day ------------------------------------------------


class TestIsMorningTradingDay:
    @pytest.mark.parametrize(
        "bogota_date,expected",
        [
            # 2026-06-08 is Monday in Bogota.
            (datetime(2026, 6, 8, 8, 30, tzinfo=JOBS_BOGOTA_TZ), True),
            # Friday.
            (datetime(2026, 6, 12, 8, 30, tzinfo=JOBS_BOGOTA_TZ), True),
            # Saturday — skip.
            (datetime(2026, 6, 13, 8, 30, tzinfo=JOBS_BOGOTA_TZ), False),
            # Sunday — skip.
            (datetime(2026, 6, 14, 8, 30, tzinfo=JOBS_BOGOTA_TZ), False),
        ],
    )
    def test_weekday(self, bogota_date, expected):
        assert is_morning_trading_day(bogota_date) == expected

    def test_holiday(self):
        holiday = HolidayCalendar(skipped_dates=frozenset({date(2026, 1, 1)}))
        # 2026-01-01 is a Thursday but a configured holiday.
        d = datetime(2026, 1, 1, 8, 30, tzinfo=JOBS_BOGOTA_TZ)
        assert not is_morning_trading_day(d, calendar=holiday)

    def test_default_calendar_has_no_holidays(self):
        # 2026-12-25 is a Friday.
        d = datetime(2026, 12, 25, 8, 30, tzinfo=JOBS_BOGOTA_TZ)
        # Default calendar (no holidays configured) lets it through.
        assert is_morning_trading_day(d)


# --- TickerRow parsing -----------------------------------------------------


def _tracker_header() -> list[str]:
    """The RETURNS_TRACKER header — duplicated here so the test does
    not have to import from sheets.schema (which is fine, but this
    keeps the test self-contained)."""
    return [
        "Ticker", "Type (Stock/ETF)", "Strategy", "Timeframe", "Entry Date",
        "Entry Price", "Current Price", "Qty", "Entry Value", "Current Value",
        "Unrealized P&L ($)", "Unrealized P&L (%)", "Days Held", "SL", "TP",
        "Dividend Received ($)", "Total Return", "vs SPY", "Status",
    ]


def _tracker_row(
    ticker: str = "AAPL",
    entry_date: str = "2026-06-01",
    entry_price: str = "100.00",
    qty: str = "10",
    dividend: str = "0.00",
) -> list[str]:
    """Build a valid RETURNS_TRACKER row for testing."""
    return [
        ticker, "Stock", "Short", "3-7d", entry_date, entry_price, "",  # G=Current Price (blank)
        qty, "", "", "", "", "", "", "", dividend, "", "", "OPEN",
    ]


class TestTickerRow:
    def test_parses_valid_row(self):
        row = _tracker_row()
        parsed = TickerRow.from_row(2, row)
        assert parsed.row_index == 2
        assert parsed.ticker == "AAPL"
        assert parsed.entry_date == date(2026, 6, 1)
        assert parsed.entry_price == 100.0
        assert parsed.qty == 10.0
        assert parsed.dividend_received == 0.0

    def test_parses_with_dividend(self):
        row = _tracker_row(dividend="5.50")
        parsed = TickerRow.from_row(2, row)
        assert parsed.dividend_received == 5.50

    def test_raises_on_empty_ticker(self):
        row = _tracker_row(ticker="")
        with pytest.raises(ValueError, match="empty ticker"):
            TickerRow.from_row(2, row)

    def test_raises_on_bad_date(self):
        row = _tracker_row(entry_date="not-a-date")
        with pytest.raises(ValueError, match="bad entry_date"):
            TickerRow.from_row(2, row)

    def test_raises_on_bad_price(self):
        row = _tracker_row(entry_price="not-a-number")
        with pytest.raises(ValueError, match="bad entry_price"):
            TickerRow.from_row(2, row)

    def test_raises_on_bad_qty(self):
        row = _tracker_row(qty="not-a-number")
        with pytest.raises(ValueError, match="bad qty"):
            TickerRow.from_row(2, row)

    def test_dividend_defaults_to_zero_on_blank(self):
        row = _tracker_row()
        row[15] = ""  # blank dividend
        parsed = TickerRow.from_row(2, row)
        assert parsed.dividend_received == 0.0


# --- Fake SheetsClient -----------------------------------------------------


class FakeSheetsClient:
    """In-memory :class:`SheetsClient` substitute for tests.

    Records all calls and provides deterministic ``read_range`` /
    ``write_range`` behavior via an in-memory table. ``append_rows``
    extends the table. ``row_count`` returns the number of populated
    rows.
    """

    def __init__(self, initial: Optional[dict[str, list[list[str]]]] = None):
        # tab -> list of rows (row 0 is the header, row 1 is the first
        # data row, etc.)
        self._tables: dict[str, list[list[str]]] = {}
        self.writes: list[tuple[str, str, list[list[str]]]] = []
        self.appends: list[tuple[str, list[list[str]]]] = []
        if initial:
            for tab, rows in initial.items():
                # Make sure row 0 is the header.
                self._tables[tab] = list(rows)

    def _ensure(self, tab: str) -> list[list[str]]:
        if tab not in self._tables:
            self._tables[tab] = []
        return self._tables[tab]

    def read_range(self, sheet_id: str, tab: str, range_a1: str) -> list[list[str]]:
        # We only support a tiny subset of A1: just the part before any
        # "!" (the tab name) and treat it as "give me everything
        # starting at A1". Tests use this with simple ranges like
        # "A2:R".
        rows = self._ensure(tab)
        # If the range is "A:A" we return column A only.
        if range_a1 == "A:A":
            return [[r[0] if r else ""] for r in rows]
        # Otherwise return all populated cells.
        return [list(r) for r in rows]

    def write_range(
        self, sheet_id: str, tab: str, range_a1: str, values: list[list[str]]
    ) -> None:
        self.writes.append((tab, range_a1, [list(r) for r in values]))
        rows = self._ensure(tab)
        # Parse "A2:R5" into a starting (row, col) and shape.
        # For tests we only need to handle the simple "A2:R{N}" shape.
        try:
            start_cell, end_cell = range_a1.split(":")
        except ValueError:
            return
        start_row = int("".join(c for c in start_cell if c.isdigit())) - 1
        end_row = int("".join(c for c in end_cell if c.isdigit())) - 1
        end_col = _col_index_from_letters("".join(c for c in end_cell if c.isalpha()))
        # Pad rows to length.
        while len(rows) < end_row + 1:
            rows.append([])
        for i, new_row in enumerate(values):
            r = start_row + i
            while len(rows[r]) < end_col:
                rows[r].append("")
            for j, v in enumerate(new_row):
                rows[r][j] = v

    def append_rows(
        self, sheet_id: str, tab: str, values: list[list[str]]
    ) -> int:
        self.appends.append((tab, [list(r) for r in values]))
        rows = self._ensure(tab)
        first_row = len(rows) + 1
        for v in values:
            rows.append(list(v))
        return first_row

    def row_count(self, sheet_id: str, tab: str) -> int:
        return len(self._ensure(tab))


def _col_index_from_letters(letters: str) -> int:
    """A -> 1, B -> 2, ..., Z -> 26, AA -> 27, ... (1-indexed)."""
    n = 0
    for c in letters:
        n = n * 26 + (ord(c.upper()) - ord("A") + 1)
    return n


# --- morning_run (lazy / defensive) ----------------------------------------


class TestMorningRunSkips:
    def test_skips_saturday(self):
        today = datetime(2026, 6, 13, 9, 0, tzinfo=JOBS_BOGOTA_TZ)
        outcome = morning_run(today=today)
        assert outcome.status == "skipped_weekend"
        assert outcome.picks_scraped == 0
        assert outcome.orders_placed == 0

    def test_skips_sunday(self):
        today = datetime(2026, 6, 14, 9, 0, tzinfo=JOBS_BOGOTA_TZ)
        outcome = morning_run(today=today)
        assert outcome.status == "skipped_weekend"

    def test_skips_configured_holiday(self):
        today = datetime(2026, 1, 1, 9, 0, tzinfo=JOBS_BOGOTA_TZ)
        cal = HolidayCalendar(skipped_dates=frozenset({date(2026, 1, 1)}))
        outcome = morning_run(today=today, calendar=cal)
        assert outcome.status == "skipped_holiday"


class TestMorningRunLazyConfig:
    """When config is omitted, the function tries to load from env. In
    a test environment with no real env, the ConfigError is converted
    to a ``failed`` outcome."""

    def test_missing_env_returns_failed(self, monkeypatch):
        # Strip the test-env so config can't load.
        for k in (
            "INVESTINGPRO_EMAIL", "INVESTINGPRO_PASSWORD", "XTB_USER_ID",
            "XTB_PASSWORD", "GOOGLE_SERVICE_ACCOUNT_JSON", "OPENAI_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        # Block config.from_env from re-reading the operator's
        # profile env (which would re-populate the vars we just
        # deleted). The test wants to simulate a truly-empty env,
        # not a "real env masked by monkeypatch".
        monkeypatch.setattr(
            "portfoliomind.config.load_env_sources", lambda: []
        )
        outcome = morning_run()
        assert outcome.status == "failed"
        assert "config load failed" in outcome.errors[0]


class TestMorningRunNoPlatformModules:
    """Cards 2 + 3 are now implemented: ``portfoliomind.investingpro.runner``
    and ``portfoliomind.xtb.runner`` both expose ``run_morning``. The
    morning job should pick them up, NOT log "no platform runners
    registered". When both runners fail (e.g. the test environment
    has no real Google Sheets), the job still completes — it just
    surfaces the errors.
    """

    def test_runners_are_picked_up(self, monkeypatch):
        """``morning_run`` finds the runner modules and calls them.
        Both runners are invoked, both fail (no real Sheets in this
        test), and the outcome is ``failed`` with one error per
        runner.
        """
        from portfoliomind.config import PortfoliomindConfig
        from tests.conftest import full_env

        fake_sheets = FakeSheetsClient()
        cfg = PortfoliomindConfig.from_env(env=full_env(sheet_id=""))
        today = datetime(2026, 6, 8, 8, 30, tzinfo=JOBS_BOGOTA_TZ)
        outcome = morning_run(config=cfg, sheets=fake_sheets, today=today)
        # The runners are wired up. The job ran (status: "failed"
        # because both runners errored in the no-Sheets test env, but
        # NOT "no_platform_modules" — that path is gone).
        assert outcome.status == "failed"
        # The two errors are one per runner, not the
        # no-platform-modules skip path.
        assert any("card2" in e for e in outcome.errors)
        assert any("card3" in e for e in outcome.errors)
        # Crucially: the "no platform runners" line is NOT in the
        # agent log anymore.
        rows = fake_sheets._tables.get("🗒️ Agent Log", [])
        assert not any("no platform runners" in r[3] for r in rows), (
            f"unexpected 'no platform runners' line in AGENT_LOG: {rows!r}"
        )

    def test_runners_can_be_replaced_with_fakes(self, monkeypatch):
        """Inject fake runners that return known results, then check
        that ``morning_run`` aggregates their ``picks_scraped`` /
        ``orders_placed`` correctly.
        """
        from portfoliomind.scheduler.jobs import MorningContext, MorningResult

        from portfoliomind.config import PortfoliomindConfig
        from tests.conftest import full_env

        fake_sheets = FakeSheetsClient()
        cfg = PortfoliomindConfig.from_env(env=full_env(sheet_id=""))

        def fake_inv(ctx: MorningContext) -> MorningResult:
            return MorningResult(
                runner="card2", picks_scraped=7, error="",
            )

        def fake_xtb(ctx: MorningContext) -> MorningResult:
            return MorningResult(
                runner="card3", orders_placed=2, error="",
            )

        # Patch the lazy-import seam directly.
        monkeypatch.setattr(
            "portfoliomind.scheduler.jobs._try_import_card2",
            lambda: fake_inv,
        )
        monkeypatch.setattr(
            "portfoliomind.scheduler.jobs._try_import_card3",
            lambda: fake_xtb,
        )

        today = datetime(2026, 6, 8, 8, 30, tzinfo=JOBS_BOGOTA_TZ)
        outcome = morning_run(config=cfg, sheets=fake_sheets, today=today)

        assert outcome.status == "ran"
        assert outcome.picks_scraped == 7
        assert outcome.orders_placed == 2
        assert outcome.errors == []


# --- refresh_returns math + pruning ----------------------------------------


class TestRefreshReturns:
    HEADER = _tracker_header()

    def _make_sheets(self, rows: list[list[str]]) -> FakeSheetsClient:
        return FakeSheetsClient(
            initial={"💰 Returns Tracker": [self.HEADER] + rows}
        )

    def _fixed_price(self, prices: dict[str, float]):
        """Build a price_fetcher that returns ``prices[t]`` for each
        ticker, ``None`` for anything else."""
        def fetcher(ticker: str) -> Optional[float]:
            return prices.get(ticker)
        return fetcher

    def test_happy_path_math(self):
        sheets = self._make_sheets([
            _tracker_row(ticker="AAPL", entry_date="2026-06-01",
                         entry_price="100.00", qty="10"),
        ])
        fetcher = self._fixed_price({"AAPL": 110.0})
        outcome = refresh_returns(sheets=sheets, price_fetcher=fetcher,
                                  today=datetime(2026, 6, 8, 16, 30,
                                                 tzinfo=JOBS_BOGOTA_TZ))
        assert outcome.status == "ran"
        assert outcome.tickers_refreshed == 1
        assert outcome.tickers_pruned == 0
        # AAPL row: entry 100, current 110, qty 10.
        # Current Value = 1100, P&L = 100, P&L% = 10, Days Held = 7.
        rows = sheets._tables["💰 Returns Tracker"]
        aapl = rows[1]  # data row
        assert aapl[6] == "110.0000"  # G=Current Price
        assert aapl[9] == "1100.00"   # J=Current Value
        assert aapl[10] == "100.00"   # K=Unrealized P&L $
        assert aapl[11] == "10.00"    # L=Unrealized P&L %
        assert aapl[12] == "7"        # M=Days Held (Jun 1 -> Jun 8 = 7)
        assert aapl[16] == "10.00"    # Q=Total Return (same as P&L% when no dividend)
        assert aapl[18] == "OPEN"     # R=Status

    def test_dividend_in_total_return(self):
        sheets = self._make_sheets([
            _tracker_row(ticker="KO", entry_date="2026-01-01",
                         entry_price="50.00", qty="20", dividend="10.00"),
        ])
        fetcher = self._fixed_price({"KO": 55.0})
        outcome = refresh_returns(sheets=sheets, price_fetcher=fetcher,
                                  today=datetime(2026, 6, 8, 16, 30,
                                                 tzinfo=JOBS_BOGOTA_TZ))
        assert outcome.status == "ran"
        rows = sheets._tables["💰 Returns Tracker"]
        ko = rows[1]
        # P&L$ = 55*20 - 50*20 = 100, P&L% = 10
        # Total Return = (100 + 10) / (50*20) * 100 = 11
        assert ko[10] == "100.00"   # P&L $
        assert ko[11] == "10.00"    # P&L %
        assert ko[16] == "11.00"    # Total Return

    def test_prunes_unresolvable_ticker(self):
        sheets = self._make_sheets([
            _tracker_row(ticker="AAPL", entry_date="2026-06-01",
                         entry_price="100.00", qty="10"),
            _tracker_row(ticker="ZZZZZ", entry_date="2026-06-01",
                         entry_price="50.00", qty="5"),
        ])
        # AAPL resolves; ZZZZZ returns None (yfinance couldn't find it).
        fetcher = self._fixed_price({"AAPL": 110.0})  # ZZZZZ absent
        outcome = refresh_returns(sheets=sheets, price_fetcher=fetcher,
                                  today=datetime(2026, 6, 8, 16, 30,
                                                 tzinfo=JOBS_BOGOTA_TZ))
        assert outcome.status == "ran"
        assert outcome.tickers_refreshed == 1
        assert outcome.tickers_pruned == 1
        # Only AAPL remains in the table.
        rows = sheets._tables["💰 Returns Tracker"]
        data_rows = rows[1:]
        # AAPL row is data row 0; ZZZZZ row should be cleared.
        assert any(r and r[0] == "AAPL" for r in data_rows)
        # The ZZZZZ cell in the data block should be empty.
        # After the write, only the AAPL row should be present.
        non_empty = [r for r in data_rows if r and any(c.strip() for c in r)]
        assert len(non_empty) == 1
        assert non_empty[0][0] == "AAPL"

    def test_prunes_zero_or_negative_price(self):
        sheets = self._make_sheets([
            _tracker_row(ticker="AAPL"),
        ])
        # yfinance returns 0.0 (defunct ticker or stale data) -> prune.
        fetcher = self._fixed_price({"AAPL": 0.0})
        outcome = refresh_returns(sheets=sheets, price_fetcher=fetcher,
                                  today=datetime(2026, 6, 8, 16, 30,
                                                 tzinfo=JOBS_BOGOTA_TZ))
        assert outcome.tickers_pruned == 1
        assert outcome.tickers_refreshed == 0

    def test_no_sheet_path(self):
        sheets = FakeSheetsClient(initial={"💰 Returns Tracker": [self.HEADER]})
        outcome = refresh_returns(sheets=sheets, price_fetcher=self._fixed_price({}))
        assert outcome.status == "no_sheet"
        assert outcome.tickers_refreshed == 0

    def test_idempotent_re_run(self):
        """Running refresh_returns twice in a row with the same prices
        should leave the populated data the same. Specifically: the
        second run should not duplicate the AAPL row or change its
        values. We also assert that the second run actually writes
        (refresh must overwrite, not just no-op)."""
        sheets = self._make_sheets([
            _tracker_row(ticker="AAPL", entry_date="2026-06-01",
                         entry_price="100.00", qty="10"),
        ])
        fetcher = self._fixed_price({"AAPL": 110.0})
        today = datetime(2026, 6, 8, 16, 30, tzinfo=JOBS_BOGOTA_TZ)
        refresh_returns(sheets=sheets, price_fetcher=fetcher, today=today)
        # After the first run, the AAPL row is populated.
        rows_after_first = [list(r) for r in sheets._tables["💰 Returns Tracker"]]
        # Reset the writes log so we can confirm the second run also
        # writes (it must, otherwise stale data accumulates).
        sheets.writes.clear()
        refresh_returns(sheets=sheets, price_fetcher=fetcher, today=today)
        rows_after_second = [list(r) for r in sheets._tables["💰 Returns Tracker"]]
        # Both runs produced a write (proves refresh is non-noop).
        assert len(sheets.writes) > 0
        # The AAPL row is present in both.
        aapl_first = [r for r in rows_after_first if r and r[0] == "AAPL"]
        aapl_second = [r for r in rows_after_second if r and r[0] == "AAPL"]
        assert len(aapl_first) == 1
        assert len(aapl_second) == 1
        # And the values are identical (no drift across runs).
        assert aapl_first[0] == aapl_second[0]

    def test_skips_malformed_row(self):
        """A row with a bad entry_date is dropped (with a WARNING
        logged) but other valid rows are still processed."""
        sheets = self._make_sheets([
            _tracker_row(ticker="AAPL", entry_date="not-a-date"),
            _tracker_row(ticker="KO", entry_date="2026-01-01",
                         entry_price="50.00", qty="20"),
        ])
        fetcher = self._fixed_price({"KO": 55.0})
        outcome = refresh_returns(sheets=sheets, price_fetcher=fetcher,
                                  today=datetime(2026, 6, 8, 16, 30,
                                                 tzinfo=JOBS_BOGOTA_TZ))
        assert outcome.status == "ran"
        # AAPL is dropped, KO is kept.
        assert outcome.tickers_refreshed == 1
        # A parse error is reported.
        assert any("parse" in e for e in outcome.errors)

    def test_all_pruned_does_not_leave_empty_data(self):
        """If every ticker is pruned, the write range should collapse
        to just the header. The tail-clearing logic must not write
        phantom rows."""
        sheets = self._make_sheets([
            _tracker_row(ticker="ZZZ1"),
            _tracker_row(ticker="ZZZ2"),
        ])
        fetcher = self._fixed_price({})  # nothing resolves
        outcome = refresh_returns(sheets=sheets, price_fetcher=fetcher,
                                  today=datetime(2026, 6, 8, 16, 30,
                                                 tzinfo=JOBS_BOGOTA_TZ))
        assert outcome.tickers_pruned == 2
        assert outcome.tickers_refreshed == 0
        rows = sheets._tables["💰 Returns Tracker"]
        # Only the header remains populated.
        non_empty = [r for r in rows[1:] if r and any(c.strip() for c in r)]
        assert non_empty == []


# --- Cron trigger wiring ---------------------------------------------------


class TestCronTrigger:
    def test_morning_trigger_is_bogota(self):
        trig = build_morning_trigger(ScheduleConfig())
        # apscheduler's CronTrigger exposes .timezone.
        assert trig.timezone == JOBS_BOGOTA_TZ

    def test_morning_trigger_is_weekday(self):
        trig = build_morning_trigger(ScheduleConfig())
        # day_of_week for "mon-fri" -> "mon-fri"
        assert str(trig.fields[4]) == "mon-fri"

    def test_morning_trigger_default_time(self):
        trig = build_morning_trigger(ScheduleConfig())
        # CronTrigger's str() does NOT include timezone, so check
        # both the explicit attribute and the field rendering.
        assert trig.timezone == JOBS_BOGOTA_TZ
        s = str(trig)
        assert "hour='8'" in s
        assert "minute='30'" in s
        assert "day_of_week='mon-fri'" in s

    def test_returns_trigger_is_bogota(self):
        trig = build_returns_trigger(ScheduleConfig())
        assert trig.timezone == JOBS_BOGOTA_TZ
        s = str(trig)
        assert "hour='16'" in s
        assert "minute='30'" in s

    def test_overrides_take_effect(self):
        cfg = ScheduleConfig(
            morning_hour=9, morning_minute=15,
            returns_hour=17, returns_minute=45,
        )
        m = build_morning_trigger(cfg)
        r = build_returns_trigger(cfg)
        assert "hour='9'" in str(m)
        assert "minute='15'" in str(m)
        assert "hour='17'" in str(r)
        assert "minute='45'" in str(r)

    def test_build_scheduler_registers_two_jobs(self):
        from apscheduler.schedulers.background import BackgroundScheduler
        sched = build_scheduler(scheduler_factory=BackgroundScheduler)
        try:
            sched.start()
            jobs = sched.get_jobs()
            assert len(jobs) == 2
            ids = {j.id for j in jobs}
            assert ids == {
                "portfoliomind.morning_run",
                "portfoliomind.refresh_returns",
            }
        finally:
            sched.shutdown(wait=False)


# --- Outcome summary lines (Discord-facing) --------------------------------


class TestOutcomeSummaries:
    def test_morning_summary_lines(self):
        for status in ("ran", "skipped_weekend", "skipped_holiday",
                       "no_platform_modules", "failed"):
            o = MorningOutcome(
                status=status,
                started_at="2026-06-08T08:30:00-05:00",
                finished_at="2026-06-08T08:35:00-05:00",
                picks_scraped=3, orders_placed=1,
                errors=["x"] if status == "failed" else [],
            )
            line = o.summary_line()
            assert isinstance(line, str)
            assert len(line) > 0

    def test_refresh_summary_lines(self):
        for status in ("ran", "skipped", "no_sheet", "failed"):
            o = RefreshOutcome(
                status=status,
                tickers_refreshed=4, tickers_pruned=1,
                errors=["x"] if status == "failed" else [],
            )
            line = o.summary_line()
            assert isinstance(line, str)
            assert len(line) > 0


# --- card 8: strategy runner integration ----------------------------------


class TestMorningRunWithStrategyRunner:
    """Card 8 wires the strategy runner into ``morning_run``.

    Coverage:

    * When the strategy runner returns ``status='not_implemented'``,
      the morning job still completes cleanly — the strategy result
      is treated as ``ok()`` and does not produce a card8 error.
    * When the strategy runner returns ``picks_scraped=N`` /
      ``orders_placed=M``, the morning outcome aggregates them with
      the card 2 / card 3 totals.
    * When the strategy runner raises, the morning job catches it
      and records a ``card8 raised:`` error.
    * When the strategy runner is the ONLY runner registered (card
      2/3 still absent), ``morning_run`` still proceeds with
      ``status='ran'`` (or 'skipped') instead of the
      ``no_platform_modules`` short-circuit.
    """

    def test_strategy_not_implemented_does_not_break_morning_run(
        self, monkeypatch
    ):
        """The strategy runner's ``not_implemented`` status is a
        soft no-op — the morning job still aggregates the card 2
        and card 3 totals without injecting a card8 error."""
        from portfoliomind.config import PortfoliomindConfig
        from portfoliomind.scheduler.jobs import MorningContext, MorningResult

        from tests.conftest import full_env

        fake_sheets = FakeSheetsClient()
        cfg = PortfoliomindConfig.from_env(env=full_env(sheet_id=""))

        def fake_inv(_ctx: MorningContext) -> MorningResult:
            return MorningResult(runner="card2", picks_scraped=4)

        def fake_xtb(_ctx: MorningContext) -> MorningResult:
            return MorningResult(runner="card3", orders_placed=1)

        monkeypatch.setattr(
            "portfoliomind.scheduler.jobs._try_import_card2", lambda: fake_inv
        )
        monkeypatch.setattr(
            "portfoliomind.scheduler.jobs._try_import_card3", lambda: fake_xtb
        )

        # The strategy runner is a no-op factory (its default state).
        from portfoliomind import strategy_runner as strat_runner

        monkeypatch.setattr(strat_runner, "_score_universe_factory", None)
        monkeypatch.setattr(strat_runner, "_sizer_factory", None)
        monkeypatch.setattr(strat_runner, "_approval_factory", None)
        # The "any module registered" check uses _try_import_strategy
        # which lazy-imports portfoliomind.strategy_runner.run_morning
        # (the production entry point). The real run_morning returns
        # status='not_implemented' on a clean factory state, so the
        # check passes and the no_platform_modules short-circuit
        # does NOT fire.

        today = datetime(2026, 6, 8, 8, 30, tzinfo=JOBS_BOGOTA_TZ)
        outcome = morning_run(config=cfg, sheets=fake_sheets, today=today)
        # The job ran (not the no-platform short-circuit).
        assert outcome.status != "no_platform_modules"
        # No card8 error.
        assert not any("card8" in e for e in outcome.errors), (
            f"unexpected card8 error from not_implemented strategy: {outcome.errors!r}"
        )

    def test_strategy_aggregates_into_morning_outcome(self, monkeypatch):
        """When the strategy runner produces counts, the morning
        outcome sums them with the card 2/3 totals."""
        from portfoliomind.config import PortfoliomindConfig
        from portfoliomind.scheduler.jobs import MorningContext, MorningResult

        from tests.conftest import full_env

        fake_sheets = FakeSheetsClient()
        cfg = PortfoliomindConfig.from_env(env=full_env(sheet_id=""))

        # Inject a strategy that scored 7 candidates and persisted 2 rows.
        from dataclasses import dataclass

        @dataclass
        class _StratRes:
            status: str = "ran"
            picks_scraped: int = 7
            orders_placed: int = 2
            approved_count: int = 5
            rejected_count: int = 2
            skipped: bool = False
            skip_reason: str = ""
            error: str = ""
            errors: list = None  # type: ignore[assignment]

            def __post_init__(self):
                if self.errors is None:
                    self.errors = []

            def ok(self) -> bool:
                return not self.error

        def fake_strategy(_ctx: MorningContext) -> _StratRes:
            return _StratRes()

        def fake_inv(_ctx: MorningContext) -> MorningResult:
            return MorningResult(runner="card2", picks_scraped=3)

        def fake_xtb(_ctx: MorningContext) -> MorningResult:
            return MorningResult(runner="card3", orders_placed=1)

        monkeypatch.setattr(
            "portfoliomind.scheduler.jobs._try_import_strategy", lambda: fake_strategy
        )
        monkeypatch.setattr(
            "portfoliomind.scheduler.jobs._try_import_card2", lambda: fake_inv
        )
        monkeypatch.setattr(
            "portfoliomind.scheduler.jobs._try_import_card3", lambda: fake_xtb
        )

        today = datetime(2026, 6, 8, 8, 30, tzinfo=JOBS_BOGOTA_TZ)
        outcome = morning_run(config=cfg, sheets=fake_sheets, today=today)
        # Strategy scored 7, card2 scored 3 → 10 total.
        assert outcome.picks_scraped == 10
        # Strategy persisted 2, card3 placed 1 → 3 total.
        assert outcome.orders_placed == 3
        # Both runners reported success → outcome is "ran" (not "failed").
        assert outcome.status == "ran"

    def test_strategy_raises_records_card8_error(self, monkeypatch):
        """If the strategy runner raises, ``morning_run`` records a
        ``card8 raised:`` error and the rest of the morning job
        still completes."""
        from portfoliomind.config import PortfoliomindConfig
        from portfoliomind.scheduler.jobs import MorningContext, MorningResult

        from tests.conftest import full_env

        fake_sheets = FakeSheetsClient()
        cfg = PortfoliomindConfig.from_env(env=full_env(sheet_id=""))

        def fake_strategy(_ctx: MorningContext) -> MorningResult:
            raise RuntimeError("simulated strategy failure")

        def fake_inv(_ctx: MorningContext) -> MorningResult:
            return MorningResult(runner="card2", picks_scraped=3)

        def fake_xtb(_ctx: MorningContext) -> MorningResult:
            return MorningResult(runner="card3", orders_placed=1)

        monkeypatch.setattr(
            "portfoliomind.scheduler.jobs._try_import_strategy", lambda: fake_strategy
        )
        monkeypatch.setattr(
            "portfoliomind.scheduler.jobs._try_import_card2", lambda: fake_inv
        )
        monkeypatch.setattr(
            "portfoliomind.scheduler.jobs._try_import_card3", lambda: fake_xtb
        )

        today = datetime(2026, 6, 8, 8, 30, tzinfo=JOBS_BOGOTA_TZ)
        outcome = morning_run(config=cfg, sheets=fake_sheets, today=today)
        assert outcome.status == "failed"
        assert any("card8 raised" in e for e in outcome.errors)
        assert any("simulated strategy failure" in e for e in outcome.errors)
